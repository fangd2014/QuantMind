from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import create_engine as sa_create_engine
from sqlalchemy import text as sa_text
from sqlalchemy.orm import sessionmaker as sa_sessionmaker

from backend.services.engine.qlib_app.celery_config import celery_app
from backend.services.engine.services.model_inference_persistence import (
    model_inference_persistence,
)
from backend.services.engine.services.pipeline_service import PipelineService
from backend.services.engine.services.strategy_loop_persistence import (
    StrategyLoopPersistence,
)
from backend.shared.database_manager_v2 import get_session
from backend.shared.redis_sentinel_client import get_redis_sentinel_client

logger = logging.getLogger(__name__)

# 与 model_management.py 保持一致的锁配置
_INFERENCE_LOCK_KEY_PREFIX = "qm:lock:inference:daily"
_INFERENCE_LOCK_TTL_SEC = 1800  # 30 分钟


def _run_async(coro: Any) -> Any:
    return asyncio.run(coro)


def _get_sync_db_session():
    sync_db_url = str(os.getenv("DATABASE_URL", "")).strip()
    if "+asyncpg" in sync_db_url:
        sync_db_url = sync_db_url.replace("+asyncpg", "+psycopg2")
    if not sync_db_url or "postgresql" not in sync_db_url:
        sync_db_url = "postgresql+psycopg2://postgres:quantmind2026@quantmind-postgresql:5432/quantmind"

    sync_engine = sa_create_engine(sync_db_url, pool_pre_ping=True)
    session_factory = sa_sessionmaker(bind=sync_engine)
    return session_factory()


def _ensure_dispatch_log_table(db) -> None:
    db.execute(
        sa_text(
            """
            CREATE TABLE IF NOT EXISTS qm_model_inference_dispatch_logs (
              id BIGSERIAL PRIMARY KEY,
              trigger_source TEXT NOT NULL,
              tenant_id TEXT NOT NULL,
              user_id TEXT NOT NULL,
              strategy_id TEXT,
              model_id TEXT,
              data_trade_date DATE,
              prediction_trade_date DATE,
              status TEXT NOT NULL,
              reason_code TEXT,
              reason_detail TEXT,
              run_id TEXT,
              created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
    )
    db.execute(
        sa_text(
            """
            CREATE INDEX IF NOT EXISTS idx_qm_inf_dispatch_owner_created
              ON qm_model_inference_dispatch_logs (tenant_id, user_id, created_at DESC);
            """
        )
    )
    db.execute(
        sa_text(
            """
            CREATE INDEX IF NOT EXISTS idx_qm_inf_dispatch_status_created
              ON qm_model_inference_dispatch_logs (status, created_at DESC);
            """
        )
    )
    db.commit()


def _write_dispatch_log(
    db,
    *,
    trigger_source: str,
    tenant_id: str,
    user_id: str,
    strategy_id: str | None,
    model_id: str | None,
    data_trade_date: str,
    prediction_trade_date: str,
    status: str,
    reason_code: str | None = None,
    reason_detail: str | None = None,
    run_id: str | None = None,
) -> None:
    db.execute(
        sa_text(
            """
            INSERT INTO qm_model_inference_dispatch_logs (
              trigger_source, tenant_id, user_id, strategy_id, model_id,
              data_trade_date, prediction_trade_date,
              status, reason_code, reason_detail, run_id, created_at
            ) VALUES (
              :trigger_source, :tenant_id, :user_id, :strategy_id, :model_id,
              :data_trade_date, :prediction_trade_date,
              :status, :reason_code, :reason_detail, :run_id, NOW()
            )
            """
        ),
        {
            "trigger_source": trigger_source,
            "tenant_id": str(tenant_id or "default"),
            "user_id": str(user_id or ""),
            "strategy_id": strategy_id,
            "model_id": model_id,
            "data_trade_date": data_trade_date,
            "prediction_trade_date": prediction_trade_date,
            "status": status,
            "reason_code": reason_code,
            "reason_detail": reason_detail,
            "run_id": run_id,
        },
    )
    db.commit()


def _try_acquire_strategy_lock(strategy_id: str, trade_date: str, owner: str) -> bool:
    """尝试获取特定策略当日推理分布式锁。"""
    try:
        from backend.shared.redis_sentinel_client import get_redis_sentinel_client

        redis = get_redis_sentinel_client()
        lock_key = f"{_INFERENCE_LOCK_KEY_PREFIX}:{strategy_id}:{trade_date}"
        return bool(redis.set(lock_key, owner, ex=_INFERENCE_LOCK_TTL_SEC, nx=True))
    except Exception as e:
        logger.warning("[InferenceLock] Redis 不可用，跳过策略锁检查: %s", e)
        return True


from backend.services.engine.services.signal_generator import global_signal_generator


@celery_app.task(
    name="engine.tasks.generate_global_signals",
    max_retries=3,
    default_retry_delay=60,
)
def generate_global_signals(
    universe: str = "all", mock: bool = False
) -> dict[str, Any]:
    """Celery 任务：生成全市场 Alpha 预测信号 (10万并发架构核心)。"""
    # 计算上海时区当日日期作为锁键（与 Admin 手动触发共享同一把锁）
    from zoneinfo import ZoneInfo

    trade_date = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()

    acquired = _try_acquire_strategy_lock("global", trade_date, owner="celery_beat")
    if not acquired:
        logger.warning(
            "[InferenceLock] 当日全局推理锁已被占用（date=%s），Celery Beat 本次跳过。",
            trade_date,
        )
        return {"status": "skipped", "reason": "lock_held", "trade_date": trade_date}

    logger.info(
        "Global signal generation task started: universe=%s trade_date=%s",
        universe,
        trade_date,
    )
    result = _run_async(
        global_signal_generator.generate_and_broadcast(universe, mock=mock)
    )
    return {
        "status": "success" if result else "failed",
        "universe": universe,
        "trade_date": trade_date,
    }


@celery_app.task(
    bind=True,
    name="engine.tasks.run_pipeline_run",
    max_retries=1,
    default_retry_delay=30,
)
def run_pipeline_run(self, run_id: str) -> dict[str, Any]:
    """Celery 任务：执行 pipeline run。"""
    logger.info(
        "Pipeline Celery task started: run_id=%s task_id=%s", run_id, self.request.id
    )
    service = PipelineService()
    result = _run_async(service.execute_run(run_id))
    return result.model_dump()


@celery_app.task(
    name="engine.tasks.auto_inference_if_needed",
    max_retries=1,
    default_retry_delay=60,
)
def auto_inference_if_needed() -> dict[str, Any]:
    """
    Celery Beat 定时任务：交易日 04:00 扫描到期的自动推理任务并分发。

    逻辑：
    1. 先确认当前自然日是否为交易日，非交易日直接跳过。
    2. 获取所有处于 'running' 状态且绑定了策略的投资组合。
    3. 获取 `next_run_at` 已到的用户自动推理设置。
    4. 将每个待执行项拆分为独立 Celery 子任务入队。
    """
    from zoneinfo import ZoneInfo
    import exchange_calendars as xcals

    now_local = datetime.now(ZoneInfo("Asia/Shanghai"))
    cal = xcals.get_calendar("XSHG")

    # 0. 先确认今天是否为交易日，节假日/周末直接跳过，避免“回看上一交易日”误触发
    try:
        trigger_date = now_local.date()
        if not cal.is_session(trigger_date):
            logger.info("[AutoInference] 今日非交易日，跳过。date=%s", trigger_date)
            return {"status": "skipped", "reason": "not_a_trading_day"}
    except Exception as e:
        logger.warning("[AutoInference] 日历检查异常: %s", e)

    # 确定特征日期 (data_trade_date) 和预测日期 (prediction_trade_date)
    # 04:00 运行时，使用上一交易日数据，生成当日可消费批次。
    data_trade_date_obj = cal.previous_session(now_local.date()).date()
    data_trade_date = data_trade_date_obj.isoformat()
    prediction_trade_date = now_local.date().isoformat()

    db = _get_sync_db_session()

    # 2. 扫描活跃策略 + 用户自动推理设置
    try:
        _ensure_dispatch_log_table(db)

        # 查询所有活跃且绑定了策略的组合
        active_portfolios = db.execute(
            sa_text(
                "SELECT id, tenant_id, user_id, strategy_id FROM portfolios "
                "WHERE run_status = 'running' AND strategy_id IS NOT NULL AND is_deleted = False"
            )
        ).all()

        # 查询已到期的用户自动推理设置（enabled=True 且 next_run_at 已到）
        auto_inference_settings = db.execute(
            sa_text(
                "SELECT tenant_id, user_id, model_id, schedule_time, next_run_at "
                "FROM qm_model_inference_settings "
                "WHERE enabled = TRUE "
                "  AND (next_run_at IS NULL OR next_run_at <= :now_local)"
            ),
            {"now_local": now_local},
        ).all()

        # 总是包含一个系统级别的虚拟任务（全局默认模型）
        tasks = [
            {
                "tenant_id": "default",
                "user_id": "system",
                "strategy_id": "global",
                "model_id": None,
            }
        ]
        for p in active_portfolios:
            tasks.append(
                {
                    "tenant_id": p.tenant_id,
                    "user_id": str(p.user_id),
                    "strategy_id": str(p.strategy_id or "default"),
                    "model_id": None,
                }
            )

        # 添加用户自动推理设置任务
        for s in auto_inference_settings:
            tasks.append(
                {
                    "tenant_id": s.tenant_id,
                    "user_id": str(s.user_id),
                    "strategy_id": None,  # 用户级别推理，不绑定特定策略
                    "model_id": str(s.model_id),
                }
            )

        # 去重（同一 tenant_id + user_id 可能同时出现在 portfolios 和 settings 中）
        seen = set()
        unique_tasks = []
        for t in tasks:
            key = (
                t["tenant_id"],
                t["user_id"],
                t.get("strategy_id"),
                t.get("model_id"),
            )
            if key not in seen:
                seen.add(key)
                unique_tasks.append(t)
        tasks = unique_tasks

        logger.info(
            "[AutoInference] 发现待分发任务总数: %d (活跃策略=%d, 自动推理设置=%d)",
            len(tasks),
            len(active_portfolios),
            len(auto_inference_settings),
        )

        dispatched = []

        # 3. 逐条分发为独立 Celery 子任务
        for task in tasks:
            child = run_auto_inference_task.apply_async(
                kwargs={
                    "tenant_id": str(task["tenant_id"]),
                    "user_id": str(task["user_id"]),
                    "strategy_id": task.get("strategy_id"),
                    "model_id": task.get("model_id"),
                    "data_trade_date": data_trade_date,
                    "prediction_trade_date": prediction_trade_date,
                    "trigger_source": "celery_auto_inference_if_needed",
                }
            )
            _write_dispatch_log(
                db,
                trigger_source="celery_auto_inference_if_needed",
                tenant_id=str(task["tenant_id"]),
                user_id=str(task["user_id"]),
                strategy_id=task.get("strategy_id"),
                model_id=task.get("model_id"),
                data_trade_date=data_trade_date,
                prediction_trade_date=prediction_trade_date,
                status="dispatched",
                reason_code="QUEUED",
                reason_detail=f"celery_task_id={child.id}",
                run_id=None,
            )
            dispatched.append(
                {
                    "tenant_id": str(task["tenant_id"]),
                    "user_id": str(task["user_id"]),
                    "strategy_id": task.get("strategy_id"),
                    "model_id": task.get("model_id"),
                    "celery_task_id": child.id,
                }
            )

        return {
            "status": "completed",
            "date": prediction_trade_date,
            "dispatched_count": len(dispatched),
            "details": dispatched,
        }
    except Exception as e:
        logger.exception("[AutoInference] 扫描/执行任务异常: %s", e)
        return {"status": "failed", "error": str(e)}
    finally:
        db.close()


@celery_app.task(
    bind=True,
    name="engine.tasks.run_auto_inference_task",
    max_retries=1,
    default_retry_delay=60,
)
def run_auto_inference_task(
    self,
    *,
    tenant_id: str,
    user_id: str,
    strategy_id: str | None,
    model_id: str | None,
    data_trade_date: str,
    prediction_trade_date: str,
    trigger_source: str = "celery_auto_inference_if_needed",
) -> dict[str, Any]:
    from backend.services.engine.inference.router_service import InferenceRouterService

    db = _get_sync_db_session()
    _ensure_dispatch_log_table(db)
    redis = None
    try:
        exists = db.execute(
            sa_text(
                "SELECT 1 FROM engine_feature_runs "
                "WHERE trade_date = :d AND status = 'signal_ready' "
                "AND tenant_id = :tid AND user_id = :uid LIMIT 1"
            ),
            {"d": prediction_trade_date, "tid": tenant_id, "uid": user_id},
        ).first()
        if exists:
            _write_dispatch_log(
                db,
                trigger_source=trigger_source,
                tenant_id=tenant_id,
                user_id=user_id,
                strategy_id=strategy_id,
                model_id=model_id,
                data_trade_date=data_trade_date,
                prediction_trade_date=prediction_trade_date,
                status="skipped",
                reason_code="ALREADY_DONE",
                reason_detail="engine_feature_runs already has signal_ready record for target trade date",
            )
            return {
                "status": "skipped",
                "reason": "already_done",
                "task_id": self.request.id,
            }

        lock_scope = f"{tenant_id}:{user_id}:{strategy_id or model_id or 'default'}"
        if not _try_acquire_strategy_lock(
            lock_scope, prediction_trade_date, f"celery_auto:{self.request.id}"
        ):
            _write_dispatch_log(
                db,
                trigger_source=trigger_source,
                tenant_id=tenant_id,
                user_id=user_id,
                strategy_id=strategy_id,
                model_id=model_id,
                data_trade_date=data_trade_date,
                prediction_trade_date=prediction_trade_date,
                status="skipped",
                reason_code="LOCK_HELD",
                reason_detail=f"lock scope conflict: {lock_scope}",
            )
            return {
                "status": "skipped",
                "reason": "lock_held",
                "task_id": self.request.id,
            }

        try:
            _write_dispatch_log(
                db,
                trigger_source=trigger_source,
                tenant_id=tenant_id,
                user_id=user_id,
                strategy_id=strategy_id,
                model_id=model_id,
                data_trade_date=data_trade_date,
                prediction_trade_date=prediction_trade_date,
                status="running",
                reason_code="STARTED",
                reason_detail=f"celery_task_id={self.request.id}",
            )
            try:
                redis = get_redis_sentinel_client()
            except Exception:
                redis = None

            router_service = InferenceRouterService()
            exec_res = router_service.run_daily_inference_script(
                date=data_trade_date,
                tenant_id=tenant_id,
                user_id=user_id,
                strategy_id=None if strategy_id == "global" else strategy_id,
                model_id=model_id,
                redis_client=redis,
            )
            result_payload = {
                "success": bool(exec_res.success),
                "run_id": str(getattr(exec_res, "run_id", "") or ""),
                "status": "completed" if exec_res.success else "failed",
                "model_id": model_id,
                "active_model_id": getattr(exec_res, "active_model_id", None),
                "effective_model_id": getattr(exec_res, "effective_model_id", None),
                "model_source": getattr(exec_res, "model_source", None),
                "active_data_source": getattr(exec_res, "active_data_source", None),
                "data_trade_date": data_trade_date,
                "prediction_trade_date": prediction_trade_date,
                "signals_count": int(getattr(exec_res, "signals_count", 0) or 0),
                "duration_ms": int(getattr(exec_res, "duration_ms", 0) or 0),
                "fallback_used": bool(getattr(exec_res, "fallback_used", False)),
                "fallback_reason": getattr(exec_res, "fallback_reason", "") or "",
                "failure_stage": getattr(exec_res, "failure_stage", "") or "",
                "error_message": getattr(exec_res, "error", "") or "",
                "stdout": getattr(exec_res, "stdout", "") or "",
                "stderr": getattr(exec_res, "stderr", "") or "",
            }
            if model_id:
                _run_async(
                    model_inference_persistence.record_run_to_settings(
                        tenant_id=tenant_id,
                        user_id=user_id,
                        model_id=model_id,
                        run_payload=result_payload,
                    )
                )
            _write_dispatch_log(
                db,
                trigger_source=trigger_source,
                tenant_id=tenant_id,
                user_id=user_id,
                strategy_id=strategy_id,
                model_id=model_id,
                data_trade_date=data_trade_date,
                prediction_trade_date=prediction_trade_date,
                status="success" if exec_res.success else "failed",
                reason_code=None if exec_res.success else "EXECUTION_FAILED",
                reason_detail=(
                    None
                    if exec_res.success
                    else str(getattr(exec_res, "message", "") or "")
                ),
                run_id=getattr(exec_res, "run_id", None),
            )
            return {
                "status": "success" if exec_res.success else "failed",
                "task_id": self.request.id,
                "run_id": getattr(exec_res, "run_id", None),
                "tenant_id": tenant_id,
                "user_id": user_id,
                "model_id": model_id,
            }
        finally:
            if redis is not None:
                try:
                    lock_key = f"{_INFERENCE_LOCK_KEY_PREFIX}:{lock_scope}:{prediction_trade_date}"
                    redis.delete(lock_key)
                except Exception:
                    pass
    except Exception as exc:
        _write_dispatch_log(
            db,
            trigger_source=trigger_source,
            tenant_id=tenant_id,
            user_id=user_id,
            strategy_id=strategy_id,
            model_id=model_id,
            data_trade_date=data_trade_date,
            prediction_trade_date=prediction_trade_date,
            status="failed",
            reason_code="EXCEPTION",
            reason_detail=str(exc),
        )
        raise
    finally:
        db.close()


@celery_app.task(
    bind=True,
    name="engine.tasks.run_strategy_backtest_loop",
    max_retries=0,
)
def run_strategy_backtest_loop(
    self, task_id: str, request_payload: dict[str, Any]
) -> dict[str, Any]:
    """Celery 任务：执行策略-回测闭环。"""
    logger.info(
        "Strategy-backtest loop Celery task started: task_id=%s celery_id=%s",
        task_id,
        self.request.id,
    )

    async def _run() -> dict[str, Any]:
        from backend.services.engine.ai_strategy.api.routes.strategy_backtest_loop import (
            StrategyBacktestLoopRequest,
        )
        from backend.services.engine.ai_strategy.shared.ai_providers import (
            ComplexityLevel,
            StrategyRequest,
            StrategyType,
        )
        from backend.services.engine.ai_strategy.shared.market_data import (
            MarketDataService,
        )
        from backend.services.engine.ai_strategy.shared.strategy_backtest_loop import (
            LoopConfig,
            StrategyBacktestLoop,
        )

        persistence = StrategyLoopPersistence()
        await persistence.ensure_tables()
        await persistence.update_task(
            task_id=task_id,
            status="running",
            updated_at=datetime.now(),
        )

        request = StrategyBacktestLoopRequest(**request_payload)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)

        def _progress_callback(
            iteration: int, stage: Any, progress: float, best_score: float
        ) -> None:
            self.update_state(
                state="STARTED",
                meta={
                    "current_iteration": int(iteration),
                    "total_iterations": int(request.max_iterations),
                    "current_stage": getattr(stage, "value", str(stage)),
                    "progress_percentage": float(progress * 100),
                    "best_score": float(best_score),
                    "errors": [],
                },
            )

        loop_config = LoopConfig(
            max_iterations=request.max_iterations,
            backtest_period=request.backtest_period,
            initial_capital=request.initial_capital,
            risk_tolerance=request.risk_tolerance,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            user_id=request_payload.get("_owner_user_id", "default"),
            tenant_id=request_payload.get("_owner_tenant_id", "default"),
        )
        strategy_request = StrategyRequest(
            prompt=request.prompt,
            strategy_type=(
                StrategyType(request.strategy_type) if request.strategy_type else None
            ),
            complexity_level=ComplexityLevel(request.complexity_level),
            target_assets=request.target_assets,
            timeframe=request.timeframe,
            risk_tolerance=request.risk_tolerance,
            backtest_period=request.backtest_period,
            custom_requirements=request.custom_requirements,
        )

        market_data_service = MarketDataService()
        market_data = await market_data_service.get_market_data(
            symbols=request.target_assets or ["SZ000001"],
            start_date=start_date,
            end_date=end_date,
            timeframe=request.timeframe,
        )

        loop_manager = StrategyBacktestLoop(loop_config)
        result = await loop_manager.run_loop(
            strategy_request, market_data, progress_callback=_progress_callback
        )

        best_strategy = {}
        performance_metrics = {}
        best_score = 0.0
        if result.best_iteration:
            best_score = float(
                getattr(result.best_iteration, "performance_score", 0.0) or 0.0
            )
            if getattr(result.best_iteration, "strategy_response", None):
                best_strategy = result.best_iteration.strategy_response.to_dict()
            if getattr(result.best_iteration, "backtest_result", None):
                performance_metrics = (
                    result.best_iteration.backtest_result.performance_metrics or {}
                )

        all_iterations = []
        for iteration in result.all_iterations:
            all_iterations.append(
                {
                    "iteration": iteration.iteration,
                    "stage": iteration.stage.value,
                    "performance_score": iteration.performance_score,
                    "improvement": iteration.improvement,
                    "execution_time": iteration.execution_time,
                    "errors": iteration.errors,
                }
            )

        # 在进入数据库持久化之前，报告 99% 进度，告知前端正在保存结果
        self.update_state(
            state="STARTED",
            meta={
                "current_iteration": int(result.total_iterations),
                "total_iterations": int(result.total_iterations),
                "current_stage": "persistence",
                "progress_percentage": 99.0,
                "best_score": best_score,
                "errors": [],
            },
        )

        payload = {
            "task_id": task_id,
            "success": bool(result.success),
            "total_iterations": int(result.total_iterations),
            "best_strategy": best_strategy,
            "performance_metrics": performance_metrics,
            "learning_insights": dict(result.learning_insights or {}),
            "execution_time": float(result.total_time),
            "all_iterations": all_iterations,
            "best_score": best_score,
        }
        await persistence.update_task(
            task_id=task_id,
            status="completed",
            updated_at=datetime.now(),
            result_payload=payload,
        )
        return payload

    owner_user_id = request_payload.get("_owner_user_id")
    if not owner_user_id:
        raise RuntimeError("missing owner identity for strategy loop task")
    try:
        return _run_async(_run())
    except Exception as exc:
        persistence = StrategyLoopPersistence()
        _run_async(persistence.ensure_tables())
        _run_async(
            persistence.update_task(
                task_id=task_id,
                status="failed",
                updated_at=datetime.now(),
                error_message=str(exc),
            )
        )
        raise


@celery_app.task(
    name="engine.tasks.sync_stock_daily_latest_task",
    max_retries=0,
)
def sync_stock_daily_latest_task(
    target_date: str | None = None, max_symbols: int = 0, apply: bool = True
) -> dict[str, Any]:
    """
    [已废弃] 数据现由官方服务器统一推送，不再需要本地 Baostock 同步。
    """
    logger.info("[SyncTask] 该任务已废弃，数据由官方服务器统一推送")
    return {
        "success": True,
        "message": "数据同步任务已废弃，数据由官方服务器统一推送",
        "deprecated": True,
    }


@celery_app.task(name="engine.tasks.sync_market_data_daily_task")
def sync_market_data_daily_task(
    target_date: str | None = None,
    max_symbols: int = 0,
    apply: bool = True,
) -> dict[str, Any]:
    """
    Celery 任务：从 Baostock 增量更新基础行情与 Qlib 日线。

    高维模型特征继续由签名官方数据包更新；此任务不会填充伪特征。
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(os.getcwd())
    script = root / "scripts" / "data" / "maintenance" / "sync_daily_from_baostock.py"
    cmd = [sys.executable, str(script)]
    if target_date:
        cmd.extend(["--target-date", target_date])
    if max_symbols:
        cmd.extend(["--max-symbols", str(max_symbols)])
    if apply:
        cmd.append("--apply")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,
            cwd=str(root),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "message": "Baostock 数据同步超时"}

    success = result.returncode == 0
    if not success:
        logger.error(
            "[DataSync] failed exit=%s stderr=%s",
            result.returncode,
            result.stderr[-1000:],
        )
    return {
        "success": success,
        "source": "baostock",
        "exit_code": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }


@celery_app.task(name="engine.tasks.get_data_status_task")
def get_data_status_task() -> dict[str, Any]:
    """
    Celery 任务：扫描 Qlib 数据目录和数据库状态，并缓存结果。
    改善 Admin UI 响应速度。
    """
    import os
    import json
    import struct
    from pathlib import Path
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from sqlalchemy import text
    import exchange_calendars as xcals

    # 复用 model_management.py 的逻辑，但增加 BJ 标持支持
    now_local = datetime.now(ZoneInfo("Asia/Shanghai"))

    # 获取目标日期规则
    cal_xshg = xcals.get_calendar("XSHG")
    if now_local.time() < datetime.strptime("09:30", "%H:%M").time():
        trade_date_obj = cal_xshg.previous_session(now_local.date()).date()
    else:
        trade_date_obj = now_local.date()
    trade_date = trade_date_obj.isoformat()

    qlib_data_dir = Path(os.getcwd()) / "db" / "qlib_data"
    calendars_path = qlib_data_dir / "calendars" / "day.txt"
    instruments_all_path = qlib_data_dir / "instruments" / "all.txt"
    features_root = qlib_data_dir / "features"

    qlib_info: dict[str, Any] = {
        "qlib_dir": str(qlib_data_dir),
        "exists": qlib_data_dir.exists() and qlib_data_dir.is_dir(),
        "calendar_total_days": 0,
        "calendar_start_date": None,
        "calendar_last_date": None,
        "instruments": {"total": 0, "sh": 0, "sz": 0, "bj": 0, "other": 0},
        "feature_dirs_total": 0,
        "feature_dirs_sh_sz_bj": 0,
        "latest_date_coverage": {
            "target_date": None,
            "at_target_count": 0,
            "older_count": 0,
            "invalid_count": 0,
        },
        "topn_samples": {
            "sample_size": 20,
            "older_samples": [],
            "invalid_samples": [],
        },
    }

    calendar: list[str] = []
    if calendars_path.exists():
        try:
            calendar = [
                x.strip()
                for x in calendars_path.read_text(encoding="utf-8").splitlines()
                if x.strip()
            ]
            if calendar:
                qlib_info["calendar_total_days"] = len(calendar)
                qlib_info["calendar_start_date"] = calendar[0]
                qlib_info["calendar_last_date"] = calendar[-1]
                qlib_info["latest_date_coverage"]["target_date"] = calendar[-1]
        except Exception as e:
            qlib_info["calendar_error"] = str(e)

    if instruments_all_path.exists():
        try:
            for line in instruments_all_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                code = line.split()[0].strip().upper()
                qlib_info["instruments"]["total"] += 1
                if code.startswith("SH"):
                    qlib_info["instruments"]["sh"] += 1
                elif code.startswith("SZ"):
                    qlib_info["instruments"]["sz"] += 1
                elif code.startswith("BJ"):
                    qlib_info["instruments"]["bj"] += 1
                else:
                    qlib_info["instruments"]["other"] += 1
        except Exception as e:
            qlib_info["instruments_error"] = str(e)

    if features_root.exists() and features_root.is_dir() and calendar:
        feature_dirs = [p for p in features_root.iterdir() if p.is_dir()]
        qlib_info["feature_dirs_total"] = len(feature_dirs)

        cal_len = len(calendar)
        target_date = calendar[-1]
        at_target_count = 0
        older_count = 0
        invalid_count = 0
        sample_size = 20
        older_samples = []
        invalid_samples = []

        for inst_dir in feature_dirs:
            inst = inst_dir.name.upper()
            # 兼容 SH, SZ, BJ
            if not inst.startswith(("SH", "SZ", "BJ")):
                continue

            qlib_info["feature_dirs_sh_sz_bj"] += 1
            close_bin = inst_dir / "close.day.bin"

            if not close_bin.exists():
                invalid_count += 1
                if len(invalid_samples) < sample_size:
                    invalid_samples.append(
                        {
                            "symbol": inst,
                            "reason": "missing_close_bin",
                            "file": str(close_bin),
                        }
                    )
                continue

            try:
                size = close_bin.stat().st_size
                if size < 8:
                    invalid_count += 1
                    if len(invalid_samples) < sample_size:
                        invalid_samples.append(
                            {
                                "symbol": inst,
                                "reason": "bin_too_small",
                                "file": str(close_bin),
                            }
                        )
                    continue

                with close_bin.open("rb") as f:
                    head = f.read(4)
                if len(head) < 4:
                    start_idx = None
                else:
                    start_idx = int(struct.unpack("<f", head)[0])

                if start_idx is None:
                    invalid_count += 1
                    continue

                nrows = (size - 4) // 4
                end_idx = start_idx + nrows - 1
                if not (0 <= end_idx < cal_len):
                    invalid_count += 1
                    continue

                last_date = calendar[end_idx]
                if last_date == target_date:
                    at_target_count += 1
                else:
                    older_count += 1
                    if len(older_samples) < sample_size:
                        older_samples.append(
                            {
                                "symbol": inst,
                                "last_date": last_date,
                                "lag_days": max(0, cal_len - 1 - end_idx),
                            }
                        )
            except:
                invalid_count += 1

        qlib_info["latest_date_coverage"] = {
            "target_date": target_date,
            "at_target_count": at_target_count,
            "older_count": older_count,
            "invalid_count": invalid_count,
        }
        qlib_info["topn_samples"] = {
            "sample_size": sample_size,
            "older_samples": sorted(older_samples, key=lambda x: x["last_date"])[
                :sample_size
            ],
            "invalid_samples": invalid_samples[:sample_size],
        }

    # feature_snapshots 检测（替代 stock_daily_latest）
    feature_snapshots_info: dict[str, Any] = {
        "exists": False,
        "snapshot_dir": str(Path(os.getcwd()) / "db" / "feature_snapshots"),
        "file_count": 0,
        "scanned_files": 0,
        "failed_files": 0,
        "total_rows": 0,
        "min_date": None,
        "max_date": None,
        "latest_date_coverage": {
            "target_date": trade_date,
            "at_target_count": 0,
            "older_count": 0,
            "invalid_count": 0,
        },
        "topn_samples": {
            "sample_size": 20,
            "older_samples": [],
            "invalid_samples": [],
        },
        "suggested_periods": None,
    }

    try:
        from backend.services.api.routers.admin.model_management_utils import (
            _scan_feature_snapshots_status,
        )

        feature_snapshots_info = _scan_feature_snapshots_status(
            target_date=trade_date, topn=20
        )
    except Exception as e:
        feature_snapshots_info["error"] = str(e)

    result = {
        "checked_at": now_local.isoformat(),
        "trade_date": trade_date,
        "qlib_data": qlib_info,
        "feature_snapshots": feature_snapshots_info,
    }

    # 存入 Redis
    try:
        redis = get_redis_sentinel_client()
        redis.set("qm:admin:data_status", json.dumps(result), ex=300)  # 5分钟缓存
    except Exception as e:
        logger.warning("Failed to cache data status to Redis: %s", e)

    return result


@celery_app.task(name="engine.tasks.warmup_stock_latest_cache")
def warmup_stock_latest_cache_task():
    """Celery 任务：预热 stock_daily_latest 表的 Redis 缓存。"""
    from backend.shared.market_data.stock_daily_latest_cache import stock_latest_cache
    import asyncio

    logger.info("[CacheWarmup] 正在启动 stock_daily_latest 缓存预热...")
    try:
        # 由于当前环境可能在同步的 Celery worker 中运行，使用 asyncio.run 或直接调用
        count = asyncio.run(stock_latest_cache.warmup_cache())
        logger.info(f"[CacheWarmup] 预热完成，共缓存 {count} 只股票行情。")
        return {"status": "success", "count": count}
    except Exception as e:
        logger.error(f"[CacheWarmup] 预热失败: {e}")
        return {"status": "failed", "error": str(e)}
