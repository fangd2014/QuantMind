"""市场定时同步调度器。

前端每个市场 tab 可配置每天 HH:MM 定时同步上游数据（精确到分钟）。
配置存 Redis（db 0，key: quantmind:sync_schedule:{market}），
Celery beat 每分钟触发 dispatch_market_sync 检查是否有市场到点，
到点则派发对应市场同步任务（Redis 记录 last_run 防止重复触发）。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

_SCHEDULE_KEY = "quantmind:sync_schedule:{market}"
_LAST_RUN_KEY = "quantmind:sync_schedule_last_run:{market}:{date}"

# market -> (标签, 同步任务名)
MARKETS = {
    "A": "QuantDB A股",
    "US": "QuantUS 美股",
    "HK": "QuantHK 港股",
    "BC": "QuantBC 区块链",
    "FUTURES": "QuantFutures 期货",
}

DEFAULT_SCHEDULE = {
    "enabled": False,
    "time": "22:30",
    "days": 5,
    "datasets": [],
    "with_qlib": False,
}


def _redis():
    import redis

    return redis.from_url(
        os.getenv("REDIS_URL", "redis://redis:6379/0"), socket_timeout=3
    )


def _normalize(cfg: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(DEFAULT_SCHEDULE)
    if not cfg:
        return out
    for k in out:
        if k in cfg:
            out[k] = cfg[k]
    # 校验 time 格式 HH:MM
    t = str(out["time"]).strip()
    try:
        datetime.strptime(t, "%H:%M")
        out["time"] = t
    except ValueError:
        out["time"] = DEFAULT_SCHEDULE["time"]
    return out


def get_schedule(market: str) -> dict[str, Any]:
    r = _redis()
    raw = r.get(_SCHEDULE_KEY.format(market=market))
    cfg = json.loads(raw) if raw else None
    return _normalize(cfg)


def get_all_schedules() -> dict[str, dict[str, Any]]:
    return {m: get_schedule(m) for m in MARKETS}


def save_schedule(market: str, cfg: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize(cfg)
    r = _redis()
    r.set(
        _SCHEDULE_KEY.format(market=market),
        json.dumps(normalized, ensure_ascii=False),
    )
    return normalized


def _last_run_today(market: str, date_str: str) -> bool:
    r = _redis()
    return r.exists(_LAST_RUN_KEY.format(market=market, date=date_str)) > 0


def _mark_run(market: str, date_str: str) -> None:
    r = _redis()
    r.set(
        _LAST_RUN_KEY.format(market=market, date=date_str),
        "1",
        ex=2 * 24 * 3600,
    )


def run_market_sync(market: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """执行指定市场的同步（按配置的数据集/天数）。"""
    days = int(cfg.get("days") or 5)
    datasets = cfg.get("datasets") or []
    with_qlib = bool(cfg.get("with_qlib"))

    result: dict[str, Any] = {"market": market, "started": datetime.now().isoformat()}

    if market == "A":
        from backend.scripts.quantdb_daily_sync import run_daily_sync

        result["result"] = run_daily_sync(skip_pg=True)
        return result

    if market == "US":
        from backend.scripts.quantus_daily_sync import run
    elif market == "HK":
        from backend.scripts.quanthk_daily_sync import run
    elif market == "BC":
        from backend.scripts.quantbc_daily_sync import run
    elif market == "FUTURES":
        from backend.scripts.quantfutures_daily_sync import run
    else:
        return {"market": market, "error": f"未知市场: {market}"}

    kwargs: dict[str, Any] = {"days": days}
    if datasets:
        kwargs["datasets"] = list(datasets)
    result["result"] = run(**kwargs)

    if with_qlib:
        try:
            from backend.services.engine.qlib_data_builder import ensure_qlib_cache

            qlib_market = {"US": "US", "HK": "HK", "BC": "CRYPTO", "FUTURES": "FUTURES"}[market]
            result["qlib"] = {"status": "ok", "provider_uri": ensure_qlib_cache(market=qlib_market)}
        except Exception as exc:  # noqa: BLE001
            logger.error("%s 定时同步 qlib 缓存失败: %s", market, exc, exc_info=True)
            result["qlib"] = {"status": "error", "reason": str(exc)}

    result["finished"] = datetime.now().isoformat()
    return result


def dispatch_due_syncs() -> dict[str, Any]:
    """检查所有市场定时配置，到点且今天未跑过的派发同步任务。"""
    from backend.services.engine.qlib_app.celery_config import celery_app

    now = datetime.now()
    now_hm = now.strftime("%H:%M")
    date_str = now.strftime("%Y-%m-%d")
    dispatched: list[str] = []
    skipped: list[dict[str, str]] = []

    for market in MARKETS:
        cfg = get_schedule(market)
        if not cfg.get("enabled"):
            continue
        if cfg.get("time") != now_hm:
            continue
        if _last_run_today(market, date_str):
            continue
        if market == "A":
            from backend.shared.runtime_secrets import (
                QUANTDB_CONFIG_HINT,
                get_quantdb_api_key,
            )

            if not get_quantdb_api_key():
                skipped.append({"market": market, "reason": QUANTDB_CONFIG_HINT})
                logger.warning("[SyncSchedule] %s", QUANTDB_CONFIG_HINT)
                continue
        _mark_run(market, date_str)
        celery_app.send_task(
            "engine.tasks.run_market_scheduled_sync",
            args=[market, cfg],
            queue="qlib_backtest_srv",
        )
        dispatched.append(market)
        logger.info("[SyncSchedule] %s 到点 %s，已派发同步任务", MARKETS[market], now_hm)

    return {"now": now_hm, "dispatched": dispatched, "skipped": skipped}
