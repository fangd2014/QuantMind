"""Extended Strategy Implementations"""

import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO

# Import base strategies
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy, WeightStrategyBase

# 引入数据接口
from backend.services.engine.qlib_app.utils.qlib_utils import D
from backend.services.engine.qlib_app.utils.recording_strategy import (
    _OUR_KWARGS,
    DynamicRiskMixin,
    RedisLoggerMixin,
    RedisRecordingStrategy,
    RedisWeightStrategy,
    strip_unsupported_kwargs,
)
from backend.services.engine.qlib_app.utils.structured_logger import StructuredTaskLogger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 两融标的池加载（模块级缓存，仅加载一次）
# ---------------------------------------------------------------------------
# instruments/margin.txt 格式：STOCK_ID\tSTART_DATE\tEND_DATE
# 例：SH600000\t2010-03-31\t2025-12-31

_MARGIN_POOL_CACHE: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] | None = None


def _find_margin_txt() -> str | None:
    """在多个候选路径中寻找 margin.txt，返回首个存在的绝对路径。"""
    candidates: list[str] = []

    # 1. Qlib provider_uri（运行时已知最精确）
    try:
        from qlib.config import C
        uri = C.get("provider_uri", None)
        if uri:
            candidates.append(os.path.join(str(uri), "instruments", "margin.txt"))
    except Exception:
        pass

    # 2. 通过 qlib_paths 统一解析（优先 QuantDB 缓存）
    try:
        from backend.shared.qlib_paths import resolve_qlib_instruments_path
        margin_path = str(resolve_qlib_instruments_path()).replace("all.txt", "margin.txt")
        candidates.insert(0, margin_path)
    except Exception:
        pass

    # 3. 常见相对于项目根目录的路径（兼容回退）
    try:
        curr = os.path.abspath(__file__)
        for _ in range(10):
            parent = os.path.dirname(curr)
            if parent == curr:
                break
            curr = parent
            marker = os.path.join(curr, "db", "qlib_data", "instruments", "margin.txt")
            if os.path.exists(marker):
                candidates.insert(0, marker)
                break
    except Exception:
        pass

    # 3. 环境变量覆盖
    env_path = os.environ.get("QLIB_MARGIN_TXT")
    if env_path:
        candidates.insert(0, env_path)

    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _load_margin_pool() -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """加载两融标的池，返回 {stock_id: [(start, end), ...]} 映射。"""
    global _MARGIN_POOL_CACHE
    if _MARGIN_POOL_CACHE is not None:
        return _MARGIN_POOL_CACHE

    path = _find_margin_txt()
    if not path:
        StructuredTaskLogger(logger, "margin-pool").warning(
            "margin_pool_missing",
            "未找到 instruments/margin.txt，空头侧将不受两融标的池约束",
        )
        _MARGIN_POOL_CACHE = {}
        return _MARGIN_POOL_CACHE

    pool: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                sid, start_str, end_str = parts[0].strip(), parts[1].strip(), parts[2].strip()
                try:
                    start = pd.Timestamp(start_str)
                    end = pd.Timestamp(end_str)
                    pool.setdefault(sid, []).append((start, end))
                except Exception:
                    continue
        StructuredTaskLogger(logger, "margin-pool").info(
            "margin_pool_loaded", "已加载两融标的池", stock_count=len(pool), path=path
        )
    except Exception as e:
        StructuredTaskLogger(logger, "margin-pool").error("margin_pool_load_failed", "加载 margin.txt 失败", error=e)
        pool = {}

    _MARGIN_POOL_CACHE = pool
    return _MARGIN_POOL_CACHE


def get_margin_eligible_set(trade_date: pd.Timestamp) -> set[str] | None:
    """
    返回指定日期可融券的股票集合。
    若 margin.txt 未找到则返回 None（不过滤）。
    """
    pool = _load_margin_pool()
    if not pool:
        return None  # 无法加载时放行，避免阻断回测

    eligible: set[str] = set()
    for sid, intervals in pool.items():
        for start, end in intervals:
            if start <= trade_date <= end:
                eligible.add(sid)
                break
    return eligible


class RedisTopkStrategy(DynamicRiskMixin, TopkDropoutStrategy, RedisLoggerMixin):
    """
    Simple TopK Strategy
    """

    def __init__(self, *args, **kwargs):
        # 提取调仓周期参数
        self.rebalance_days = int(kwargs.pop("rebalance_days", 1))

        self.init_redis(kwargs)
        self.init_dynamic_risk(kwargs)

        # 必须显式剔除所有不被 BaseStrategy 接受的参数
        # _OUR_KWARGS 可能未包含 rebalance_days，如果 recording_strategy.py 没有更新 _OUR_KWARGS
        for k in list(kwargs.keys()):
            if k in _OUR_KWARGS or k == "rebalance_days":
                kwargs.pop(k, None)

        # 兜底：剔除 qlib 签名不接受的其余参数（如前端传入但未实现的 momentum_period）
        strip_unsupported_kwargs(type(self), kwargs, strategy_name="RedisTopkStrategy")

        # 全局规则：选股时剔除涨停/跌停/停牌股，避免无效名额占用
        kwargs.setdefault("only_tradable", True)
        self._current_step = 0
        super().__init__(*args, **kwargs)

    def reset(self, *args, **kwargs):
        """兼容 qlib reset 签名差异（level_infra/common_infra/trade_exchange）。"""
        self._current_step = 0
        self.reset_dynamic_risk()
        try:
            return super().reset(*args, **kwargs)
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            filtered = dict(kwargs)
            filtered.pop("level_infra", None)
            filtered.pop("common_infra", None)
            filtered.pop("trade_exchange", None)
            try:
                return super().reset(*args, **filtered)
            except TypeError:
                return super().reset()

    def _safe_generate_trade_decision(self, execute_result=None):
        """安全地生成交易决策，处理 get_deal_price 返回 None 的情况。"""
        try:
            return super().generate_trade_decision(execute_result)
        except TypeError as e:
            if "unsupported operand type(s) for /: 'float' and 'NoneType'" in str(e):
                # 价格为 None，可能是股票停牌或数据缺失，跳过本次交易
                StructuredTaskLogger(
                    logger,
                    "redis-topk-strategy",
                    {"backtest_id": getattr(self, "backtest_id", None)},
                ).warning("skip_trade_no_price", "Skip trade due to missing price data")
                return TradeDecisionWO([], self)
            raise

    def generate_trade_decision(self, execute_result=None):
        if hasattr(self, "check_account_stop_loss") and self.check_account_stop_loss():
            StructuredTaskLogger(
                logger,
                "redis-topk-strategy",
                {"backtest_id": getattr(self, "backtest_id", None)},
            ).info("account_stop_loss", "Account stop-loss triggered. Liquidating.")
            return self._liquidate_all()

        # 调仓周期控制
        if self.rebalance_days > 1:
            try:
                # 统一使用 safe 方法获取
                from backend.services.engine.qlib_app.utils.recording_strategy import RedisRecordingStrategy

                trade_step = RedisRecordingStrategy._get_trade_step_safe(self) or 0
                if trade_step % self.rebalance_days != 0:
                    return TradeDecisionWO([], self)
            except Exception as e:
                StructuredTaskLogger(
                    logger,
                    "redis-topk-strategy",
                    {"backtest_id": getattr(self, "backtest_id", None), "rebalance_days": self.rebalance_days},
                ).warning("rebalance_check_failed", "Error checking rebalance_days", error=e)

        # Generate new orders (with safe handling for None prices)
        return self._safe_generate_trade_decision(execute_result)

    def post_exe_step(self, execute_result=None):
        self.log_progress()
        self.log_executed_trades(execute_result)


class RedisRiskGuardTopkStrategy(RedisRecordingStrategy):
    """
    大盘风控 Top-K 选股策略 (Risk Guard Top-K)
    继承 RedisRecordingStrategy（DynamicRiskMixin + FundamentalFilterMixin + TopkDropout）：
    1. FundamentalFilterMixin 消化 f_* 基本面/交易硬过滤（排除 ST、涨跌停、流动性等）；
    2. DynamicRiskMixin 结合大盘状态（market_state_series）自动降仓；
    3. 维持 Top-K-Dropout 的低换手优势。
    额外消费行业分散参数（max_industry_count / industry_cap_ratio）。
    """

    def __init__(self, *args, **kwargs):
        # 行业分散参数：当前由平台上层消费，策略层消化避免传给 qlib BaseStrategy
        self.max_industry_count = int(kwargs.pop("max_industry_count", 0) or 0)
        self.industry_cap_ratio = float(kwargs.pop("industry_cap_ratio", 0.0) or 0.0)
        self.market_state_window = int(kwargs.pop("market_state_window", 20) or 20)
        super().__init__(*args, **kwargs)


class RedisMomentumStrategy(RedisTopkStrategy):
    """
    趋势动量策略 (Momentum Strategy)
    继承 RedisTopkStrategy 复用 TopK-Dropout 低换手、Redis 记录与动态风控，
    在选股前用「过去 momentum_period 日累计收益率」作为动量因子与模型分数融合，
    使得动量强（近期涨幅领先）的标的获得更高排名，符合 A 股趋势跟随逻辑。

    融合方式：rank_score = model_score + lambda * momentum_factor
    - momentum_factor 为过去 N 日累计收益（截面归一化到 [-1,1]）
    - lambda 为动量强度，保守默认 0.5（模型分数仍占主导）
    """

    def __init__(self, *args, **kwargs):
        self.momentum_period = int(kwargs.pop("momentum_period", 20))
        self.momentum_weight = float(kwargs.pop("momentum_weight", 0.5))
        super().__init__(*args, **kwargs)
        StructuredTaskLogger(
            logger,
            "redis-momentum-strategy",
            {"backtest_id": getattr(self, "backtest_id", None), "momentum_period": self.momentum_period, "momentum_weight": self.momentum_weight},
        ).info("init", "RedisMomentumStrategy initialized")

    def _compute_momentum(self, stocks: list[str], ref_date) -> pd.Series:
        """计算各标的过去 momentum_period 交易日的累计收益率。"""
        try:
            lookback_buf = int(self.momentum_period * 2.0)
            start_dt = ref_date - pd.Timedelta(days=lookback_buf)
            price_df = D.features(stocks, ["$close"], start_dt, ref_date, freq="day")
            if price_df is None or price_df.empty:
                raise ValueError("empty price data")
            prices = price_df["$close"].unstack(level="instrument")
            # 累计收益率 = 最后价格 / 区间初价格 - 1（仅用最新 period 个交易日）
            momentum = prices.iloc[-self.momentum_period :].iloc[-1] / prices.iloc[-self.momentum_period :].iloc[0] - 1
            # 截面归一化到 [-1, 1]，避免量纲影响
            std = momentum.std(ddof=1)
            if std is None or std == 0 or float(std) == 0:
                return pd.Series(0.0, index=stocks)
            norm = ((momentum - momentum.mean()) / std).clip(-1, 1)
            return norm.reindex(stocks).fillna(0.0)
        except Exception as exc:
            StructuredTaskLogger(
                logger,
                "redis-momentum-strategy",
                {"backtest_id": getattr(self, "backtest_id", None)},
            ).warning("momentum_computation_failed", "Momentum computation failed, falling back to model score only", error=exc)
            return pd.Series(0.0, index=stocks)

    def generate_target_weight_position(self, score, current=None, trade_exchange=None, *args, **kwargs):
        if self.check_account_stop_loss():
            return {}
        if score is None or score.empty:
            return {}
        if isinstance(score, pd.DataFrame):
            score = score.iloc[:, 0]
        t_start = kwargs.get("trade_start_time") or kwargs.get("t_start")
        if t_start is not None:
            stocks = list(score.index)
            momentum = self._compute_momentum(stocks, pd.Timestamp(t_start))
            # 模型分数占主导 + 动量增强（融合后保留正负号以区分多空强度）
            score = score.add(momentum.reindex(score.index) * self.momentum_weight)
        return super().generate_target_weight_position(score, current, trade_exchange, *args, **kwargs)


class RedisAdvancedAlphaStrategy(RedisTopkStrategy):
    """
    高级截面 Alpha 策略：结合了 TopK-Dropout 的低换手优势和分数权重的盈利能力。
    1. 继承 RedisTopkStrategy 获得 Redis 记录、动态风控及 TopK-Dropout 核心逻辑。
    2. 通过覆写 generate_target_weight_position，将等权替换为“分数加权 + 单票上限”。
    """

    def __init__(self, *args, **kwargs):
        # 提取高级参数
        self.max_weight = float(kwargs.pop("max_weight", 0.05))
        self.min_score = float(kwargs.pop("min_score", 0.0))
        # 调用父类初始化（它会处理 signal, topk, n_drop, rebalance_days 等）
        super().__init__(*args, **kwargs)
        StructuredTaskLogger(
            logger,
            "redis-advanced-alpha-strategy",
            {"backtest_id": getattr(self, "backtest_id", None), "max_weight": self.max_weight, "min_score": self.min_score},
        ).info("init", "RedisAdvancedAlphaStrategy initialized")

    def generate_target_weight_position(self, score, current=None, trade_exchange=None, *args, **kwargs):
        # 1. 前置过滤
        if score is None or score.empty:
            return {}

        # 确保 score 是 Series
        if isinstance(score, pd.DataFrame):
            if score.shape[1] > 0:
                score = score.iloc[:, 0]
            else:
                return {}

        # 仅保留大于阈值的正分
        score = score[score > self.min_score]
        if score.empty:
            return {}

        # 2. 调用父类的选股逻辑 (TopK-Dropout)
        # 该逻辑会结合 current 持仓计算出本次应持有的股票集合（等权形式）
        base_weights = super().generate_target_weight_position(score, current, trade_exchange, *args, **kwargs)
        if not base_weights:
            return {}

        # 3. 选股后的“分数权重”分配
        selected_sids = list(base_weights.keys())
        # 注意：selected_sids 中可能包含由于 Dropout 保留但当前无分数的标的，需稳健处理
        sub_score = score.reindex(selected_sids).fillna(score.min() if not score.empty else 0.0)

        # 计算原始权重
        total_score = sub_score.sum()
        if total_score <= 1e-9:
            return base_weights  # 如果总分为 0，回退到等权

        weights = sub_score / total_score

        # 4. 应用单票权重限制 (max_weight)
        if 0 < self.max_weight < 1.0:
            weights_series = pd.Series(weights)
            # 简单的迭代重分配逻辑
            for _ in range(10):
                over = weights_series > self.max_weight
                if not over.any():
                    break
                weights_series.loc[over] = self.max_weight
                remaining_sum = weights_series[~over].sum()
                if remaining_sum > 1e-9:
                    weights_series.loc[~over] = (
                        weights_series[~over] * (1.0 - weights_series[over].sum()) / remaining_sum
                    )
                else:
                    # 如果剩下全溢出了，只好强行截断后归一化
                    weights_series = weights_series / weights_series.sum()
                    break
            weights = weights_series.to_dict()

        return weights


class RedisLongShortTopkStrategy(DynamicRiskMixin, WeightStrategyBase, RedisLoggerMixin):
    """
    原生多空 TopK 策略。

    逻辑：
    1. 取预测分最高的 topk 只做多。
    2. 取预测分最低的 short_topk 只做空。
    3. 多头和空头分别按分数绝对值归一化，再映射到 long/short exposure。
    """

    def __init__(
        self,
        *args,
        topk=None,
        short_topk=None,
        min_score=0.0,
        max_weight=1.0,
        long_exposure=1.0,
        short_exposure=1.0,
        **kwargs,
    ):
        self.topk = int(topk) if topk is not None else 50
        self.short_topk = int(short_topk) if short_topk is not None else self.topk
        self.min_score = float(min_score) if min_score is not None else 0.0
        self.max_weight = float(max_weight) if max_weight is not None else 1.0
        self.long_exposure = float(long_exposure) if long_exposure is not None else 1.0
        self.short_exposure = float(short_exposure) if short_exposure is not None else 1.0
        self.rebalance_days = int(kwargs.pop("rebalance_days", 1))

        self.init_redis(kwargs)
        self.init_dynamic_risk(kwargs)
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in _OUR_KWARGS}
        strip_unsupported_kwargs(type(self), clean_kwargs, strategy_name="RedisLongShortTopkStrategy")
        super().__init__(*args, **clean_kwargs)

    def reset(self, *args, **kwargs):
        self._qm_trade_step_counter = 0
        self._initial_capital = None
        self.reset_dynamic_risk()
        try:
            return super().reset(*args, **kwargs)
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            filtered = dict(kwargs)
            filtered.pop("level_infra", None)
            filtered.pop("common_infra", None)
            filtered.pop("trade_exchange", None)
            try:
                return super().reset(*args, **filtered)
            except TypeError:
                return super().reset()

    def _filter_tradeable_scores(
        self,
        score: pd.Series,
        exchange: Any,
        t_start: Any,
        t_end: Any,
        direction: int,
        side_label: str,
    ) -> pd.Series:
        if exchange is None or score is None or score.empty or t_start is None:
            return score

        filtered_index = []
        skipped_trade = 0
        skipped_margin = 0

        # 空头侧额外加载两融标的池（仅在 SELL 方向触发一次）
        margin_set: set[str] | None = None
        if direction == Order.SELL:
            try:
                trade_date = pd.Timestamp(t_start)
                margin_set = get_margin_eligible_set(trade_date)
            except Exception as e:
                StructuredTaskLogger(
                    logger,
                    "redis-long-short-topk",
                    {"backtest_id": getattr(self, "backtest_id", None)},
                ).warning("margin_pool_error", "获取两融标的池异常，空头侧不受限", error=e)

        for sid in score.index:
            try:
                if exchange.check_stock_suspended(sid, t_start, t_end):
                    skipped_trade += 1
                    continue
                if exchange.check_stock_limit(sid, t_start, t_end, direction=direction):
                    skipped_trade += 1
                    continue
            except Exception:
                pass

            # 空头侧：必须是当日可融券标的
            if direction == Order.SELL and margin_set is not None and sid not in margin_set:
                skipped_margin += 1
                continue

            filtered_index.append(sid)

        if skipped_trade:
            StructuredTaskLogger(
                logger,
                "redis-long-short-topk",
                {"backtest_id": getattr(self, "backtest_id", None)},
            ).info(
                "side_filter",
                "剔除停牌/涨跌停标的",
                trade_date=str(pd.Timestamp(t_start).date()),
                side=side_label,
                skipped=skipped_trade,
            )
        if skipped_margin:
            StructuredTaskLogger(
                logger,
                "redis-long-short-topk",
                {"backtest_id": getattr(self, "backtest_id", None)},
            ).info(
                "margin_filter",
                "剔除非两融标的",
                trade_date=str(pd.Timestamp(t_start).date()),
                side=side_label,
                skipped=skipped_margin,
            )
        return score.loc[filtered_index]

    def _build_side_weights(self, scores: pd.Series, target_exposure: float) -> pd.Series:
        if scores is None or scores.empty or target_exposure <= 0:
            return pd.Series(dtype=float)

        # 防御性去重：同一股票若出现多次，保留首个（最高分）
        if not scores.index.is_unique:
            scores = scores[~scores.index.duplicated(keep="first")]

        abs_scores = scores.abs().astype(float)
        total = abs_scores.sum()
        if total <= 0:
            return pd.Series(dtype=float)

        if self.max_weight is None or self.max_weight <= 0:
            return (abs_scores / total * target_exposure).astype(float)

        cap = min(float(self.max_weight), float(target_exposure))
        weights = pd.Series(0.0, index=abs_scores.index, dtype=float)
        remaining = abs_scores.copy()
        remaining_exposure = float(target_exposure)

        while not remaining.empty and remaining_exposure > 1e-12:
            scaled = remaining / remaining.sum() * remaining_exposure
            over = scaled > cap
            if not over.any():
                weights.loc[remaining.index] = scaled.astype(float)
                break

            weights.loc[scaled[over].index] = cap
            remaining_exposure = max(0.0, float(target_exposure) - float(weights.sum()))
            remaining = remaining.loc[~over]
            if remaining.sum() <= 0:
                break

        return weights[weights > 0]

    def generate_target_weight_position(self, score, current=None, trade_exchange=None, *args, **kwargs):
        if hasattr(self, "check_account_stop_loss") and self.check_account_stop_loss():
            StructuredTaskLogger(
                logger,
                "redis-long-short-topk",
                {"backtest_id": getattr(self, "backtest_id", None)},
            ).info("account_stop_loss", "Account stop-loss triggered. Target position is empty.")
            return {}

        if current is None and args:
            current = args[0]
        if trade_exchange is None and len(args) > 1:
            trade_exchange = args[1]
        if score is None or len(score) == 0:
            return {}

        if isinstance(score, pd.DataFrame):
            if score.shape[1] == 0:
                return {}
            score = score.iloc[:, 0]

        score = score.dropna()
        if score.empty:
            return {}

        # 防御性去重：确保索引唯一，避免后续 reindex 报错
        if not score.index.is_unique:
            score = score[~score.index.duplicated(keep="first")]

        threshold = abs(float(self.min_score or 0.0))
        long_scores = score[score > threshold]
        short_scores = score[score < -threshold]

        if self.topk > 0 and len(long_scores) > self.topk:
            long_scores = long_scores.nlargest(self.topk)
        if self.short_topk > 0 and len(short_scores) > self.short_topk:
            short_scores = short_scores.nsmallest(self.short_topk)
        elif self.short_topk <= 0:
            short_scores = pd.Series(dtype=float)

        exchange = trade_exchange or getattr(self, "trade_exchange", None)
        t_start = kwargs.get("trade_start_time") or kwargs.get("t_start")
        t_end = kwargs.get("trade_end_time") or kwargs.get("t_end") or t_start
        long_scores = self._filter_tradeable_scores(long_scores, exchange, t_start, t_end, Order.BUY, "多头")
        short_scores = self._filter_tradeable_scores(short_scores, exchange, t_start, t_end, Order.SELL, "空头")

        # --- 融资融券动态授信额度 ---
        # 1. 获取当前净值
        current_equity = None
        try:
            tp = getattr(self, "trade_position", None)
            if tp is not None:
                pos_obj = tp.get_current_position() if hasattr(tp, "get_current_position") else tp
                val = float(pos_obj.calculate_value())
                if val > 0:
                    current_equity = val
        except Exception:
            pass

        # 记录初始本金（首次调用时）
        if current_equity is not None:
            if not hasattr(self, "_initial_capital") or self._initial_capital is None:
                self._initial_capital = current_equity

        # 2. 动态授信比率：额度 = 权益余额的 100%（1:1 配资）
        #    ratio = current_equity / initial_capital，随盈亏等比调整：
        #      - 盈利时 ratio > 1.0 → 信用额度随盈利增加
        #      - 亏损时 ratio < 1.0 → 信用额度随亏损收缩，避免过度杠杆
        _ic = getattr(self, "_initial_capital", None)
        if current_equity and _ic and _ic > 0:
            ratio = current_equity / _ic
        else:
            ratio = 1.0

        # 3. 动态约束各侧敞口
        req_long = self.long_exposure
        req_short = self.short_exposure

        # 融券做空占用信用额度，不超过当前授信上限
        actual_short_exposure = min(req_short, ratio)
        # 剩余信用额度可用于融资加多
        rem_ratio = max(0.0, ratio - actual_short_exposure)
        # 融资做多上限 = 自有本金(1.0) + 剩余信用额度
        max_long_exposure = 1.0 + rem_ratio
        actual_long_exposure = min(req_long, max_long_exposure)

        # 4. 全局杠杆保护
        max_total_leverage = getattr(self, "max_leverage", 5.0)
        if actual_long_exposure + actual_short_exposure > max_total_leverage:
            scale = max_total_leverage / (actual_long_exposure + actual_short_exposure)
            actual_long_exposure *= scale
            actual_short_exposure *= scale

        StructuredTaskLogger(
            logger,
            "redis-long-short-topk",
            {"backtest_id": getattr(self, "backtest_id", None)},
        ).info(
            "exposure",
            "动态授信额度计算完成",
            equity=current_equity or 0,
            initial_capital=_ic or 0,
            ratio=f"{ratio:.2f}",
            long_exposure=f"{actual_long_exposure:.2f}",
            short_exposure=f"{actual_short_exposure:.2f}",
        )

        # 5. 多空股票池重叠去重：同一标的不得同时出现在两侧
        overlap = set(long_scores.index) & set(short_scores.index)
        if overlap:
            for stock in overlap:
                long_val = long_scores.get(stock, 0.0)
                short_val = abs(short_scores.get(stock, 0.0))
                if long_val >= short_val:
                    short_scores = short_scores.drop(stock)
                else:
                    long_scores = long_scores.drop(stock)
            StructuredTaskLogger(
                logger,
                "redis-long-short-topk",
                {"backtest_id": getattr(self, "backtest_id", None)},
            ).info("overlap_removed", "去除多空重叠标的", count=len(overlap))

        long_weights = self._build_side_weights(long_scores, actual_long_exposure)
        short_weights = -self._build_side_weights(short_scores, actual_short_exposure)

        combined = pd.concat([long_weights, short_weights])
        if combined.empty:
            return {}
        return combined.to_dict()

    def _safe_generate_trade_decision(self, execute_result=None):
        """安全地生成交易决策，处理 get_deal_price 返回 None 的情况。"""
        try:
            return super().generate_trade_decision(execute_result)
        except TypeError as e:
            if "unsupported operand type(s) for /: 'float' and 'NoneType'" in str(e):
                StructuredTaskLogger(
                    logger,
                    "redis-long-short-topk",
                    {"backtest_id": getattr(self, "backtest_id", None)},
                ).warning("skip_trade_no_price", "Skip trade due to missing price data")
                return TradeDecisionWO([], self)
            raise

    def generate_trade_decision(self, execute_result=None):
        if hasattr(self, "check_account_stop_loss") and self.check_account_stop_loss():
            StructuredTaskLogger(
                logger,
                "redis-long-short-topk",
                {"backtest_id": getattr(self, "backtest_id", None)},
            ).warning("account_stop_loss", "Account stop-loss triggered. Liquidating.")
            return self._liquidate_all()

        if not self._should_rebalance(self.rebalance_days):
            return TradeDecisionWO([], self)
        return self._safe_generate_trade_decision(execute_result)

    def post_exe_step(self, execute_result=None):
        self.log_progress()
        self.log_executed_trades(execute_result)


class RedisSectorRotationStrategy(DynamicRiskMixin, TopkDropoutStrategy, RedisLoggerMixin):
    """
    行业轮动策略 (Sector Rotation)
    逻辑：
    1. 获取当前股票池中所有股票的行业分类。
    2. 计算各行业过去 N 天的平均累计收益（动量）。
    3. 选出 Top K 个强势行业。
    4. 仅保留属于这些强势行业的候选股票，再进行 TopK 选股。
    """

    def __init__(self, *args, **kwargs):
        self.init_redis(kwargs)
        self.init_dynamic_risk(kwargs)

        self.topk_sectors = int(kwargs.pop("topk_sectors", 5))
        self.lookback_days = int(kwargs.pop("lookback_days", 20))

        # 信号兼容处理：与 RedisRecordingStrategy、RedisWeightStrategy 保持一致。
        # RedisSectorRotationStrategy 继承 TopkDropoutStrategy，
        # qlib create_signal_from() 不接受 dict 类型，必须在此处提前实例化。
        if "signal" in kwargs:
            sig = kwargs["signal"]
            if isinstance(sig, dict) and "class" in sig:
                from qlib.utils import init_instance_by_config

                _log = StructuredTaskLogger(
                    logger,
                    "redis-sector-rotation",
                    {"backtest_id": getattr(self, "backtest_id", None)},
                )
                try:
                    kwargs["signal"] = init_instance_by_config(sig)
                    _log.info(
                        "signal_instantiated",
                        "RedisSectorRotationStrategy: signal dict 已实例化",
                        signal_class=sig.get("class"),
                    )
                except Exception as e:
                    _log.error(
                        "signal_instantiate_failed",
                        "RedisSectorRotationStrategy: signal 实例化失败，移除 signal 参数",
                        signal_class=sig.get("class"),
                        error=str(e),
                    )
                    kwargs.pop("signal", None)

        for k in _OUR_KWARGS:
            kwargs.pop(k, None)
        strip_unsupported_kwargs(type(self), kwargs, strategy_name="RedisSectorRotationStrategy")
        super().__init__(*args, **kwargs)

    def generate_trade_decision(self, execute_result=None):
        if hasattr(self, "check_account_stop_loss") and self.check_account_stop_loss():
            StructuredTaskLogger(
                logger,
                "redis-sector-rotation",
                {"backtest_id": getattr(self, "backtest_id", None)},
            ).info("account_stop_loss", "Account stop-loss triggered. Liquidating.")
            return self._liquidate_all()

        # 获取父类计算出的原始信号（通常是模型预测分）
        # TopkDropoutStrategy 不直接暴露 score，它是在 generate_trade_decision 内部获取 signal 的。
        # 因此我们需要 Hook 这里的逻辑，或者在获取 signal 后进行过滤。
        # 但 Qlib 的 TopkDropoutStrategy 逻辑比较紧凑，重写 generate_trade_decision 比较复杂。
        # 替代方案：我们可以重写 `get_candidate_inf` 或者在 `super().generate_trade_decision` 之后修改结果。
        # 但修改结果（Order 列表）很难，因为我们要修改的是"选股范围"。

        # 最佳切入点是 `get_pred_score` 如果父类有的话，或者利用 Python 的动态特性临时修改 signal。
        # 这里我们选择重写核心逻辑的一个简化版本：在每一步开始前，动态修改 self.signal (如果它是 dataframe)。
        # 但 self.signal 可能是 Dataset 对象，很难修改。

        # 妥协方案：完全重写 generate_trade_decision 中涉及信号获取的部分逻辑太重。
        # 我们采用 "后处理" 模式：
        # 1. 让父类生成 TopK 决策 (假设 topk 设置得稍大一点，比如 2*topk)。
        # 2. 我们检查这些决策涉及的股票，剔除不在强势行业的。
        # 3. 如果剔除后数量不足，这可能会导致仓位不足。

        # 更稳健的方案：在这一步动态计算行业动量，并生成一个 Mask。
        # 但由于无法轻易注入 Mask 到父类逻辑，我们这里实现一个简化的"动量因子加成"逻辑是不太行的。

        # 最终决定：我们不复用 TopkDropoutStrategy 的决策逻辑，而是自己实现一个简单的 SectorFilter 逻辑，
        # 然后手动构造 TradeDecision。这实际上把 TopkDropout 退化为了 SectorStrategy。

        return self._safe_generate_trade_decision(execute_result)

    def _get_sector_momentum(self, trade_date) -> list[str]:
        """计算强势行业列表"""
        try:
            # 1. 获取全市场行业数据 (假设字段名为 'industry' 或 'sector')
            # 注意：实际字段名取决于数据源 (Alpha360/Alpha158 通常不含行业，需额外数据)
            # 这里做一个容错：如果取不到行业，返回 None，策略退化为普通 TopK
            instruments = D.instruments("csi300")  # 默认用 CSI300 样本计算行业动量  # noqa: F841
            end_time = trade_date.strftime("%Y-%m-%d")  # noqa: F841
            start_time = (trade_date - pd.Timedelta(days=self.lookback_days * 2)).strftime("%Y-%m-%d")  # noqa: F841

            # 尝试获取行业分类 (CSSws1 是申万一级行业常用名)
            # 如果没有，尝试 'industry'
            fields = ["$close", "$factor.industry"]  # noqa: F841
            # 注意：$factor.industry 这种写法取决于 dataset 构建

            # 由于无法确定具体字段，且 D.features 可能很慢。
            # 如果没有行业数据，我们 log warning 并跳过
            return None
        except Exception:
            return None


class RedisSectorMomentumLeaderCoreStrategy(
    DynamicRiskMixin, TopkDropoutStrategy, RedisLoggerMixin
):
    """Dedicated stateful execution strategy for sector momentum signals."""

    def __init__(self, *args, **kwargs):
        self.init_redis(kwargs)
        self.init_dynamic_risk(kwargs)
        self.candidate_details = kwargs.pop("candidate_details", {})
        self.board_states = kwargs.pop("board_states", {})
        self.signal_metadata = kwargs.pop("signal_metadata", {})
        self.max_holding_days = int(kwargs.pop("max_holding_days", 10))
        self.rebalance_days = int(kwargs.pop("rebalance_days", 1))
        self.board_exit_rank = int(kwargs.pop("board_exit_rank", 20))
        self.board_exit_days = int(kwargs.pop("board_exit_days", 2))
        self.board_score_drop_exit = float(kwargs.pop("board_score_drop_exit", 20))
        self.leader_gap_down = float(kwargs.pop("leader_gap_down", -0.03))
        self.leader_gap_up = float(kwargs.pop("leader_gap_up", 0.05))
        self.core_gap_down = float(kwargs.pop("core_gap_down", -0.02))
        self.core_gap_up = float(kwargs.pop("core_gap_up", 0.03))
        self.stop_loss = float(kwargs.pop("stop_loss", -0.03))
        self.leader_trailing_stop = float(kwargs.pop("leader_trailing_stop", 0.06))
        self.core_trailing_stop = float(kwargs.pop("core_trailing_stop", 0.05))
        self.max_stock_weight = float(kwargs.pop("max_stock_weight", 0.10))
        self.max_board_weight = float(kwargs.pop("max_board_weight", 0.20))
        self.weak_market_position = float(kwargs.pop("weak_market_position", 0.50))
        self.market_breadth_reduce = float(kwargs.pop("market_breadth_reduce", 0.40))
        self.market_breadth_pause = float(kwargs.pop("market_breadth_pause", 0.30))
        self._entry_trade_step: dict[str, int] = {}
        self._position_context: dict[str, dict[str, Any]] = {}
        self._board_rank_breach_days: dict[str, int] = defaultdict(int)
        self._forced_exit_blocked: dict[str, dict[str, Any]] = {}
        self._trade_step_by_date: dict[str, int] = {}
        self.execution_diagnostics = self.signal_metadata.setdefault(
            "execution_diagnostics", []
        )

        kwargs = _resolve_signal_kwarg(kwargs)
        sector_only = {
            "board_universe", "topk_sectors", "topk_stocks", "lookback_days",
            "min_board_members", "min_board_coverage", "crowding_warning_quantile",
            "crowding_overheat_quantile", "crowding_penalty_max",
            "launch_breadth_min", "launch_breadth_max", "launch_breadth_delta_min",
            "launch_limit_count_min", "launch_limit_count_max",
            "launch_limit_ratio_min", "launch_limit_ratio_max",
            "launch_amount_ratio_min", "launch_amount_ratio_max",
            "launch_relative_return_min", "launch_relative_return_max",
            "diffusion_breadth_min", "diffusion_limit_count_min",
            "diffusion_limit_ratio_min", "diffusion_amount_quantile_min",
            "core_start_amount_ratio_min", "overheat_breadth_min",
            "overheat_relative_return_min", "retreat_breadth_max",
            "retreat_breadth_delta_max", "retreat_relative_return_max",
            "leader_float_mv_min", "leader_float_mv_max", "leader_amount_min",
            "leader_turnover_min", "leader_turnover_max", "leader_rps_min",
            "leader_amount_ratio_min", "leader_amount_ratio_max",
            "leader_return_3d_min", "leader_return_3d_max",
            "leader_upper_shadow_amount_ratio", "leader_upper_shadow_ratio",
            "core_float_mv_min", "core_mv_top_quantile", "core_amount_min",
            "core_volatility_min", "core_volatility_max", "core_drawdown_min",
            "core_amount_ratio_min", "core_amount_ratio_max", "core_return_5d_max",
        }
        for key in sector_only | _OUR_KWARGS:
            kwargs.pop(key, None)
        kwargs.setdefault("only_tradable", True)
        super().__init__(*args, **kwargs)

    def reset(self, *args, **kwargs):
        self._entry_trade_step.clear()
        self._position_context.clear()
        self._board_rank_breach_days.clear()
        self._forced_exit_blocked.clear()
        self._trade_step_by_date.clear()
        self.execution_diagnostics.clear()
        self.reset_dynamic_risk()
        try:
            return super().reset(*args, **kwargs)
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            return super().reset()

    @staticmethod
    def _date_key(value: Any) -> str:
        return str(pd.Timestamp(value).date())

    def _current_trade_context(self) -> tuple[int, Any, Any, str]:
        trade_step = int(self.trade_calendar.get_trade_step())
        start_time, end_time = self.trade_calendar.get_step_time(trade_step)
        return trade_step, start_time, end_time, self._date_key(start_time)

    def _append_execution_diagnostic(self, event: str, **values: Any) -> None:
        record = {"event": event, **values}
        if self.execution_diagnostics and self.execution_diagnostics[-1] == record:
            return
        self.execution_diagnostics.append(record)
        if len(self.execution_diagnostics) > 500:
            del self.execution_diagnostics[:-500]

    def _record_fills(self, execute_result: Any, trade_step: int | None = None) -> None:
        if not execute_result:
            return
        for item in execute_result:
            if not isinstance(item, tuple) or not item:
                continue
            order = item[0]
            if float(getattr(order, "deal_amount", 0) or 0) <= 0:
                continue
            stock = str(order.stock_id)
            order_date = self._date_key(getattr(order, "start_time", ""))
            actual_trade_step = self._trade_step_by_date.get(order_date, trade_step)
            if actual_trade_step is None:
                continue
            if order.direction == OrderDir.BUY:
                self._entry_trade_step.setdefault(stock, actual_trade_step)
                detail = self.candidate_details.get(
                    order_date, {}
                ).get(stock, {})
                if stock not in self._position_context:
                    trade_price = item[3] if len(item) > 3 else None
                    self._position_context[stock] = {
                        "board_code": detail.get("board_code"),
                        "entry_board_score": detail.get("board_score"),
                        "role": detail.get("role"),
                        "signal_date": detail.get("signal_date"),
                        "entry_price": float(trade_price)
                        if trade_price is not None
                        else None,
                        "highest_close": float(trade_price)
                        if trade_price is not None
                        else None,
                        "below_ma5_days": 0,
                    }
            elif order.direction == OrderDir.SELL:
                try:
                    amount = float(self.trade_position.get_stock_amount(stock))
                except Exception as exc:
                    self._append_execution_diagnostic(
                        "position_amount_read_failed",
                        date=order_date,
                        stock=stock,
                        error_type=type(exc).__name__,
                    )
                    StructuredTaskLogger(
                        logger,
                        "redis-sector-momentum-leader-core",
                        {"backtest_id": getattr(self, "backtest_id", None)},
                    ).warning(
                        "position_amount_read_failed",
                        "卖出成交后读取剩余持仓失败，保留持仓跟踪状态",
                        stock=stock,
                        error=exc,
                    )
                    continue
                if amount > 0:
                    continue
                blocked = self._forced_exit_blocked.get(stock)
                if blocked:
                    self._append_execution_diagnostic(
                        "forced_exit_completed",
                        date=order_date,
                        stock=stock,
                        reason=blocked["reason"],
                        retries=blocked["retries"],
                    )
                self._entry_trade_step.pop(stock, None)
                self._position_context.pop(stock, None)
                self._board_rank_breach_days.pop(stock, None)
                self._forced_exit_blocked.pop(stock, None)

    def _board_exit_reason(self, stock: str, date_key: str) -> str | None:
        context = self._position_context.get(stock, {})
        board_code = context.get("board_code")
        state = self.board_states.get(date_key, {}).get(str(board_code), {})
        if not state or state.get("evaluable") is False:
            return None
        phase = state.get("phase")
        if phase in {"retreat", "overheated"}:
            return f"board_phase_{phase}"
        if context.get("role") == "leader" and phase == "diffusion":
            return "leader_to_diffusion"
        rank = state.get("rank")
        if rank is not None and float(rank) > self.board_exit_rank:
            self._board_rank_breach_days[stock] += 1
        else:
            self._board_rank_breach_days[stock] = 0
        if self._board_rank_breach_days[stock] >= self.board_exit_days:
            return "board_rank_exit"
        current_score = state.get("board_score", state.get("score"))
        entry_score = context.get("entry_board_score")
        if current_score is not None and entry_score is not None:
            if float(entry_score) - float(current_score) >= self.board_score_drop_exit:
                return "board_score_drop"
        return None

    def _stock_exit_reason(
        self, stock: str, snapshot: dict[str, dict[str, float]]
    ) -> str | None:
        values = snapshot.get(stock, {})
        close = values.get("close")
        ma5 = values.get("ma5")
        ma20 = values.get("ma20")
        context = self._position_context.get(stock, {})
        if close is None or not np.isfinite(close) or close <= 0:
            return None
        highest = context.get("highest_close")
        highest = max(float(highest or close), float(close))
        context["highest_close"] = highest
        entry_price = context.get("entry_price")
        if entry_price and float(close) / float(entry_price) - 1 <= self.stop_loss:
            return "stop_loss"
        trailing_stop = (
            self.leader_trailing_stop
            if context.get("role") == "leader"
            else self.core_trailing_stop
        )
        if highest > 0 and (highest - float(close)) / highest >= trailing_stop:
            return "trailing_stop"
        if ma20 is not None and np.isfinite(ma20) and float(close) < float(ma20):
            return "trend_below_ma20"
        if ma5 is not None and np.isfinite(ma5) and float(close) < float(ma5):
            context["below_ma5_days"] = int(context.get("below_ma5_days", 0)) + 1
        else:
            context["below_ma5_days"] = 0
        if context["below_ma5_days"] >= 2:
            return "trend_below_ma5_twice"
        return None

    def _forced_exit_reasons(
        self,
        trade_step: int,
        date_key: str,
        snapshot: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, str]:
        reasons: dict[str, str] = {}
        stocks = getattr(self, "trade_position", None)
        stocks = stocks.get_stock_list() if stocks is not None else []
        for stock in stocks:
            stock = str(stock)
            blocked = self._forced_exit_blocked.get(stock)
            if blocked:
                reasons[stock] = str(blocked["reason"])
                continue
            entry_step = self._entry_trade_step.get(stock)
            if (
                entry_step is not None
                and trade_step - entry_step >= self.max_holding_days
            ):
                reasons[stock] = "max_holding_days"
                continue
            stock_reason = self._stock_exit_reason(stock, snapshot or {})
            if stock_reason:
                reasons[stock] = stock_reason
                continue
            board_reason = self._board_exit_reason(stock, date_key)
            if board_reason:
                reasons[stock] = board_reason
        return reasons

    def _passes_gap_filter(
        self,
        order: Order,
        date_key: str,
        start_time: Any,
        end_time: Any,
        snapshot: dict[str, dict[str, float]],
    ) -> bool:
        detail = self.candidate_details.get(date_key, {}).get(str(order.stock_id), {})
        previous_close = snapshot.get(str(order.stock_id), {}).get("close")
        try:
            open_price = float(
                self._execution_open_price(str(order.stock_id), start_time, end_time)
            )
            previous_close = float(previous_close)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(open_price) or not np.isfinite(previous_close):
            return False
        if open_price <= 0 or previous_close <= 0:
            return False
        gap = open_price / previous_close - 1
        role = detail.get("role")
        lower, upper = (
            (self.leader_gap_down, self.leader_gap_up)
            if role == "leader"
            else (self.core_gap_down, self.core_gap_up)
        )
        return lower <= float(gap) <= upper

    def _execution_open_price(
        self, stock: str, start_time: Any, end_time: Any
    ) -> float | None:
        exchange = self.trade_exchange
        quote = getattr(exchange, "quote", None)
        field = getattr(exchange, "buy_price", "$open")
        if quote is not None and hasattr(quote, "get_data"):
            value = quote.get_data(
                stock, start_time, end_time, field=field, method="ts_data_last"
            )
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            return value if np.isfinite(value) and value > 0 else None
        return exchange.get_deal_price(
            stock_id=stock,
            start_time=start_time,
            end_time=end_time,
            direction=OrderDir.BUY,
        )

    def _load_prior_close_snapshot(
        self, stocks: list[str], trade_date: Any
    ) -> dict[str, dict[str, float]]:
        if not stocks:
            return {}
        end_date = pd.Timestamp(trade_date).normalize() - pd.Timedelta(days=1)
        start_date = end_date - pd.Timedelta(days=60)
        fields = ["$close", "Mean($close,5)", "Mean($close,20)"]
        try:
            frame = D.features(
                stocks,
                fields,
                start_time=str(start_date.date()),
                end_time=str(end_date.date()),
            )
        except Exception as exc:
            StructuredTaskLogger(
                logger,
                "redis-sector-momentum-leader-core",
                {"backtest_id": getattr(self, "backtest_id", None)},
            ).warning("prior_snapshot_failed", "前一交易日行情读取失败", error=exc)
            return {}
        if frame is None or frame.empty:
            return {}
        snapshot: dict[str, dict[str, float]] = {}
        for stock in stocks:
            try:
                rows = frame.xs(stock, level="instrument").sort_index()
                row = rows.iloc[-1]
                snapshot[stock] = {
                    "close": float(row[fields[0]]),
                    "ma5": float(row[fields[1]]),
                    "ma20": float(row[fields[2]]),
                }
            except (KeyError, IndexError, TypeError, ValueError):
                continue
        return snapshot

    def _market_position_limit(self, date_key: str) -> float:
        breadth = self.signal_metadata.get("market_breadth_by_date", {}).get(date_key)
        if breadth is None or not np.isfinite(float(breadth)):
            return 0.0
        if float(breadth) < self.market_breadth_pause:
            return 0.0
        if float(breadth) < self.market_breadth_reduce:
            return self.weak_market_position
        return 1.0

    def _position_equity(self) -> float:
        try:
            return float(self.trade_position.calculate_value())
        except Exception as exc:
            self._append_execution_diagnostic(
                "position_equity_read_failed",
                error_type=type(exc).__name__,
            )
            StructuredTaskLogger(
                logger,
                "redis-sector-momentum-leader-core",
                {"backtest_id": getattr(self, "backtest_id", None)},
            ).warning(
                "position_equity_read_failed",
                "读取账户权益失败，禁止新增买入",
                error=exc,
            )
            return 0.0

    def _cap_buy_order(
        self,
        order: Order,
        date_key: str,
        start_time: Any,
        end_time: Any,
        board_values: dict[str, float],
        total_stock_value: float,
    ) -> Order | None:
        stock = str(order.stock_id)
        detail = self.candidate_details.get(date_key, {}).get(stock, {})
        board_code = str(detail.get("board_code") or "")
        try:
            open_price = float(
                self.trade_exchange.get_deal_price(
                    stock_id=stock,
                    start_time=start_time,
                    end_time=end_time,
                    direction=OrderDir.BUY,
                )
            )
            equity = self._position_equity()
        except (TypeError, ValueError):
            return None
        if equity <= 0 or open_price <= 0 or not board_code:
            return None
        position_limit = self._market_position_limit(date_key)
        if position_limit <= 0:
            return None
        existing_amount = float(self.trade_position.get_stock_amount(stock) or 0)
        existing_value = existing_amount * open_price
        stock_room = equity * self.max_stock_weight - existing_value
        board_room = equity * self.max_board_weight - board_values.get(board_code, 0.0)
        portfolio_room = equity * position_limit - total_stock_value
        requested_value = float(order.amount) * open_price
        allowed_value = min(requested_value, stock_room, board_room, portfolio_room)
        if allowed_value <= 0:
            return None
        factor = self.trade_exchange.get_factor(
            stock_id=stock, start_time=start_time, end_time=end_time
        )
        amount = self.trade_exchange.round_amount_by_trade_unit(
            allowed_value / open_price, factor
        )
        if amount <= 0:
            return None
        board_values[board_code] = board_values.get(board_code, 0.0) + amount * open_price
        order.amount = amount
        return order

    def _generate_execution_date_decision(
        self, start_time: Any, end_time: Any
    ) -> TradeDecisionWO:
        """Use the already lagged execution-date score without another Qlib shift."""
        pred_score = self.signal.get_signal(start_time=start_time, end_time=end_time)
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None or getattr(pred_score, "empty", False):
            return TradeDecisionWO([], self)
        ranked = pred_score.sort_values(ascending=False)
        desired: list[str] = []
        for stock in ranked.index:
            stock = str(stock)
            if self.trade_exchange.is_stock_tradable(
                stock_id=stock, start_time=start_time, end_time=end_time
            ):
                desired.append(stock)
            if len(desired) >= self.topk:
                break
        current_temp = copy.deepcopy(self.trade_position)
        current_stocks = [str(stock) for stock in current_temp.get_stock_list()]
        orders: list[Order] = []
        cash = float(current_temp.get_cash())
        for stock in current_stocks:
            if stock in desired or not self.trade_exchange.is_stock_tradable(
                stock_id=stock,
                start_time=start_time,
                end_time=end_time,
                direction=OrderDir.SELL,
            ):
                continue
            order = Order(
                stock_id=stock,
                amount=current_temp.get_stock_amount(stock),
                start_time=start_time,
                end_time=end_time,
                direction=OrderDir.SELL,
            )
            if self.trade_exchange.check_order(order):
                orders.append(order)
                trade_value, trade_cost, _price = self.trade_exchange.deal_order(
                    order, position=current_temp
                )
                cash += float(trade_value) - float(trade_cost)
        buy = [stock for stock in desired if stock not in current_stocks]
        allocation = cash / len(buy) if buy else 0.0
        for stock in buy:
            price = self.trade_exchange.get_deal_price(
                stock_id=stock,
                start_time=start_time,
                end_time=end_time,
                direction=OrderDir.BUY,
            )
            if price is None or float(price) <= 0:
                continue
            factor = self.trade_exchange.get_factor(
                stock_id=stock, start_time=start_time, end_time=end_time
            )
            amount = self.trade_exchange.round_amount_by_trade_unit(
                allocation / float(price), factor
            )
            if amount > 0:
                orders.append(
                    Order(
                        stock_id=stock,
                        amount=amount,
                        start_time=start_time,
                        end_time=end_time,
                        direction=OrderDir.BUY,
                    )
                )
        return TradeDecisionWO(orders, self)

    def generate_trade_decision(self, execute_result=None):
        trade_step, start_time, end_time, date_key = self._current_trade_context()
        self._trade_step_by_date[date_key] = trade_step
        self._record_fills(execute_result, max(0, trade_step - 1))
        held_stocks = [str(stock) for stock in self.trade_position.get_stock_list()]
        candidate_stocks = list(self.candidate_details.get(date_key, {}))
        snapshot = self._load_prior_close_snapshot(
            sorted(set(held_stocks + candidate_stocks)), start_time
        )
        forced = self._forced_exit_reasons(trade_step, date_key, snapshot)
        if self.rebalance_days > 1 and trade_step % self.rebalance_days:
            decision = TradeDecisionWO([], self)
        else:
            decision = self._generate_execution_date_decision(start_time, end_time)
        total_stock_value = 0.0
        board_values: dict[str, float] = defaultdict(float)
        for stock in held_stocks:
            price = snapshot.get(stock, {}).get("close")
            if price is None or not np.isfinite(price) or price <= 0:
                continue
            value = float(self.trade_position.get_stock_amount(stock) or 0) * float(price)
            total_stock_value += value
            board_code = str(self._position_context.get(stock, {}).get("board_code") or "")
            if board_code:
                board_values[board_code] += value
        orders = []
        for order in decision.get_decision():
            stock = str(order.stock_id)
            if order.direction == OrderDir.BUY:
                if stock in forced or not self._passes_gap_filter(
                    order, date_key, start_time, end_time, snapshot
                ):
                    continue
                order = self._cap_buy_order(
                    order,
                    date_key,
                    start_time,
                    end_time,
                    board_values,
                    total_stock_value,
                )
                if order is None:
                    continue
                total_stock_value += float(order.amount) * float(
                    self.trade_exchange.get_deal_price(
                        stock_id=stock,
                        start_time=start_time,
                        end_time=end_time,
                        direction=OrderDir.BUY,
                    )
                )
            orders.append(order)
        existing_sells = {
            str(order.stock_id)
            for order in orders
            if order.direction == OrderDir.SELL
        }
        for stock, reason in forced.items():
            if stock in existing_sells:
                continue
            tradable = self.trade_exchange.is_stock_tradable(
                stock_id=stock,
                start_time=start_time,
                end_time=end_time,
                direction=OrderDir.SELL,
            )
            if not tradable:
                blocked = self._forced_exit_blocked.get(stock)
                if blocked is None:
                    blocked = {
                        "reason": reason,
                        "blocked_since": date_key,
                        "retries": 0,
                    }
                    self._forced_exit_blocked[stock] = blocked
                    event = "forced_exit_blocked"
                else:
                    blocked["retries"] = int(blocked["retries"]) + 1
                    event = "forced_exit_retry"
                self._append_execution_diagnostic(
                    event,
                    date=date_key,
                    stock=stock,
                    reason=blocked["reason"],
                    retries=blocked["retries"],
                )
                continue
            amount = self.trade_position.get_stock_amount(stock)
            if amount > 0:
                orders.append(
                    Order(
                        stock_id=stock,
                        amount=amount,
                        start_time=start_time,
                        end_time=end_time,
                        direction=OrderDir.SELL,
                    )
                )
                blocked = self._forced_exit_blocked.get(stock)
                if blocked and blocked.get("recovered_order_date") != date_key:
                    blocked["recovered_order_date"] = date_key
                    self._append_execution_diagnostic(
                        "forced_exit_recovered",
                        date=date_key,
                        stock=stock,
                        reason=blocked["reason"],
                        retries=blocked["retries"],
                    )
        return TradeDecisionWO(orders, self)

    def post_exe_step(self, execute_result=None):
        trade_step = int(self.trade_calendar.get_trade_step())
        self._record_fills(execute_result, trade_step)
        self.log_progress()
        self.log_executed_trades(execute_result)


class RedisStopLossStrategy(DynamicRiskMixin, TopkDropoutStrategy, RedisLoggerMixin):
    """
    止损止盈策略 (Stop Loss / Take Profit)
    逻辑：
    1. 在内存中维护每个股票的持仓成本 (avg_price)。
    2. 每日检查 (current_price - avg_price) / avg_price。
    3. 若低于 stop_loss 或高于 take_profit，强制卖出。
    4. 正常的 TopK 调仓逻辑叠加在止损逻辑之后 (即止损优先)。
    """

    def __init__(self, *args, **kwargs):
        self.init_redis(kwargs)
        self.init_dynamic_risk(kwargs)
        self.stop_loss = float(kwargs.pop("stop_loss", -0.08))
        self.take_profit = float(kwargs.pop("take_profit", 0.15))
        self.holding_cost = {}
        for k in _OUR_KWARGS:
            kwargs.pop(k, None)
        # 安全兜底：移除所有 Qlib BaseStrategy 不认识的剩余参数，避免意外传递
        strip_unsupported_kwargs(type(self), kwargs, strategy_name="RedisStopLossStrategy")
        # 全局规则：选股时剔除涨停/跌停/停牌股
        kwargs.setdefault("only_tradable", True)
        super().__init__(*args, **kwargs)

    def generate_trade_decision(self, execute_result=None):
        if hasattr(self, "check_account_stop_loss") and self.check_account_stop_loss():
            StructuredTaskLogger(
                logger,
                "redis-stop-loss",
                {"backtest_id": getattr(self, "backtest_id", None)},
            ).info("account_stop_loss", "Account stop-loss triggered. Liquidating.")
            return self._liquidate_all()

        # 1. 更新持仓成本 (利用上一步的执行结果)
        # execute_result 结构: [(Order, trade_val, trade_cost, trade_price), ...]
        if execute_result:
            for item in execute_result:
                if not item:
                    continue
                order, _, _, trade_price = item
                if order.deal_amount > 0:
                    stock = str(order.stock_id)
                    if order.direction == OrderDir.BUY:
                        # 简化处理：每次买入都更新为最新成交价 (或者可以使用移动平均)
                        # 为了严格止损，这里使用"最后一次买入价"作为基准可能更敏感，
                        # 但"平均成本"更符合会计逻辑。这里暂用最新买入价 (Last Buy Price)。
                        self.holding_cost[stock] = trade_price
                    elif order.direction == OrderDir.SELL:
                        # 卖出不更新成本，除非清仓
                        # 如果需要判断是否清仓，需要访问 Current Position。
                        pass

        # 2. 获取当前持仓和价格
        # Qlib 的 self.trade_position 本身就是一个 Position 对象，不需要调用 get_current_position()
        current_position = getattr(self, "trade_position", None)
        if current_position is None:
            return TradeDecisionWO([], self)

        current_stocks = current_position.get_stock_list()

        # 获取当前时间
        if hasattr(self.trade_calendar, "get_step_time"):
            trade_step = self.trade_calendar.get_trade_step()
            trade_date, _ = self.trade_calendar.get_step_time(trade_step)
        else:
            trade_step = getattr(self, "trade_step", 0)
            trade_date = self.trade_calendar[trade_step]

        # 强制止损列表
        force_sell_stocks = set()

        if current_stocks:
            try:
                # 批量获取当前价格
                current_prices = D.features(
                    current_stocks,
                    ["$close"],
                    start_time=trade_date.strftime("%Y-%m-%d"),
                    end_time=trade_date.strftime("%Y-%m-%d"),
                )
                if current_prices is not None and not current_prices.empty:
                    # current_prices index: (instrument, date)
                    for stock in current_stocks:
                        try:
                            # 查找该股票的最新价格
                            # 安全获取：先看该股票是否在 dataframe index level 0
                            if stock not in current_prices.index.get_level_values("instrument"):
                                continue

                            price = current_prices.xs(stock, level="instrument")["$close"].iloc[-1]
                            cost = self.holding_cost.get(stock)

                            if cost and cost > 0:
                                ret = (price - cost) / cost
                                if ret <= self.stop_loss:
                                    StructuredTaskLogger(
                                        logger,
                                        "redis-stop-loss",
                                        {"backtest_id": getattr(self, "backtest_id", None)},
                                    ).info(
                                        "stop_loss_triggered",
                                        "Stop loss triggered",
                                        stock=stock,
                                        return_rate=f"{ret:.2%}",
                                        price=price,
                                        cost=cost,
                                    )
                                    force_sell_stocks.add(stock)
                                    # 止损后清除成本记录，防止反复触发（虽然 generate_trade_decision 卖出后仓位也没了）
                                    # 但在卖出执行前，保持记录
                                elif ret >= self.take_profit:
                                    StructuredTaskLogger(
                                        logger,
                                        "redis-stop-loss",
                                        {"backtest_id": getattr(self, "backtest_id", None)},
                                    ).info(
                                        "take_profit_triggered",
                                        "Take profit triggered",
                                        stock=stock,
                                        return_rate=f"{ret:.2%}",
                                        price=price,
                                        cost=cost,
                                    )
                                    force_sell_stocks.add(stock)
                        except Exception:
                            continue
            except Exception as e:
                StructuredTaskLogger(
                    logger,
                    "redis-stop-loss",
                    {"backtest_id": getattr(self, "backtest_id", None)},
                ).warning("check_failed", "StopLoss check failed", error=e)

        # 3. 生成常规决策
        decision = super().generate_trade_decision(execute_result)

        # 4. 注入强制卖出单
        if force_sell_stocks:
            new_orders = []
            # 保留原有的非卖出单，或者根据逻辑调整
            # 简单逻辑：如果原决策里有买入这些股票的，取消买入；
            # 额外添加卖出这些股票的订单 (Sell All)

            # 首先处理原决策
            # decision 可能是 TradeDecisionWO (list of orders)
            original_orders = decision.get_decision()

            for order in original_orders:
                stock = str(order.stock_id)
                if stock in force_sell_stocks:
                    # 如果原计划是买入止损股，取消买入
                    if order.direction == OrderDir.BUY:
                        continue
                    # 如果原计划已经是卖出，保留（或者检查数量是否足够全卖）
                    new_orders.append(order)
                else:
                    new_orders.append(order)

            # 添加强制卖出单 (Target 0)
            # TopKDropout 也就是 Target 模式，但它生成的 Order 已经是具体的 Buy/Sell amount
            # 我们最好生成 "Target 0" 的 Order。
            # 但 Order 对象通常是 amount 模式。

            for stock in force_sell_stocks:
                # 检查 new_orders 里是否已经有该股票的卖单
                has_sell = any(str(o.stock_id) == stock and o.direction == OrderDir.SELL for o in new_orders)
                if not has_sell:
                    # 构造全卖单
                    # 需要知道持仓数量
                    amount = current_position.get_stock_amount(stock)
                    if amount > 0:
                        new_orders.append(
                            Order(
                                stock_id=stock,
                                amount=amount,
                                direction=OrderDir.SELL,
                                start_time=trade_date,
                                end_time=trade_date,
                            )
                        )
                        # 清除成本记录
                        self.holding_cost.pop(stock, None)

            # 替换决策列表
            # TradeDecisionWO 只需要 list[Order]
            decision = TradeDecisionWO(new_orders, decision.strategy)

        return decision

    def post_exe_step(self, execute_result=None):
        self.log_progress()
        self.log_executed_trades(execute_result)


class RedisCrashBuyDipStrategy(DynamicRiskMixin, WeightStrategyBase, RedisLoggerMixin):
    """
    大盘暴跌抄底策略 (Crash Buy-the-Dip)

    基于 WeightStrategyBase，通过 generate_target_weight_position 返回目标权重。
    """

    def __init__(self, *args, **kwargs):
        self.init_redis(kwargs)
        self.init_dynamic_risk(kwargs)

        self.crash_threshold_points = float(kwargs.pop("crash_threshold_points", 100))
        self.crash_threshold_pct = float(kwargs.pop("crash_threshold_pct", 0.025))
        self.top_k = int(kwargs.pop("top_k", 5))
        self.hold_days = int(kwargs.pop("hold_days", 5))
        self.take_profit = float(kwargs.pop("take_profit", 0.08))
        self.stop_loss = float(kwargs.pop("stop_loss", -0.05))
        self.max_wait_days = int(kwargs.pop("max_wait_days", 3))
        self.min_oversold_margin = float(kwargs.pop("min_oversold_margin", 0.01))
        self.index_scale = float(kwargs.pop("index_scale", 1000))
        self.benchmark = kwargs.pop("benchmark", "SH000300")
        self.trend_window = int(kwargs.pop("trend_window", 20))
        self.ma_fast = int(kwargs.pop("ma_fast", 5))
        self.ma_slow = int(kwargs.pop("ma_slow", 20))
        self.vol_lookback = int(kwargs.pop("vol_lookback", 10))
        self.rebalance_days = int(kwargs.pop("rebalance_days", 1))

        for k in list(kwargs.keys()):
            if k in _OUR_KWARGS or k == "rebalance_days":
                kwargs.pop(k, None)

        self._crash_dates: set = set()
        self._crash_info: dict = {}
        self._factor_df: pd.DataFrame | None = None
        self._all_dates: list = []
        self._instruments: list = []
        self._buy_plan: dict = {}
        self._positions: dict = {}
        self._initialized = False

        super().__init__(*args, **kwargs)

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        cal = getattr(getattr(self, "trade_exchange", None), "_trade_calendar", None)
        if cal is None:
            tc = getattr(getattr(self, "trade_exchange", None), "trade_calendar", None)
            if tc is not None and hasattr(tc, "_calendar"):
                cal = tc._calendar
        if cal is None:
            return
        self._all_dates = [pd.Timestamp(d) for d in cal]
        if not self._all_dates:
            return

        start_str = self._all_dates[0].strftime("%Y-%m-%d")
        end_str = self._all_dates[-1].strftime("%Y-%m-%d")

        try:
            self._instruments = list(D.instruments("csi300"))
        except Exception:
            return

        try:
            idx = D.features([self.benchmark], ["$close", "$change"], start_time=start_str, end_time=end_str).copy()
            idx["point_change"] = idx["$close"].diff() * self.index_scale
            for ridx, row in idx[(idx["point_change"] < -self.crash_threshold_points) | (idx["$change"] < -self.crash_threshold_pct)].iterrows():
                dt = pd.Timestamp(ridx[1]) if isinstance(ridx, tuple) else pd.Timestamp(ridx)
                self._crash_dates.add(dt)
                self._crash_info[dt] = {"point_change": float(row["point_change"]), "pct_change": float(row["$change"])}
        except Exception:
            pass

        if self._crash_dates:
            try:
                df = D.features(self._instruments, ["$close", "$open", "$high", "$low", "$volume", "$change"], start_time=start_str, end_time=end_str)
                if df is not None and not df.empty:
                    for w in [5, 10, self.trend_window, 30]:
                        df[f"ROC{w}"] = df.groupby(level="instrument")["$close"].pct_change(w)
                    df[f"MA{self.ma_fast}"] = df.groupby(level="instrument")["$close"].transform(lambda x: x.rolling(self.ma_fast).mean())
                    df[f"MA{self.ma_slow}"] = df.groupby(level="instrument")["$close"].transform(lambda x: x.rolling(self.ma_slow).mean())
                    df[f"VMA{self.vol_lookback}"] = df.groupby(level="instrument")["$volume"].transform(lambda x: x.rolling(self.vol_lookback).mean())
                    df["VOL_RATIO"] = df["$volume"] / (df[f"VMA{self.vol_lookback}"] + 1e-8)
                    df["KMID"] = (df["$close"] - df["$open"]) / (df["$open"] + 1e-8)
                    df["LOWER_SHADOW"] = (pd.concat([df["$close"], df["$open"]], axis=1).max(axis=1) - df["$low"]) / (df["$low"] + 1e-8)
                    self._factor_df = df
            except Exception:
                pass
        self._initialized = True

    def _find_oversold(self, crash_date, idx_pct):
        if self._factor_df is None or self._factor_df.empty:
            return []
        try:
            dd = self._factor_df.xs(crash_date, level=1)
        except KeyError:
            return []
        if dd.empty:
            return []
        mask = (dd.get(f"ROC{self.trend_window}", pd.Series(dtype=float)) > 0) & \
               (dd.get(f"MA{self.ma_fast}", pd.Series(dtype=float)) > dd.get(f"MA{self.ma_slow}", pd.Series(dtype=float))) & \
               (dd["$change"] > -0.095) & (dd["$change"] < 0.095) & \
               (dd["$change"] < idx_pct - self.min_oversold_margin) & \
               (dd.get("VOL_RATIO", pd.Series(dtype=float)) > 0.5)
        cands = dd[mask].copy()
        if cands.empty:
            return []
        for cn, expr in [("oversold_degree", cands["$change"] - idx_pct), ("trend_strength", cands.get(f"ROC{self.trend_window}", pd.Series(0.0, index=cands.index))), ("volume_spike", cands.get("VOL_RATIO", pd.Series(1.0, index=cands.index)))]:
            cands[cn] = expr
            s = cands[cn].std()
            cands[f"{cn}_n"] = (cands[cn] - cands[cn].mean()) / (s + 1e-8) if s > 0 else 0.0
        cands["score"] = 0.5 * cands["oversold_degree_n"] + 0.3 * cands["trend_strength_n"] + 0.2 * cands["volume_spike_n"]
        cands = cands.sort_values("score", ascending=False)
        out = []
        for _, row in cands.head(self.top_k).iterrows():
            sym = row.name[0] if isinstance(row.name, tuple) else row.name
            out.append({"stock": sym, "score": float(row["score"])})
        return out

    def _find_entry(self, stock, crash_date):
        try:
            ci = self._all_dates.index(crash_date)
        except ValueError:
            return None
        for w in range(1, self.max_wait_days + 1):
            if ci + w >= len(self._all_dates):
                break
            bd = self._all_dates[ci + w]
            try:
                dd = self._factor_df.loc[(stock, bd), :]
            except KeyError:
                continue
            if dd["$change"] > -0.01 or (dd.get("KMID", 0) > 0 and dd.get("LOWER_SHADOW", 0) > 0.01) or w == self.max_wait_days:
                return bd
        return None

    def _get_trade_date(self):
        step = getattr(self, "trade_step", None)
        cal = getattr(self, "trade_calendar", [])
        if step is not None and cal and 0 <= step < len(cal):
            return pd.Timestamp(cal[step])
        return None

    def generate_target_weight_position(self, score=None, current=None, trade_exchange=None, *args, **kwargs):
        if self.check_account_stop_loss():
            return {}
        self._ensure_initialized()
        td = self._get_trade_date()
        if td is None:
            return {}

        # 卖出：到期或止盈止损
        for stock in list(self._positions.keys()):
            info = self._positions[stock]
            try:
                bi = self._all_dates.index(info["buy_date"])
                ti = self._all_dates.index(td)
                if ti >= bi + self.hold_days:
                    self._positions.pop(stock)
                    continue
            except ValueError:
                pass
            try:
                cp = self._factor_df.loc[(stock, td), "$close"]
                ret = cp / info["buy_price"] - 1
                if ret >= self.take_profit or ret <= self.stop_loss:
                    self._positions.pop(stock)
            except KeyError:
                pass

        # 大跌日规划买入
        if td in self._crash_dates and not any(p.get("crash_date") == td for p in self._positions.values()):
            cands = self._find_oversold(td, self._crash_info.get(td, {}).get("pct_change", 0))
            for c in cands:
                bd = self._find_entry(c["stock"], td)
                if bd is not None:
                    self._buy_plan.setdefault(bd, []).append({"stock": c["stock"], "crash_date": td})

        # 执行今日买入
        if td in self._buy_plan:
            plan = self._buy_plan.pop(td)
            n = len(plan)
            if n > 0:
                w = 1.0 / n
                for item in plan:
                    self._positions[item["stock"]] = {"buy_date": td, "buy_price": 0, "crash_date": item["crash_date"]}
                return {item["stock"]: 1.0 / n for item in plan}

        return {}

    def reset(self, *args, **kwargs):
        self._positions.clear()
        self._buy_plan.clear()
        self._qm_trade_step_counter = 0
        self.reset_dynamic_risk()
        try:
            return super().reset(*args, **kwargs)
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            filtered = dict(kwargs)
            filtered.pop("level_infra", None)
            filtered.pop("common_infra", None)
            filtered.pop("trade_exchange", None)
            try:
                return super().reset(*args, **filtered)
            except TypeError:
                return super().reset()

    def post_exe_step(self, execute_result=None):
        self.log_progress()
        self.log_executed_trades(execute_result)


class RedisVolatilityWeightedStrategy(RedisWeightStrategy):
    """
    波动率加权 TopK 策略 (Volatility-Weighted Top-K)

    继承 RedisWeightStrategy 以复用：
    1. 调仓周期控制 (rebalance_days)
    2. 涨停 / 停牌过滤
    3. Redis 进度与交易日志记录
    """

    def __init__(self, *args, **kwargs):
        # vol_lookback 是本类特有参数，必须 pop 掉，否则 BaseStrategy 不认识
        self.vol_lookback = int(kwargs.pop("vol_lookback", 20))

        # 获取 topk 等参数供本类逻辑使用，但不 pop 掉（除非基类不需要）
        self.topk = int(kwargs.get("topk", 50))
        self.min_score = float(kwargs.get("min_score", 0.0))
        self.max_weight = float(kwargs.get("max_weight", 0.10))

        # 显式读取并记录 rebalance_days，用于调试
        r_days = kwargs.get("rebalance_days", 1)
        StructuredTaskLogger(
            logger,
            "redis-volatility-weighted",
            {"backtest_id": getattr(self, "backtest_id", None)},
        ).info("init", "Initializing VolatilityWeighted", rebalance_days=r_days, vol_lookback=self.vol_lookback)

        # 调用 RedisWeightStrategy 的初始化，它会处理 rebalance_days, redis 等
        super().__init__(*args, **kwargs)

    def generate_trade_decision(self, execute_result=None):
        """
        显式重写以确保调仓周期拦截逻辑执行。
        """
        # 1. 调仓周期拦截
        if not self._should_rebalance(self.rebalance_days):
            return TradeDecisionWO([], self)

        # 2. 正常生成订单
        current_step = self._get_trade_step_safe() or 0
        StructuredTaskLogger(
            logger,
            "redis-volatility-weighted",
            {"backtest_id": getattr(self, "backtest_id", None)},
        ).info("rebalance", "Rebalancing", step=current_step, rebalance_days=self.rebalance_days)
        return self._safe_generate_trade_decision(execute_result)

    def _estimate_volatility(self, stocks: list[str], ref_date) -> pd.Series:
        """计算各标的近期日度已实现波动率；数据不足时回退中位数填充。"""
        try:
            lookback_buf = int(self.vol_lookback * 2.0)
            start_dt = ref_date - pd.Timedelta(days=lookback_buf)
            # 使用模块顶层已解析的 D
            price_df = D.features(stocks, ["$close"], start_dt, ref_date, freq="day")
            if price_df is None or price_df.empty:
                raise ValueError("empty price data")
            prices = price_df["$close"].unstack(level=1)
            returns = prices.pct_change().dropna(how="all")
            vol = returns.tail(self.vol_lookback).std(ddof=1)
            median_vol = float(vol.median()) if not vol.empty else 0.02
            return vol.reindex(stocks).fillna(median_vol).clip(lower=1e-4)
        except Exception as exc:
            StructuredTaskLogger(
                logger,
                "redis-volatility-weighted",
                {"backtest_id": getattr(self, "backtest_id", None)},
            ).warning("volatility_estimation_failed", "Volatility estimation failed, falling back to equal weights", error=exc)
            return pd.Series(1.0, index=stocks)

    def generate_target_weight_position(self, score, current=None, trade_exchange=None, **kwargs):
        """
        覆写目标权重计算逻辑。
        注意：RedisWeightStrategy.generate_target_weight_position 已经在上层做了涨停/停牌过滤。
        """
        if self.check_account_stop_loss():
            return {}

        # 1. 基础过滤（由基类或此处处理）
        if score is None or score.empty:
            return {}

        # 统一 Series 格式
        if isinstance(score, pd.DataFrame):
            score = score.iloc[:, 0]

        scores = score.dropna()
        if self.min_score is not None:
            scores = scores[scores > self.min_score]

        if scores.empty:
            return {}

        # 2. TopK 选股
        if self.topk > 0 and len(scores) > self.topk:
            scores = scores.nlargest(self.topk)

        stocks = list(scores.index)
        trade_start_time = kwargs.get("trade_start_time") or kwargs.get("t_start")

        # 3. 历史波动率估计
        vol = self._estimate_volatility(stocks, trade_start_time)

        # 4. 倒数权重 + 归一化
        inv_vol = 1.0 / vol
        weights = inv_vol / inv_vol.sum()

        # 5. 单票上限约束后再归一化
        if self.max_weight > 0:
            weights = weights.clip(upper=self.max_weight)
            total = weights.sum()
            if total > 0:
                weights = weights / total

        StructuredTaskLogger(
            logger,
            "redis-volatility-weighted",
            {"backtest_id": getattr(self, "backtest_id", None)},
        ).info(
            "weights_built",
            "波动率权重构建完成",
            trade_date=trade_start_time.date() if hasattr(trade_start_time, "date") else trade_start_time,
            stock_count=len(weights),
            min_weight=f"{weights.min() * 100:.2f}%",
            max_weight=f"{weights.max() * 100:.2f}%",
        )
        return weights.to_dict()


class RedisFullAlphaStrategy(RedisWeightStrategy):
    """
    全量截面 Alpha 预测策略。

    目标行为：
    1. 每次调仓都按预测分从高到低构建目标持仓（不限制卖出数量）；
    2. 跌出 TopK 的标的全部卖出，补齐到 TopK；
    3. 若候选标的涨停/停牌导致不可买入，则自动顺延到下一只可交易标的。

    说明：
    - 继承 RedisWeightStrategy 以复用调仓周期控制、Redis 记录和风控；
    - 覆写 generate_target_weight_position 以显式实现“顺延补位”与“全量重构”。
    """

    def __init__(self, *args, **kwargs):
        self.topk = int(kwargs.get("topk", 50))
        self.max_weight = float(kwargs.get("max_weight", 0.05))
        super().__init__(*args, **kwargs)

    def generate_target_weight_position(self, score, current=None, trade_exchange=None, *args, **kwargs):
        if self.check_account_stop_loss():
            StructuredTaskLogger(
                logger,
                "redis-full-alpha",
                {"backtest_id": getattr(self, "backtest_id", None)},
            ).info("account_stop_loss", "Account stop-loss triggered. Target position is empty.")
            return {}

        if current is None and args:
            current = args[0]
        if trade_exchange is None and len(args) > 1:
            trade_exchange = args[1]
        if score is None or score.empty:
            return {}

        if isinstance(score, pd.DataFrame):
            if score.shape[1] == 0:
                return {}
            score = score.iloc[:, 0]

        ranked_scores = score.dropna().sort_values(ascending=False)
        if ranked_scores.empty:
            return {}

        exchange = trade_exchange or getattr(self, "trade_exchange", None)
        t_start = kwargs.get("trade_start_time") or kwargs.get("t_start")
        t_end = kwargs.get("trade_end_time") or kwargs.get("t_end") or t_start

        # 买入侧可交易性过滤：若某只不可买入，自动顺延到下一只候选。
        selected_symbols: list[str] = []
        skipped_untradable = 0
        if exchange is not None and t_start is not None:
            for sid in ranked_scores.index:
                try:
                    if exchange.check_stock_suspended(sid, t_start, t_end):
                        skipped_untradable += 1
                        continue
                    if exchange.check_stock_limit(sid, t_start, t_end, direction=Order.BUY):
                        skipped_untradable += 1
                        continue
                except Exception:
                    # 交易所检查异常时，降级为放行，避免阻断整个调仓流程
                    pass
                selected_symbols.append(sid)
                if self.topk > 0 and len(selected_symbols) >= self.topk:
                    break
            else:
                if exchange is None:
                    StructuredTaskLogger(
                        logger,
                        "redis-full-alpha",
                        {"backtest_id": getattr(self, "backtest_id", None)},
                    ).warning("trade_exchange_missing", "trade_exchange 未注入，跳过涨停/停牌过滤")
            selected_symbols = list(ranked_scores.index[: self.topk]) if self.topk > 0 else list(ranked_scores.index)

        if not selected_symbols:
            return {}

        selected_scores = ranked_scores.loc[selected_symbols].astype(float)
        if skipped_untradable:
            StructuredTaskLogger(
                logger,
                "redis-full-alpha",
                {"backtest_id": getattr(self, "backtest_id", None)},
            ).info(
                "fallback_fill",
                "候选顺延补位",
                trade_date=t_start.date() if hasattr(t_start, "date") else t_start,
                skipped=skipped_untradable,
            )

        # 将分数映射为严格正值再做权重归一化，避免全负分导致 sum<=0。
        min_val = float(selected_scores.min())
        if min_val <= 0:
            selected_scores = selected_scores - min_val + 1e-9

        total = float(selected_scores.sum())
        if total <= 0:
            return {}

        if self.max_weight is not None and 0 < self.max_weight < 1.0:
            weights = self._build_capped_weights(selected_scores, self.max_weight)
        else:
            weights = selected_scores / total
        return weights.to_dict()
