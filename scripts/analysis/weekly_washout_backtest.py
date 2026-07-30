#!/usr/bin/env python3
"""One-year weekly walk-forward backtest for leading-industry washout picks.

The production signal is evaluated after each calendar week's final trading
close.  Orders execute at the next week's first trading open and exit at the
following week's first trading open.  Parameter selection is walk-forward:
only the preceding training weeks are scored, then the chosen specification is
locked for the next application block.

"Control" remains a public OHLCV proxy.  It is not evidence of real positions
held by institutions or any other class of market participant.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.analysis.leading_control_backtest import (
    DEFAULT_CACHE_DIR,
    DEFAULT_SNAPSHOT_DIR,
    FORBIDDEN_REGIMES,
    _default_signal_builder,
    _finite,
    _paired_returns,
    _records,
    enrich_report_stock_names,
    load_historical_sw_industry_universe,
    load_hs300_benchmark,
    load_market_data,
    normalize_symbol,
    point_in_time_memberships,
    validate_market_data,
)


LOGGER = logging.getLogger("weekly_washout_backtest")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "weekly-washout-backtest"


@dataclass(frozen=True)
class WashoutConfig:
    """A stricter overlay on production washout candidates.

    The baseline exactly matches the production washout bounds.  Alternative
    variants only tighten them, because the production selector intentionally
    does not expose candidates that fail its safety filter.
    """

    name: str
    amount_ratio_max: float
    drawdown_min: float
    ret5_min: float
    up_down_amount_ratio_min: float
    trend_efficiency_min: float


CONFIGS = (
    WashoutConfig("防守", 0.85, -0.08, -0.04, 1.10, 0.16),
    WashoutConfig("均衡", 0.95, -0.10, -0.05, 1.02, 0.14),
    WashoutConfig("生产基准", 1.05, -0.12, -0.06, 0.95, 0.12),
)
CONFIG_BY_NAME = {config.name: config for config in CONFIGS}
BASELINE_CONFIG = CONFIG_BY_NAME["生产基准"]


def build_weekly_periods(
    stock: pd.DataFrame,
    start_date: Any,
    end_date: Any,
    *,
    weeks: int | None = None,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, str]]:
    """Build signal/entry/exit boundaries from consecutive exchange weeks."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    dates = pd.Series(
        pd.to_datetime(stock["trade_date"], errors="coerce").dropna().unique()
    )
    dates = dates.sort_values()
    groups = {
        period: [pd.Timestamp(value).normalize() for value in group.tolist()]
        for period, group in dates.groupby(dates.dt.to_period("W-FRI"))
    }
    keys = sorted(groups)
    periods: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, str]] = []
    for index in range(len(keys) - 2):
        signal_week, holding_week, exit_week = keys[index : index + 3]
        if (
            holding_week.ordinal != signal_week.ordinal + 1
            or exit_week.ordinal != holding_week.ordinal + 1
        ):
            continue
        signal_date = max(groups[signal_week])
        buy_date = min(groups[holding_week])
        sell_date = min(groups[exit_week])
        if signal_date < start or sell_date > end:
            continue
        periods.append(
            (
                signal_date,
                buy_date,
                sell_date,
                buy_date.date().isoformat(),
            )
        )
    if weeks is not None and weeks > 0:
        periods = periods[-int(weeks) :]
    return periods


def filter_washout_picks(
    picks: list[dict[str, Any]], config: WashoutConfig, max_stocks: int
) -> list[dict[str, Any]]:
    """Keep only production washout candidates satisfying ``config``."""
    eligible: list[dict[str, Any]] = []
    for position, raw in enumerate(picks):
        if str(raw.get("stage") or "") != "洗盘":
            continue
        try:
            amount_ratio = float(raw["amount_ratio"])
            drawdown = float(raw["drawdown20"])
            ret5 = float(raw["ret5"])
            amount_control = float(raw["up_down_amount_ratio20"])
            trend_efficiency = float(raw["trend_efficiency20"])
        except (KeyError, TypeError, ValueError):
            continue
        if not all(
            math.isfinite(value)
            for value in (
                amount_ratio,
                drawdown,
                ret5,
                amount_control,
                trend_efficiency,
            )
        ):
            continue
        if not (
            0.55 <= amount_ratio <= config.amount_ratio_max
            and config.drawdown_min <= drawdown <= -0.02
            and config.ret5_min <= ret5 <= 0.03
            and amount_control >= config.up_down_amount_ratio_min
            and trend_efficiency >= config.trend_efficiency_min
        ):
            continue
        symbol = normalize_symbol(raw.get("symbol"))
        if not symbol:
            continue
        eligible.append(
            {
                **raw,
                "symbol": symbol,
                "_source_position": position,
                "selection_score": float(raw.get("selection_score") or 0.0),
            }
        )

    limit = max(0, min(int(max_stocks), 10))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    industry_counts: dict[str, int] = {}
    for item in sorted(
        eligible,
        key=lambda value: (-value["selection_score"], value["_source_position"]),
    ):
        symbol = str(item["symbol"])
        industry = str(item.get("industry_code") or "未知")
        if symbol in seen or industry_counts.get(industry, 0) >= 2:
            continue
        selected.append(
            {key: value for key, value in item.items() if key != "_source_position"}
        )
        seen.add(symbol)
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def _benchmark_open_return(
    benchmark: pd.DataFrame, buy_date: pd.Timestamp, sell_date: pd.Timestamp
) -> float:
    if benchmark.empty:
        raise RuntimeError("formal weekly backtest requires an HS300 benchmark")
    if buy_date not in benchmark.index or sell_date not in benchmark.index:
        raise RuntimeError(
            f"沪深300基准缺少周度边界 {buy_date.date()} 或 {sell_date.date()}"
        )
    buy = float(benchmark.loc[buy_date, "open"])
    sell = float(benchmark.loc[sell_date, "open"])
    if not math.isfinite(buy) or not math.isfinite(sell) or buy <= 0 or sell <= 0:
        raise RuntimeError(
            f"沪深300周度边界价格无效: {buy_date.date()} / {sell_date.date()}"
        )
    return sell / buy - 1


def prepare_weekly_observations(
    stock: pd.DataFrame,
    industries: list[dict[str, Any]],
    memberships: dict[str, list[dict[str, Any]]],
    *,
    start_date: Any,
    end_date: Any,
    weeks: int = 52,
    lookback_calendar_days: int = 180,
    signal_builder=None,
    benchmark: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Evaluate production signals without exposing any post-signal rows."""
    frame = stock.copy()
    frame["trade_date"] = pd.to_datetime(
        frame["trade_date"], errors="coerce"
    ).dt.normalize()
    frame["symbol"] = frame["symbol"].map(normalize_symbol)
    frame = frame.dropna(subset=["trade_date", "symbol"])
    frame = frame.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    periods = build_weekly_periods(frame, start_date, end_date, weeks=weeks)
    builder = signal_builder or _default_signal_builder
    benchmark_frame = benchmark.copy()
    benchmark_frame["trade_date"] = pd.to_datetime(
        benchmark_frame["trade_date"], errors="coerce"
    ).dt.normalize()
    benchmark_frame["open"] = pd.to_numeric(benchmark_frame["open"], errors="coerce")
    benchmark_frame = (
        benchmark_frame.dropna(subset=["trade_date", "open"])
        .drop_duplicates("trade_date", keep="last")
        .set_index("trade_date")
    )

    observations: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for period_index, (signal_date, buy_date, sell_date, holding_week) in enumerate(
        periods, start=1
    ):
        LOGGER.info(
            "Evaluating weekly signal %d/%d at %s",
            period_index,
            len(periods),
            signal_date.date(),
        )
        history = frame[
            (
                frame["trade_date"]
                >= signal_date - pd.Timedelta(days=lookback_calendar_days)
            )
            & (frame["trade_date"] <= signal_date)
        ].copy()
        pit_members = point_in_time_memberships(memberships, signal_date)
        try:
            signal = builder(history, industries, pit_members, 10)
            market = dict(signal.get("market") or {})
            regime = str(market.get("regime") or "未知")
            raw_picks = list(signal.get("picks") or [])
        except Exception as exc:  # formal output rejects any captured error
            LOGGER.exception("Weekly signal failed for %s", signal_date.date())
            regime = "数据异常"
            market = {"regime": regime}
            raw_picks = []
            errors.append(
                {"signal_date": signal_date.date().isoformat(), "error": str(exc)}
            )
        paired = _paired_returns(frame, buy_date, sell_date)
        prices: dict[str, dict[str, Any]] = {}
        for symbol, row in paired.iterrows():
            prices[str(symbol)] = {
                "buy_open": _finite(row.get("buy_open")),
                "sell_open": _finite(row.get("sell_open")),
            }
        observations.append(
            {
                "holding_week": holding_week,
                "signal_date": signal_date.date().isoformat(),
                "buy_date": buy_date.date().isoformat(),
                "sell_date": sell_date.date().isoformat(),
                "regime": regime,
                "market": market,
                "raw_picks": raw_picks,
                "prices": prices,
                "benchmark_return": _benchmark_open_return(
                    benchmark_frame, buy_date, sell_date
                ),
            }
        )
    return observations, errors


def simulate_weekly_strategy(
    observations: list[dict[str, Any]],
    *,
    configs: WashoutConfig | list[WashoutConfig] = BASELINE_CONFIG,
    transaction_cost_bps: float = 15.0,
    max_stocks: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply fixed or per-week specifications to prepared observations."""
    weekly_rows: list[dict[str, Any]] = []
    holding_rows: list[dict[str, Any]] = []
    previous_weights: dict[str, float] = {}
    cost = max(float(transaction_cost_bps), 0.0) / 10_000
    if isinstance(configs, WashoutConfig):
        chosen_configs = [configs] * len(observations)
    else:
        chosen_configs = list(configs)
        if len(chosen_configs) != len(observations):
            raise ValueError("one configuration is required for every observation")

    for observation, config in zip(observations, chosen_configs, strict=True):
        regime = str(observation.get("regime") or "未知")
        picks = (
            []
            if regime in FORBIDDEN_REGIMES
            else filter_washout_picks(
                list(observation.get("raw_picks") or []), config, max_stocks
            )
        )
        target_weight = 1 / len(picks) if picks else 0.0
        current_weights: dict[str, float] = {}
        gross_return = 0.0
        net_return = 0.0
        holding_count = 0
        for rank, pick in enumerate(picks, start=1):
            symbol = str(pick["symbol"])
            price = dict((observation.get("prices") or {}).get(symbol) or {})
            buy_open = _finite(price.get("buy_open"))
            sell_open = _finite(price.get("sell_open"))
            if buy_open is None or float(buy_open) <= 0:
                continue
            holding_count += 1
            current_weights[symbol] = target_weight
            if sell_open is None or float(sell_open) <= 0:
                gross = -1.0
                net = -1.0
                exit_status = "退出日无报价，保守按全部损失"
            else:
                gross = float(sell_open) / float(buy_open) - 1
                net = (1 + gross) * (1 - cost) ** 2 - 1
                exit_status = "正常退出"
            gross_return += target_weight * gross
            net_return += target_weight * net
            holding_rows.append(
                {
                    "holding_week": observation["holding_week"],
                    "signal_date": observation["signal_date"],
                    "buy_date": observation["buy_date"],
                    "sell_date": observation["sell_date"],
                    "config": config.name,
                    "symbol": symbol,
                    "stock_name": str(pick.get("stock_name") or symbol),
                    "industry_code": str(pick.get("industry_code") or "未知"),
                    "industry_name": str(pick.get("industry_name") or "未知"),
                    "quadrant": str(pick.get("quadrant") or "领先区"),
                    "stage": "洗盘",
                    "selection_rank": rank,
                    "buy_open": float(buy_open),
                    "sell_open": float(sell_open) if sell_open is not None else None,
                    "exit_status": exit_status,
                    "gross_return": gross,
                    "net_return": net,
                    "weight": target_weight,
                    "contribution": target_weight * net,
                    "reason": str(pick.get("reason") or ""),
                    "invalidation": str(pick.get("invalidation") or ""),
                }
            )
        symbols = set(previous_weights) | set(current_weights)
        previous_cash = 1 - sum(previous_weights.values())
        current_cash = 1 - sum(current_weights.values())
        turnover = 0.5 * (
            sum(
                abs(current_weights.get(symbol, 0) - previous_weights.get(symbol, 0))
                for symbol in symbols
            )
            + abs(current_cash - previous_cash)
        )
        previous_weights = current_weights
        weekly_rows.append(
            {
                "holding_week": observation["holding_week"],
                "signal_date": observation["signal_date"],
                "buy_date": observation["buy_date"],
                "sell_date": observation["sell_date"],
                "config": config.name,
                "regime": regime,
                "signal_count": len(picks),
                "holding_count": holding_count,
                "gross_return": gross_return,
                "net_return": net_return,
                "benchmark_return": float(observation["benchmark_return"]),
                "excess_return": net_return - float(observation["benchmark_return"]),
                "turnover": float(turnover),
                "cash_reason": regime if regime in FORBIDDEN_REGIMES else None,
            }
        )
    return pd.DataFrame(weekly_rows), pd.DataFrame(holding_rows)


def calculate_weekly_metrics(
    weekly: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Calculate weekly risk, return and benchmark statistics."""
    if weekly.empty:
        return {"weeks": 0}, pd.DataFrame()
    returns = pd.to_numeric(weekly["net_return"], errors="coerce").fillna(0.0)
    benchmark = pd.to_numeric(weekly["benchmark_return"], errors="coerce").fillna(0.0)
    equity = (1 + returns).cumprod()
    benchmark_equity = (1 + benchmark).cumprod()
    drawdown = equity / equity.cummax().clip(lower=1.0) - 1
    weeks = len(returns)
    total = float(equity.iloc[-1] - 1)
    annualized = (
        float(equity.iloc[-1] ** (52 / weeks) - 1) if equity.iloc[-1] > 0 else -1.0
    )
    std = float(returns.std(ddof=0))
    volatility = std * math.sqrt(52)
    downside = returns[returns < 0]
    downside_deviation = (
        float(np.sqrt(np.mean(np.square(downside))) * math.sqrt(52))
        if not downside.empty
        else 0.0
    )
    benchmark_variance = float(benchmark.var(ddof=0))
    beta = (
        float(np.cov(returns, benchmark, ddof=0)[0, 1] / benchmark_variance)
        if benchmark_variance > 0
        else None
    )
    excess = returns - benchmark
    maximum_drawdown = float(drawdown.min())
    holdings = pd.to_numeric(weekly["holding_count"], errors="coerce").fillna(0)
    invested = holdings > 0
    invested_returns = returns[invested]
    tracking_error = float(excess.std(ddof=0) * math.sqrt(52))
    metrics = {
        "weeks": weeks,
        "invested_weeks": int(invested.sum()),
        "total_return": total,
        "annualized_return": annualized,
        "annualized_volatility": volatility,
        "sharpe_ratio": float(returns.mean() / std * math.sqrt(52))
        if std > 0
        else None,
        "sortino_ratio": (
            float(returns.mean() * 52 / downside_deviation)
            if downside_deviation > 0
            else None
        ),
        "max_drawdown": maximum_drawdown,
        "calmar_ratio": annualized / abs(maximum_drawdown)
        if maximum_drawdown < 0
        else None,
        "win_rate": float((invested_returns > 0).mean())
        if not invested_returns.empty
        else None,
        "best_week": float(returns.max()),
        "worst_week": float(returns.min()),
        "cash_weeks": int((~invested).sum()),
        "cash_week_ratio": float((~invested).mean()),
        "average_weekly_turnover": float(
            pd.to_numeric(weekly["turnover"], errors="coerce").fillna(0).mean()
        ),
        "benchmark_total_return": float(benchmark_equity.iloc[-1] - 1),
        "excess_total_return": float(equity.iloc[-1] / benchmark_equity.iloc[-1] - 1),
        "tracking_error": tracking_error,
        "information_ratio": (
            float(excess.mean() / excess.std(ddof=0) * math.sqrt(52))
            if excess.std(ddof=0) > 0
            else None
        ),
        "alpha_annualized": (
            float((returns.mean() - beta * benchmark.mean()) * 52)
            if beta is not None
            else None
        ),
        "beta": beta,
    }
    curve = pd.DataFrame(
        {
            "holding_week": weekly["holding_week"].to_numpy(),
            "strategy_equity": equity,
            "benchmark_equity": benchmark_equity,
            "drawdown": drawdown,
        }
    )
    return metrics, curve


def walk_forward_optimize(
    observations: list[dict[str, Any]],
    *,
    candidate_configs: tuple[WashoutConfig, ...] = CONFIGS,
    train_weeks: int = 26,
    apply_weeks: int = 13,
    transaction_cost_bps: float = 15.0,
    max_stocks: int = 10,
) -> dict[str, Any]:
    """Choose parameters on trailing weeks and lock them for future blocks."""
    if train_weeks <= 0 or apply_weeks <= 0:
        raise ValueError("train_weeks and apply_weeks must be positive")
    if len(observations) <= train_weeks:
        raise ValueError("insufficient observations for walk-forward optimization")
    selections: list[dict[str, Any]] = []
    oos_observations: list[dict[str, Any]] = []
    oos_configs: list[WashoutConfig] = []
    for apply_start in range(train_weeks, len(observations), apply_weeks):
        train_start = apply_start - train_weeks
        training = observations[train_start:apply_start]
        scores: list[dict[str, Any]] = []
        for config in candidate_configs:
            weekly, _holdings = simulate_weekly_strategy(
                training,
                configs=config,
                transaction_cost_bps=transaction_cost_bps,
                max_stocks=max_stocks,
            )
            metrics, _curve = calculate_weekly_metrics(weekly)
            sharpe = metrics.get("sharpe_ratio")
            scores.append(
                {
                    "config": config.name,
                    "sharpe_ratio": _finite(sharpe),
                    "total_return": _finite(metrics.get("total_return")),
                    "invested_weeks": metrics.get("invested_weeks", 0),
                    "_score": (
                        float(sharpe) if sharpe is not None else -1_000_000.0,
                        float(metrics.get("total_return") or 0.0),
                        -list(candidate_configs).index(config),
                    ),
                }
            )
        chosen_row = max(scores, key=lambda item: item["_score"])
        chosen = CONFIG_BY_NAME[chosen_row["config"]]
        apply_end = min(apply_start + apply_weeks, len(observations))
        applied = observations[apply_start:apply_end]
        oos_observations.extend(applied)
        oos_configs.extend([chosen] * len(applied))
        selections.append(
            {
                "training_start": training[0]["holding_week"],
                "training_end": training[-1]["holding_week"],
                "application_start": applied[0]["holding_week"],
                "application_end": applied[-1]["holding_week"],
                "chosen_config": chosen.name,
                "candidate_scores": [
                    {key: value for key, value in row.items() if key != "_score"}
                    for row in scores
                ],
            }
        )
    optimized, optimized_holdings = simulate_weekly_strategy(
        oos_observations,
        configs=oos_configs,
        transaction_cost_bps=transaction_cost_bps,
        max_stocks=max_stocks,
    )
    baseline, _baseline_holdings = simulate_weekly_strategy(
        oos_observations,
        configs=BASELINE_CONFIG,
        transaction_cost_bps=transaction_cost_bps,
        max_stocks=max_stocks,
    )
    optimized_metrics, optimized_curve = calculate_weekly_metrics(optimized)
    baseline_metrics, baseline_curve = calculate_weekly_metrics(baseline)
    return {
        "train_weeks": train_weeks,
        "apply_weeks": apply_weeks,
        "objective": "训练窗口成本后周收益 Sharpe；并列时取累计收益，再取更严格参数",
        "selections": selections,
        "optimized_metrics": {
            key: _finite(value) for key, value in optimized_metrics.items()
        },
        "baseline_metrics_same_oos": {
            key: _finite(value) for key, value in baseline_metrics.items()
        },
        "optimized_weekly": _records(optimized),
        "optimized_holdings": _records(optimized_holdings),
        "optimized_equity_curve": _records(optimized_curve),
        "baseline_equity_curve_same_oos": _records(baseline_curve),
    }


def _sensitivity(
    observations: list[dict[str, Any]], transaction_cost_bps: float
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scenarios = [(config, 10) for config in CONFIGS]
    scenarios.extend((BASELINE_CONFIG, limit) for limit in (3, 5))
    for config, limit in scenarios:
        weekly, _holdings = simulate_weekly_strategy(
            observations,
            configs=config,
            transaction_cost_bps=transaction_cost_bps,
            max_stocks=limit,
        )
        metrics, _curve = calculate_weekly_metrics(weekly)
        rows.append(
            {
                "scenario": f"{config.name} / 最多{limit}只",
                "config": config.name,
                "max_stocks": limit,
                **{key: _finite(value) for key, value in metrics.items()},
            }
        )
    gross, _holdings = simulate_weekly_strategy(
        observations, configs=BASELINE_CONFIG, transaction_cost_bps=0, max_stocks=10
    )
    gross_metrics, _curve = calculate_weekly_metrics(gross)
    rows.append(
        {
            "scenario": "生产基准 / 最多10只 / 不计成本",
            "config": BASELINE_CONFIG.name,
            "max_stocks": 10,
            **{key: _finite(value) for key, value in gross_metrics.items()},
        }
    )
    return rows


def build_report(
    observations: list[dict[str, Any]],
    *,
    data_quality: dict[str, Any],
    signal_errors: list[dict[str, str]],
    transaction_cost_bps: float = 15.0,
    max_stocks: int = 10,
    train_weeks: int = 26,
    apply_weeks: int = 13,
) -> dict[str, Any]:
    baseline, holdings = simulate_weekly_strategy(
        observations,
        configs=BASELINE_CONFIG,
        transaction_cost_bps=transaction_cost_bps,
        max_stocks=max_stocks,
    )
    metrics, curve = calculate_weekly_metrics(baseline)
    walk_forward = walk_forward_optimize(
        observations,
        train_weeks=train_weeks,
        apply_weeks=apply_weeks,
        transaction_cost_bps=transaction_cost_bps,
        max_stocks=max_stocks,
    )
    attribution: list[dict[str, Any]] = []
    if not holdings.empty:
        for industry, group in holdings.groupby("industry_name", dropna=False):
            attribution.append(
                {
                    "industry_name": str(industry or "未知"),
                    "holding_records": int(len(group)),
                    "weeks": int(group["holding_week"].nunique()),
                    "win_rate": float((group["net_return"] > 0).mean()),
                    "contribution": float(group["contribution"].sum()),
                }
            )
        attribution.sort(key=lambda row: row["contribution"], reverse=True)
    return {
        "meta": {
            "title": "领先区洗盘策略最近一年周频滚动回测报告",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "strategy": "申万领先区 + 控盘量价代理 + 仅洗盘 + 周频滚动",
        },
        "parameters": {
            "start_signal_date": observations[0]["signal_date"]
            if observations
            else None,
            "end_date": observations[-1]["sell_date"] if observations else None,
            "frequency": "weekly",
            "signal_timing": "每周最后一个交易日收盘后",
            "entry_timing": "下一周首个交易日开盘",
            "exit_timing": "再下一周首个交易日开盘",
            "transaction_cost_bps_per_side": float(transaction_cost_bps),
            "max_stocks": min(max(int(max_stocks), 0), 10),
            "max_stocks_per_industry": 2,
            "cash_regimes": sorted(FORBIDDEN_REGIMES),
            "benchmark": "沪深300",
            "baseline_config": asdict(BASELINE_CONFIG),
            "candidate_configs": [asdict(config) for config in CONFIGS],
        },
        "data_quality": data_quality,
        "metrics": {key: _finite(value) for key, value in metrics.items()},
        "equity_curve": _records(curve),
        "weekly": _records(baseline),
        "holdings": _records(holdings),
        "walk_forward": walk_forward,
        "sensitivity": _sensitivity(observations, transaction_cost_bps),
        "industry_attribution": attribution,
        "signal_errors": signal_errors,
        "limitations": [
            "“主力控盘”是公开日线量价代理，不代表真实机构持仓或资金身份。",
            "每周末收盘生成信号，下一周首开成交、再下一周首开退出；选股函数从未接收信号日之后的数据。",
            "调优结果仅报告 walk-forward 样本外区间；全年参数敏感性属于描述性分析，不作为无偏绩效。",
            "候选参数只会收紧生产洗盘阈值，不会纳入被生产安全门槛排除的股票。",
            "历史申万成分按 in_date/out_date 点时过滤，避免使用当前成分回填历史。",
            "特征快照缺少完整历史 ST 名称，当前 Tushare Token 又无 stock_st 权限，可能高估少量历史样本可交易性。",
            "退出日无报价按 -100% 保守处理；尚未逐笔模拟涨跌停排队、容量和随成交额变化的冲击成本。",
            "回测结果仅供研究，不构成投资建议；一年样本很短，结果对市场风格高度敏感。",
        ],
    }


def _percent(value: Any) -> str:
    return "-" if value is None else f"{float(value):.2%}"


def _number(value: Any) -> str:
    return "-" if value is None else f"{float(value):.2f}"


def _svg_equity(report: dict[str, Any], key: str = "equity_curve") -> str:
    rows = report.get(key) or []
    if len(rows) < 2:
        return "<p>暂无足够数据。</p>"
    values = [float(row["strategy_equity"]) for row in rows] + [
        float(row["benchmark_equity"]) for row in rows
    ]
    low, high = min(values), max(values)
    span = high - low or 1.0

    def points(field: str) -> str:
        return " ".join(
            f"{30 + index * 720 / (len(rows) - 1):.1f},{220 - (float(row[field]) - low) * 180 / span:.1f}"
            for index, row in enumerate(rows)
        )

    return (
        '<svg viewBox="0 0 780 250" role="img" aria-label="权益曲线">'
        '<rect x="30" y="25" width="720" height="195" fill="#f7faf9"/>'
        f'<polyline fill="none" stroke="#176b5b" stroke-width="3" points="{points("strategy_equity")}"/>'
        f'<polyline fill="none" stroke="#8a96a3" stroke-width="2" points="{points("benchmark_equity")}"/>'
        '<text x="35" y="20" fill="#176b5b">策略</text><text x="85" y="20" fill="#697586">沪深300</text>'
        "</svg>"
    )


def render_html(report: dict[str, Any], target: Path) -> None:
    metrics = report["metrics"]
    oos = report["walk_forward"]["optimized_metrics"]
    oos_base = report["walk_forward"]["baseline_metrics_same_oos"]
    weekly_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['holding_week']))}</td><td>{escape(str(row['regime']))}</td>"
        f"<td>{row['holding_count']}</td><td>{_percent(row['net_return'])}</td>"
        f"<td>{_percent(row['benchmark_return'])}</td><td>{escape(str(row['config']))}</td></tr>"
        for row in report["weekly"]
    )
    selection_rows = "".join(
        "<tr>"
        f"<td>{row['training_start']} ～ {row['training_end']}</td>"
        f"<td>{row['application_start']} ～ {row['application_end']}</td>"
        f"<td>{escape(row['chosen_config'])}</td></tr>"
        for row in report["walk_forward"]["selections"]
    )
    holding_rows = "".join(
        "<tr>"
        f"<td>{row['holding_week']}</td><td>{row['symbol']} {escape(row['stock_name'])}</td>"
        f"<td>{escape(row['industry_name'])}</td><td>{_percent(row['net_return'])}</td>"
        f"<td class=reason>{escape(row['reason'])}</td></tr>"
        for row in report["holdings"]
    )
    sensitivity_rows = "".join(
        "<tr>"
        f"<td>{escape(row['scenario'])}</td><td>{_percent(row['total_return'])}</td>"
        f"<td>{_percent(row['annualized_return'])}</td><td>{_percent(row['max_drawdown'])}</td>"
        f"<td>{_number(row['sharpe_ratio'])}</td><td>{row['invested_weeks']}</td></tr>"
        for row in report["sensitivity"]
    )
    limitation_rows = "".join(
        f"<li>{escape(item)}</li>" for item in report["limitations"]
    )
    payload = escape(json.dumps(report, ensure_ascii=False, allow_nan=False))
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(report["meta"]["title"])}</title>
<style>:root{{--ink:#172235;--muted:#697586;--green:#176b5b;--line:#dbe2ea;--bg:#eef2f5}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.6 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}}main{{max-width:1180px;margin:24px auto;padding:0 18px}}header,.panel{{background:white;border:1px solid var(--line);border-radius:14px;padding:22px;margin-bottom:16px}}h1{{margin:0;font-size:28px}}h2{{margin:0 0 14px}}.sub{{color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:16px}}.card{{background:#f7faf9;padding:12px;border-radius:9px}}.card b{{display:block;color:var(--green);font-size:20px}}.compare{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;white-space:nowrap}}th{{background:#f3f6f8;position:sticky;top:0}}.scroll{{overflow:auto;max-height:560px}}.reason{{white-space:normal;min-width:420px}}svg{{width:100%;height:auto}}@media(max-width:700px){{.compare{{grid-template-columns:1fr}}h1{{font-size:22px}}}}</style></head>
<body><main><header><h1>{escape(report["meta"]["title"])}</h1><div class=sub>{escape(report["meta"]["strategy"])}</div><div class=sub>信号 {report["parameters"]["start_signal_date"]} 至退出 {report["parameters"]["end_date"]} · 单边成本 {report["parameters"]["transaction_cost_bps_per_side"]:.0f}bp</div><div class=cards><div class=card>全年累计<b>{_percent(metrics["total_return"])}</b></div><div class=card>年化收益<b>{_percent(metrics["annualized_return"])}</b></div><div class=card>最大回撤<b>{_percent(metrics["max_drawdown"])}</b></div><div class=card>Sharpe<b>{_number(metrics["sharpe_ratio"])}</b></div><div class=card>胜率<b>{_percent(metrics["win_rate"])}</b></div><div class=card>空仓周<b>{metrics["cash_weeks"]}/{metrics["weeks"]}</b></div></div></header>
<section class=panel><h2>全年生产基准权益</h2>{_svg_equity(report)}</section>
<section class=panel><h2>Walk-forward 样本外对照</h2><p class=sub>过去 {report["walk_forward"]["train_weeks"]} 周选参，随后 {report["walk_forward"]["apply_weeks"]} 周锁定；只比较样本外区间。</p><div class=compare><div class=card>滚动优化累计<b>{_percent(oos["total_return"])}</b>Sharpe {_number(oos["sharpe_ratio"])} · 回撤 {_percent(oos["max_drawdown"])}</div><div class=card>同区间生产基准<b>{_percent(oos_base["total_return"])}</b>Sharpe {_number(oos_base["sharpe_ratio"])} · 回撤 {_percent(oos_base["max_drawdown"])}</div></div><table><thead><tr><th>训练窗口</th><th>未来应用窗口</th><th>锁定参数</th></tr></thead><tbody>{selection_rows}</tbody></table></section>
<section class=panel><h2>参数与持仓数敏感性（描述性）</h2><div class=scroll><table><thead><tr><th>情景</th><th>累计</th><th>年化</th><th>回撤</th><th>Sharpe</th><th>投资周</th></tr></thead><tbody>{sensitivity_rows}</tbody></table></div></section>
<section class=panel><h2>逐周表现</h2><div class=scroll><table><thead><tr><th>持有周</th><th>状态</th><th>持仓</th><th>策略</th><th>基准</th><th>参数</th></tr></thead><tbody>{weekly_rows}</tbody></table></div></section>
<section class=panel><h2>逐周持仓与推荐理由</h2><div class=scroll><table><thead><tr><th>持有周</th><th>股票</th><th>申万行业</th><th>收益</th><th>当时理由</th></tr></thead><tbody>{holding_rows}</tbody></table></div></section>
<section class=panel><h2>限制与风险</h2><ul>{limitation_rows}</ul></section><section class=panel><details><summary>嵌入式完整 JSON</summary><pre id=report-data>{payload}</pre></details></section></main></body></html>"""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")


def render_pdf(report: dict[str, Any], target: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    font_name = "QuantMindCN"
    candidates = [
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/STHeiti Light.ttc"),
    ]
    font_path = next((path for path in candidates if path.exists()), None)
    if font_path is None:
        raise RuntimeError("No embeddable Chinese TTF/TTC font is available for PDF")
    try:
        pdfmetrics.getFont(font_name)
    except KeyError:
        pdfmetrics.registerFont(TTFont(font_name, str(font_path), subfontIndex=0))
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "weekly-body",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=8,
        leading=12,
    )
    heading = ParagraphStyle(
        "weekly-heading",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=14,
        leading=20,
        textColor=colors.HexColor("#173a5e"),
    )
    title = ParagraphStyle(
        "weekly-title",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=19,
        leading=27,
    )

    def p(value: Any, style=body):
        return Paragraph(escape(str(value if value is not None else "-")), style)

    def table(rows: list[list[Any]], widths: list[float]):
        result = Table(
            [[p(cell) for cell in row] for row in rows], colWidths=widths, repeatRows=1
        )
        result.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173a5e")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cad3dc")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f4f7fa")],
                    ),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        return result

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.pdf")
    document = SimpleDocTemplate(
        str(temporary),
        pagesize=A4,
        leftMargin=13 * mm,
        rightMargin=13 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=report["meta"]["title"],
        author="QuantMind",
    )
    metrics = report["metrics"]
    oos = report["walk_forward"]["optimized_metrics"]
    oos_base = report["walk_forward"]["baseline_metrics_same_oos"]
    story: list[Any] = [
        p(report["meta"]["title"], title),
        p(f"策略：{report['meta']['strategy']}"),
        p(
            f"区间：信号 {report['parameters']['start_signal_date']} 至退出 {report['parameters']['end_date']}；完整持有周 {metrics['weeks']}；基准 沪深300"
        ),
        p(
            f"数据质量：{report['data_quality'].get('status', '-')}；信号错误 {len(report['signal_errors'])}"
        ),
        Spacer(1, 4 * mm),
        p("核心指标", heading),
        table(
            [
                ["累计", "年化", "波动", "最大回撤", "Sharpe", "胜率"],
                [
                    _percent(metrics.get("total_return")),
                    _percent(metrics.get("annualized_return")),
                    _percent(metrics.get("annualized_volatility")),
                    _percent(metrics.get("max_drawdown")),
                    _number(metrics.get("sharpe_ratio")),
                    _percent(metrics.get("win_rate")),
                ],
            ],
            [28 * mm] * 6,
        ),
        Spacer(1, 4 * mm),
        p("Walk-forward 样本外对照", heading),
        p(
            f"过去 {report['walk_forward']['train_weeks']} 周选参，随后 {report['walk_forward']['apply_weeks']} 周锁定。滚动优化累计 {_percent(oos.get('total_return'))}、Sharpe {_number(oos.get('sharpe_ratio'))}、回撤 {_percent(oos.get('max_drawdown'))}；同区间生产基准累计 {_percent(oos_base.get('total_return'))}、Sharpe {_number(oos_base.get('sharpe_ratio'))}、回撤 {_percent(oos_base.get('max_drawdown'))}。"
        ),
        table(
            [["训练窗口", "未来应用窗口", "锁定参数"]]
            + [
                [
                    f"{row['training_start']}~{row['training_end']}",
                    f"{row['application_start']}~{row['application_end']}",
                    row["chosen_config"],
                ]
                for row in report["walk_forward"]["selections"]
            ],
            [60 * mm, 60 * mm, 35 * mm],
        ),
        Spacer(1, 4 * mm),
        p("敏感性分析", heading),
    ]
    story.append(
        table(
            [["情景", "累计", "年化", "回撤", "Sharpe", "投资周"]]
            + [
                [
                    row["scenario"],
                    _percent(row.get("total_return")),
                    _percent(row.get("annualized_return")),
                    _percent(row.get("max_drawdown")),
                    _number(row.get("sharpe_ratio")),
                    row.get("invested_weeks"),
                ]
                for row in report["sensitivity"]
            ],
            [55 * mm, 24 * mm, 24 * mm, 24 * mm, 22 * mm, 20 * mm],
        )
    )
    story.extend(
        [
            PageBreak(),
            p("逐周表现", heading),
            table(
                [["持有周", "信号日", "状态", "持仓", "策略", "基准"]]
                + [
                    [
                        row["holding_week"],
                        row["signal_date"],
                        row["regime"],
                        row["holding_count"],
                        _percent(row["net_return"]),
                        _percent(row["benchmark_return"]),
                    ]
                    for row in report["weekly"]
                ],
                [28 * mm, 29 * mm, 24 * mm, 18 * mm, 25 * mm, 25 * mm],
            ),
            PageBreak(),
            p("逐周持仓与理由", heading),
        ]
    )
    holding_table = [["持有周", "股票 / 行业", "收益", "当时理由"]] + [
        [
            row["holding_week"],
            f"{row['symbol']} {row['stock_name']} / {row['industry_name']}",
            _percent(row["net_return"]),
            row["reason"],
        ]
        for row in report["holdings"]
    ]
    story.extend(
        [
            table(holding_table, [25 * mm, 48 * mm, 22 * mm, 74 * mm]),
            Spacer(1, 4 * mm),
            p("限制与风险", heading),
        ]
    )
    story.extend(p(f"• {item}") for item in report["limitations"])

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(font_name, 8)
        canvas.setFillColor(colors.HexColor("#697586"))
        canvas.drawCentredString(A4[0] / 2, 7 * mm, f"第 {doc.page} 页")
        canvas.restoreState()

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    temporary.replace(target)


def write_outputs(report: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    token = str(report["parameters"].get("end_date") or "latest").replace("-", "")
    stem = f"weekly_washout_backtest_{token}"
    paths = {
        "json": output_dir / f"{stem}.json",
        "weekly_csv": output_dir / f"{stem}_weekly.csv",
        "holdings_csv": output_dir / f"{stem}_holdings.csv",
        "walk_forward_csv": output_dir / f"{stem}_walk_forward.csv",
        "selection_csv": output_dir / f"{stem}_selections.csv",
        "sensitivity_csv": output_dir / f"{stem}_sensitivity.csv",
        "html": output_dir / f"{stem}.html",
        "pdf": output_dir / f"{stem}.pdf",
    }
    paths["json"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame(report["weekly"]).to_csv(paths["weekly_csv"], index=False)
    pd.DataFrame(report["holdings"]).to_csv(paths["holdings_csv"], index=False)
    pd.DataFrame(report["walk_forward"]["optimized_weekly"]).to_csv(
        paths["walk_forward_csv"], index=False
    )
    pd.DataFrame(report["walk_forward"]["selections"]).to_csv(
        paths["selection_csv"], index=False
    )
    pd.DataFrame(report["sensitivity"]).to_csv(paths["sensitivity_csv"], index=False)
    render_html(report, paths["html"])
    render_pdf(report, paths["pdf"])
    for kind in ("json", "html", "pdf"):
        shutil.copyfile(paths[kind], output_dir / f"latest.{kind}")
    return paths


def parse_args() -> argparse.Namespace:
    today = pd.Timestamp.now().normalize()
    parser = argparse.ArgumentParser(
        description="回测领先区仅洗盘策略最近一年周频滚动表现"
    )
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--end-date", default=today.date().isoformat())
    parser.add_argument("--weeks", type=int, default=52)
    parser.add_argument("--train-weeks", type=int, default=26)
    parser.add_argument("--apply-weeks", type=int, default=13)
    parser.add_argument("--cost-bps", type=float, default=15.0)
    parser.add_argument("--max-stocks", type=int, default=10)
    parser.add_argument("--member-cache-days", type=float, default=30.0)
    parser.add_argument("--skip-db-tail", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.weeks <= args.train_weeks:
        raise ValueError("weeks must be greater than train-weeks")
    requested_end = pd.Timestamp(args.end_date).normalize()
    load_start = (requested_end - pd.Timedelta(days=560)).to_period("Y").start_time
    LOGGER.info("Loading feature snapshots and database tail")
    stock, source_audit = load_market_data(
        args.snapshot_dir,
        load_start,
        requested_end,
        supplement_database=not args.skip_db_tail,
    )
    stock, quality = validate_market_data(stock)
    quality["sources"] = source_audit
    actual_end = min(requested_end, pd.Timestamp(stock["trade_date"].max()).normalize())
    period_start = actual_end - pd.Timedelta(days=430)
    LOGGER.info("Loading point-in-time SW2021 universe and HS300 benchmark")
    industries, memberships = load_historical_sw_industry_universe(
        args.cache_dir, cache_days=args.member_cache_days
    )
    benchmark = load_hs300_benchmark(
        args.cache_dir, period_start, actual_end, cache_days=args.member_cache_days
    )
    observations, errors = prepare_weekly_observations(
        stock,
        industries,
        memberships,
        start_date=period_start,
        end_date=actual_end,
        weeks=args.weeks,
        benchmark=benchmark,
    )
    if len(observations) != args.weeks:
        raise RuntimeError(
            f"expected {args.weeks} complete weekly periods, got {len(observations)}"
        )
    if errors:
        raise RuntimeError(
            f"{len(errors)} weekly signals failed; refusing a formal report"
        )
    report = build_report(
        observations,
        data_quality=quality,
        signal_errors=errors,
        transaction_cost_bps=args.cost_bps,
        max_stocks=args.max_stocks,
        train_weeks=args.train_weeks,
        apply_weeks=args.apply_weeks,
    )
    enrich_report_stock_names(report, memberships)
    enrich_report_stock_names(
        {"holdings": report["walk_forward"]["optimized_holdings"]}, memberships
    )
    paths = write_outputs(report, args.output_dir)
    LOGGER.info(
        "Weekly backtest complete: %s", ", ".join(str(path) for path in paths.values())
    )
    print(
        json.dumps({key: str(path) for key, path in paths.items()}, ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
