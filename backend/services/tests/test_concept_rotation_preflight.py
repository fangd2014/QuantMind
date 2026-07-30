from __future__ import annotations

from scripts.data.maintenance.concept_rotation_preflight import evaluate_quality


def _healthy_measurements() -> dict[str, object]:
    return {
        "latest_date": "2026-07-29",
        "latest_rows": 5200,
        "previous_rows": 5190,
        "history_days": 26,
        "latest_duplicate_rows": 0,
        "latest_invalid_symbol_rows": 0,
        "latest_null_critical_rows": 0,
        "latest_invalid_ohlc_rows": 0,
        "latest_negative_flow_rows": 0,
        "latest_extreme_return_rows": 0,
        "recent_invalid_rows": 0,
        "pct_change_q99": 6.5,
        "latest_market_amount_ratio": 1.05,
    }


def test_evaluate_quality_accepts_complete_expected_snapshot() -> None:
    issues, warnings = evaluate_quality(
        _healthy_measurements(), expected_date="2026-07-29"
    )

    assert issues == []
    assert warnings == []


def test_evaluate_quality_blocks_stale_incomplete_or_invalid_data() -> None:
    measurements = _healthy_measurements()
    measurements.update(
        {
            "latest_date": "2026-07-28",
            "latest_rows": 800,
            "previous_rows": 5200,
            "history_days": 20,
            "latest_invalid_symbol_rows": 2,
            "latest_null_critical_rows": 3,
            "latest_invalid_ohlc_rows": 4,
            "recent_invalid_rows": 12,
        }
    )

    issues, _warnings = evaluate_quality(measurements, expected_date="2026-07-29")

    joined = "；".join(issues)
    assert "不等于应有交易日" in joined
    assert "低于1000行" in joined
    assert "截面比" in joined
    assert "少于26日" in joined
    assert "证券代码格式异常2行" in joined
    assert "关键行情字段缺失3行" in joined
    assert "OHLC价格关系异常4行" in joined
    assert "另有异常行情" in joined


def test_evaluate_quality_warns_on_suspicious_units_without_rewriting() -> None:
    measurements = _healthy_measurements()
    measurements["pct_change_q99"] = 18.0
    measurements["latest_market_amount_ratio"] = 7.0

    issues, warnings = evaluate_quality(measurements, expected_date="2026-07-29")

    assert issues == []
    assert any("涨跌幅99分位" in item for item in warnings)
    assert any("全市场成交额" in item for item in warnings)
