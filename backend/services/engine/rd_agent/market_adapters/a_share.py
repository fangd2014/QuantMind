"""A股市场适配器"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import register_adapter
from .base import BacktestConfig, DataConfig, MarketAdapter


@register_adapter
class AShareAdapter(MarketAdapter):
    """A股 (CSI300) 市场适配器"""

    market_id = "a_share"
    market_name = "A股"
    description = "中国 A 股市场 (CSI300)，Qlib Alpha158 因子集"

    def get_data_config(self) -> DataConfig:
        provider_uri = self.get_qlib_provider_uri()
        return DataConfig(
            provider_uri=provider_uri,
            data_dir=self._get_quantdb_dir() or provider_uri,
            calendar="day",
            market="csi300",
        )

    def get_qlib_provider_uri(self) -> str:
        from backend.shared.qlib_paths import resolve_qlib_provider_uri

        return resolve_qlib_provider_uri("CN")

    def is_data_ready(self) -> bool:
        """仅在 Qlib 日历、标的和特征目录均存在时判定为就绪。"""
        provider = Path(self.get_qlib_provider_uri())
        return (
            (provider / "calendars" / "day.txt").is_file()
            and (provider / "instruments" / "all.txt").is_file()
            and (provider / "features").is_dir()
        )

    @staticmethod
    def _get_quantdb_dir() -> str | None:
        """获取 QuantDB 数据目录路径。"""
        quantdb_dir = os.getenv("QM_QUANTDB_DATA_DIR", "").strip()
        if quantdb_dir and os.path.isdir(quantdb_dir):
            return quantdb_dir
        for d in ("/data/quantdb", "/app/data/quantdb"):
            if os.path.isdir(d):
                return d
        return None

    def get_backtest_config(self) -> BacktestConfig:
        return BacktestConfig(
            annualization_days=252,
            limit_threshold=0.1,
            commission_rate=0.001,
            min_commission=5.0,
            region="cn",
            needs_adjustment_factor=True,
        )

    def get_factor_set(self) -> dict[str, str]:
        """从 QuantDB L1 因子动态加载因子集（每类 5 个代表性因子）。

        若 QuantDB 不可用则回退到 Alpha158(20) 子集。
        """
        try:
            from backend.services.engine.data_platform.quantdb_hub import QuantDBDataHub
            hub = QuantDBDataHub.get_instance()
            if hub.available:
                categories = hub.fetch_l1_factor_categories()
                cat_list = categories.get("categories", [])
                if cat_list:
                    result = {}
                    for cat in cat_list:
                        for feat in cat.get("sample_features", [])[:5]:
                            result[feat] = cat["name"] + "类因子"
                    return result if result else self._fallback_factors()
        except Exception:
            pass
        return self._fallback_factors()

    def generate_base_factors_json(self, output_dir: str) -> str | None:
        """生成 RD-Agent 可读取的 base_factors.json 文件。

        从 feature catalog 读取 L1/L2 因子名列表，写入符合 RD-Agent 格式的
        JSON 文件（feature_name -> expression/description）。供 LLM 在因子挖掘时
        参考已有因子作为构建基础。

        Returns:
            生成的文件路径，失败返回 None。
        """
        json_path = os.path.join(output_dir, "base_factors.json")
        try:
            from backend.services.engine.data_platform.quantdb_hub import QuantDBDataHub
            hub = QuantDBDataHub.get_instance()
            if not hub.available:
                return None

            categories = hub.fetch_l1_factor_categories()
            if not categories.get("categories"):
                return None

            # 从 catalog 文件读取完整 feature 列表
            catalog_path = (
                Path(__file__).resolve().parents[5]
                / "config" / "features" / "model_training_feature_catalog_v1.json"
            )
            all_features: list[dict] = []
            if catalog_path.exists():
                import json as _json
                with open(catalog_path, encoding="utf-8") as f:
                    catalog = _json.load(f)
                for cat in catalog.get("categories", []):
                    cat_name = cat.get("name", "因子")
                    for feat in cat.get("features", []):
                        all_features.append({**feat, "_category": cat_name})

            factors: dict[str, str] = {}

            # Alpha158 表达式模板：给 LLM 提供可直接仿写的 Qlib 语法范例
            known_expr = self._fallback_factors()
            factors.update(known_expr)

            for feat in all_features:
                feat_key = feat.get("key", "")
                if not feat_key or feat_key in factors:
                    continue
                # catalog 因子已在 QuantDB 预计算，给 LLM 提供分类/含义/公式作为参考
                parts = [feat.get("_category", "因子")]
                desc = (feat.get("description") or "").strip()
                if desc:
                    parts.append(desc)
                formula = (feat.get("formula") or "").strip()
                if formula:
                    parts.append(f"公式: {formula}")
                factors[feat_key] = (
                    " | ".join(parts) + "（QuantDB 已预计算，仅供参考实现思路，不可直接引用）"
                )

            if not factors:
                return None

            os.makedirs(output_dir, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                import json as _json
                _json.dump(factors, f, ensure_ascii=False, indent=2)
            import logging
            logging.getLogger(__name__).info(
                "[%s] Generated base_factors.json: %d factors -> %s",
                self.market_id, len(factors), json_path,
            )
            return json_path

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "[%s] Failed to generate base_factors.json: %s", self.market_id, e
            )
            return None

    @staticmethod
    def _fallback_factors() -> dict[str, str]:
        """Alpha158 风格默认因子集兜底（K线形态 + 动量 + 均线 + 波动率 + 量价 + 趋势强度）。"""
        return {
            "KMID": "($close - $open) / $open",
            "KLEN": "($high - $low) / $open",
            "KMID2": "($close - $open) / ($high - $low + 1e-12)",
            "KUP": "($high - Max($open, $close)) / $open",
            "KUP2": "($high - Max($open, $close)) / ($high - $low + 1e-12)",
            "KLOW": "(Min($open, $close) - $low) / $open",
            "KLOW2": "(Min($open, $close) - $low) / ($high - $low + 1e-12)",
            "KSFT": "(2 * $close - $high - $low) / $open",
            "KSFT2": "(2 * $close - $high - $low) / ($high - $low + 1e-12)",
            "ROC5": "Ref($close, 5) / $close - 1",
            "ROC10": "Ref($close, 10) / $close - 1",
            "ROC20": "Ref($close, 20) / $close - 1",
            "ROC30": "Ref($close, 30) / $close - 1",
            "ROC60": "Ref($close, 60) / $close - 1",
            "ROC120": "Ref($close, 120) / $close - 1",
            "ROC250": "Ref($close, 250) / $close - 1",
            "MA5": "Mean($close, 5) / $close",
            "MA10": "Mean($close, 10) / $close",
            "MA20": "Mean($close, 20) / $close",
            "MA30": "Mean($close, 30) / $close",
            "MA60": "Mean($close, 60) / $close",
            "STD5": "Std($close, 5) / $close",
            "STD10": "Std($close, 10) / $close",
            "STD20": "Std($close, 20) / $close",
            "STD30": "Std($close, 30) / $close",
            "STD60": "Std($close, 60) / $close",
            # 量价相关（A股量价背离是最经典 alpha 来源，原兜底集缺失）
            "CORR5": "Corr(Log($close), Log($volume + 1), 5)",
            "CORR10": "Corr(Log($close), Log($volume + 1), 10)",
            "CORR20": "Corr(Log($close), Log($volume + 1), 20)",
            "VOLCH5": "Log($volume / Ref($volume, 5))",
            "VOLCH10": "Log($volume / Ref($volume, 10))",
            # 趋势强度（线性回归斜率/拟合度/残差）
            "BETA20": "Slope($close, 20) / $close",
            "BETA60": "Slope($close, 60) / $close",
            "RSQR20": "Rsquare($close, 20)",
            "RESI20": "Resi($close, 20) / $close",
        }

    def get_prop_setting_class(self) -> str:
        return "rdagent.app.qlib_rd_loop.conf.FactorBasePropSetting"

    def get_env_overrides(self) -> dict[str, str]:
        # API key resolution: AI_IDE_LLM_API_KEY > AI_IDE_API_KEY > OPENAI_API_KEY
        api_key = (
            os.getenv("AI_IDE_LLM_API_KEY")
            or os.getenv("AI_IDE_API_KEY")
            or os.getenv("OPENAI_API_KEY", "")
        )
        return {
            "QLIB_PROVIDER_URI": self.get_qlib_provider_uri(),
            "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL", ""),
            "OPENAI_API_KEY": api_key,
            "CHAT_MODEL": os.getenv("CHAT_MODEL", ""),
            "REASONING_MODEL": os.getenv("CHAT_MODEL", ""),
            "CHAT_STREAM": "false",
            "CHAT_MAX_TOKENS": os.getenv("CHAT_MAX_TOKENS", "8000"),
            "CHAT_TEMPERATURE": os.getenv("CHAT_TEMPERATURE", "0.3"),
        }
