from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from scripts.data.maintenance.sync_etf_futures_from_tushare import (
    TushareClient,
    effective_start,
    fetch_etf_daily,
    fetch_etf_instruments,
    futures_mapping_records,
    futures_product,
    normalize_etf_symbol,
    yearly_ranges,
)


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class TushareAssetSyncTests(unittest.TestCase):
    def test_fund_basic_fallback_excludes_non_etf_products(self) -> None:
        class Client:
            def query(self, api_name, _params, _fields=()):
                if api_name == "etf_basic":
                    raise ValueError("no permission")
                return [
                    {"ts_code": "510300.SH", "name": "沪深300ETF"},
                    {"ts_code": "180101.SZ", "name": "基础设施REIT"},
                ]

        rows = fetch_etf_instruments(Client())
        self.assertEqual([item["symbol"] for item in rows], ["SH510300"])

    def test_recent_etf_sync_batches_by_trade_date(self) -> None:
        class Client:
            def query(self, api_name, params, _fields=()):
                if api_name == "trade_cal":
                    return [{"cal_date": "20260724", "is_open": 1}]
                if api_name == "fund_adj":
                    return [
                        {
                            "ts_code": "510300.SH",
                            "trade_date": "20260724",
                            "adj_factor": 1.5,
                        }
                    ]
                self.assertEqual(params, {"trade_date": "20260724"})
                return [
                    {
                        "ts_code": "510300.SH",
                        "trade_date": "20260724",
                        "close": 4.25,
                    },
                    {
                        "ts_code": "159919.SZ",
                        "trade_date": "20260724",
                        "close": 4.20,
                    },
                ]

        client = Client()
        client.assertEqual = self.assertEqual
        rows, failures, attempts = fetch_etf_daily(
            client,
            [{"ts_code": "510300.SH"}],
            "20260720",
            "20260724",
        )
        self.assertEqual(attempts, 1)
        self.assertEqual(failures, [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["symbol"], "SH510300")
        self.assertEqual(rows[0]["adj_factor"], 1.5)

    def test_etf_symbol_uses_project_prefix_format(self) -> None:
        self.assertEqual(normalize_etf_symbol("510300.SH"), "SH510300")
        self.assertEqual(normalize_etf_symbol("159919.SZ"), "SZ159919")
        self.assertIsNone(normalize_etf_symbol("IF2608.CFX"))

    def test_futures_product_has_separate_namespace(self) -> None:
        self.assertEqual(futures_product("IF2608.CFX"), "IF")
        self.assertEqual(futures_product("IM2612.CFX"), "IM")
        self.assertIsNone(futures_product("CU2608.SHF"))

    def test_watermark_advances_without_reloading_last_day(self) -> None:
        self.assertEqual(effective_start("20160101", "20260724"), "20260725")
        self.assertEqual(effective_start("20260726", "20260724"), "20260726")
        self.assertEqual(effective_start("20160101", None), "20160101")

    def test_year_ranges_are_bounded(self) -> None:
        self.assertEqual(
            yearly_ranges("20251220", "20260103"),
            [("20251220", "20251231"), ("20260101", "20260103")],
        )

    def test_mapping_records_require_real_contract(self) -> None:
        rows = futures_mapping_records(
            [
                {
                    "trade_date": "20260724",
                    "continuous_code": "IF.CFX",
                    "mapping_ts_code": "IF2608.CFX",
                    "product_code": "IF",
                },
                {
                    "trade_date": "20260724",
                    "continuous_code": "IH.CFX",
                    "mapping_ts_code": "",
                    "product_code": "IH",
                },
            ]
        )
        self.assertEqual(
            rows,
            [("2026-07-24", "IF.CFX", "IF", "IF2608.CFX", "tushare")],
        )

    @patch("urllib.request.urlopen")
    def test_http_client_maps_fields_without_exposing_token(self, urlopen) -> None:
        urlopen.return_value = _Response(
            {
                "code": 0,
                "msg": None,
                "data": {
                    "fields": ["ts_code", "trade_date", "close"],
                    "items": [["510300.SH", "20260724", 4.25]],
                },
            }
        )
        rows = TushareClient(token="secret", interval=0).query(
            "fund_daily", {"ts_code": "510300.SH"}
        )
        self.assertEqual(
            rows,
            [{"ts_code": "510300.SH", "trade_date": "20260724", "close": 4.25}],
        )
        request = urlopen.call_args.args[0]
        self.assertIn(b'"token": "secret"', request.data)
        self.assertNotIn("secret", repr(rows))


if __name__ == "__main__":
    unittest.main()
