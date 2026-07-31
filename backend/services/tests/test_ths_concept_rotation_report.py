from __future__ import annotations

from pathlib import Path

from scripts.analysis.concept_rotation_report import (
    build_feishu_payload,
    render_interactive_html,
)
from scripts.analysis.ths_concept_rotation_report import (
    THS_REPORT_PROFILE,
    THS_PUBLIC_DETAIL_URL,
    THS_PUBLIC_INDEX_URL,
    load_ths_concept_universe,
    parse_ths_public_concepts,
    parse_ths_public_members,
)


def test_load_ths_concept_universe_uses_a_share_concepts_and_cache(
    tmp_path: Path,
) -> None:
    concepts = [
        {
            "ts_code": f"88{index:04d}.TI",
            "name": f"同花顺概念{index}",
            "count": 8,
            "exchange": "A",
            "type": "N",
        }
        for index in range(60)
    ]
    calls: list[tuple[str, dict[str, str]]] = []

    def fake_query(
        api_name: str,
        params: dict[str, str],
        _fields: tuple[str, ...],
    ) -> list[dict[str, object]]:
        calls.append((api_name, params))
        if api_name == "ths_index":
            return concepts
        return [
            {
                "ts_code": params["ts_code"],
                "con_code": f"600{member:03d}.SH",
                "con_name": f"成分股{member}",
                "is_new": "Y",
            }
            for member in range(1, 9)
        ]

    boards, memberships = load_ths_concept_universe(
        tmp_path, cache_days=7, query=fake_query
    )

    assert len(boards) == 60
    assert boards[0]["source"] == "THS_TUSHARE"
    assert boards[0]["level"] == "concept"
    assert len(memberships[boards[0]["code"]]) == 8
    assert calls[0] == ("ths_index", {"exchange": "A", "type": "N"})
    assert sum(name == "ths_member" for name, _params in calls) == 60

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("fresh cache should avoid Tushare requests")

    cached_boards, cached_memberships = load_ths_concept_universe(
        tmp_path, cache_days=7, query=fail_if_called
    )

    assert cached_boards == boards
    assert cached_memberships == memberships


def _public_member_fixture(name_prefix: str = "成分股") -> bytes:
    rows = "".join(
        f"""
        <tr>
          <td>{index}</td>
          <td><a href="/stock/{600000 + index}/">{600000 + index}</a></td>
          <td><a>{name_prefix}{index}</a></td>
          <td>10.00</td>
        </tr>
        """
        for index in range(1, 6)
    )
    return f"""
    <html><head><meta charset="utf-8"></head><body>
      <div class="body m-pager-box" id="maincont">
        <table><tbody>{rows}</tbody></table>
        <div class="m-pager"><a class="cur" page="1">1</a>
          <span class="page_info">1/1</span></div>
      </div>
    </body></html>
    """.encode()


def test_parse_ths_public_pages() -> None:
    index_html = """
    <html><head><meta charset="utf-8"></head><body>
      <a href="https://q.10jqka.com.cn/gn/detail/code/308614/">人工智能概念</a>
      <a href="/gn/detail/code/309121/" target="_blank"><span>AI PC</span></a>
      <a href="/not-a-concept/123">忽略</a>
    </body></html>
    """.encode()

    concepts = parse_ths_public_concepts(index_html)
    members, page_count = parse_ths_public_members(_public_member_fixture())

    assert concepts == [
        {
            "code": "THS308614",
            "route_code": "308614",
            "name": "人工智能概念",
            "level": "concept",
            "source": "THS_PUBLIC",
            "reported_count": 0,
        },
        {
            "code": "THS309121",
            "route_code": "309121",
            "name": "AI PC",
            "level": "concept",
            "source": "THS_PUBLIC",
            "reported_count": 0,
        },
    ]
    assert page_count == 1
    assert members[0] == {"symbol": "600001", "name": "成分股1"}
    assert len(members) == 5


def test_load_ths_concept_universe_falls_back_to_public_pages(
    tmp_path: Path,
) -> None:
    index_html = (
        "<html><head><meta charset='utf-8'></head><body>"
        + "".join(
            f"<a href='/gn/detail/code/{300000 + index}/'>公开概念{index}</a>"
            for index in range(50)
        )
        + "</body></html>"
    ).encode()
    requested: list[str] = []

    def denied_query(*_args, **_kwargs):
        raise ValueError("没有接口(ths_index)访问权限")

    def fake_http_get(url: str) -> bytes:
        requested.append(url)
        if url == THS_PUBLIC_INDEX_URL:
            return index_html
        assert url.startswith(THS_PUBLIC_DETAIL_URL.split("{")[0])
        return _public_member_fixture()

    boards, memberships = load_ths_concept_universe(
        tmp_path,
        cache_days=7,
        query=denied_query,
        public_http_get=fake_http_get,
    )

    assert len(boards) == 50
    assert boards[0]["source"] == "THS_PUBLIC"
    assert boards[0]["reported_count"] == 5
    assert "route_code" not in boards[0]
    assert len(memberships[boards[0]["code"]]) == 5
    assert len(requested) == 51


def test_ths_feishu_payload_is_a_separate_concept_message() -> None:
    report = {
        "report_profile": THS_REPORT_PROFILE,
        "market": {"latest_date": "2026-07-31", "regime": "活跃"},
        "top": [{"name": "机器人概念", "quadrant": "领先区"}],
        "buy_candidates": [],
        "leading_control_picks": [
            {
                "symbol": "SH600001",
                "stock_name": "概念龙头",
                "industry_name": "机器人概念",
                "stage": "洗盘",
                "main_force_intent": "洗盘吸筹",
                "intent_confidence": "中",
                "intent_evidence": "缩量守住MA20",
                "reason": "机器人概念位于领先区；控盘量价代理成立。",
                "invalidation": "跌破MA20时失效",
                "action_label": "次日关注",
            }
        ],
        "watchlist": [],
    }

    payload = build_feishu_payload(
        report,
        "http://example/ths.pdf",
        "http://example/ths.html",
    )

    serialized = str(payload)
    assert "QuantMind 同花顺概念板块轮动日报 2026-07-31" in serialized
    assert "强势同花顺概念" in serialized
    assert "机器人概念" in serialized
    assert "SH600001" in serialized
    assert "申万行业轮动日报" not in serialized


def test_ths_interactive_html_uses_concept_labels(tmp_path: Path) -> None:
    board = {
        "code": "885001.TI",
        "name": "机器人概念",
        "score": 91.0,
        "quadrant": "领先区",
        "rs_ratio": 108.0,
        "rs_momentum": 105.0,
        "breadth20": 0.72,
        "breadth_delta5": 0.12,
        "member_count": 42,
        "mapped_count": 39,
        "coverage": 0.93,
        "leader_symbol": "SH600001",
        "leader_name": "概念龙头",
        "leader_confirmed": True,
        "eligible": True,
    }
    report = {
        "report_profile": THS_REPORT_PROFILE,
        "market": {"latest_date": "2026-07-31", "regime": "活跃"},
        "plot": [board],
        "focus_industries": [board],
        "leading_control_picks": [],
    }
    target = tmp_path / "ths.html"

    render_interactive_html(report, target)

    html = target.read_text(encoding="utf-8")
    assert "同花顺概念板块 RRG 四象限" in html
    assert "同花顺A股概念板块" in html
    assert "搜索同花顺概念" in html
    assert "领先区 / 改善区概念板块清单" in html
    assert "申万一级行业" not in html
