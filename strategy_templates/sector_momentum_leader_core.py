"""板块动量轮动+龙头中军选股（专用原生策略）。"""

# 模板代码用于模板 API 展示；实际回测由 StrategyFactory 的专用 Builder 路由，
# 不允许作为未知模板动态执行并退化到普通 TopK。
STRATEGY_CONFIG = {
    "class": "RedisSectorMomentumLeaderCoreStrategy",
    "module_path": "backend.services.engine.qlib_app.utils.extended_strategies",
    "kwargs": {
        "signal": "<SECTOR_MOMENTUM_LEADER_CORE>",
        "topk": 10,
        "n_drop": 10,
        "max_holding_days": 10,
        "rebalance_days": 1,
    },
}
