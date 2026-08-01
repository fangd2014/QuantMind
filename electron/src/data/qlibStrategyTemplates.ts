/**
 * Qlib 策略模板类型定义 + 轻量离线 Fallback
 *
 * 主数据源：后端 GET /api/v1/strategies/templates（动态加载）
 * 本文件中的 QLIB_STRATEGY_TEMPLATES 仅作离线 / 后端不可用时的兜底展示。
 * 如需动态获取最新模板，请使用 strategyTemplateService.getTemplates()。
 */

export interface StrategyTemplate {
  id: string;
  name: string;
  description: string;
  category: 'basic' | 'advanced' | 'risk_control';
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  code: string;
  params: StrategyTemplateParam[];
  execution_defaults?: Record<string, unknown>;
  live_defaults?: Record<string, unknown>;
  live_config_tips?: string[];
}

export interface StrategyTemplateOption {
  value: string | number;
  label: string;
  description?: string;
}

export interface StrategyTemplateParam {
  name: string;
  description: string;
  default: number | string | boolean;
  label?: string;
  min?: number;
  max?: number;
  step?: number;
  type?: 'number' | 'integer' | 'boolean' | 'select' | 'string';
  options?: StrategyTemplateOption[];
}

function intParam(
  name: string,
  label: string,
  description: string,
  defaultValue: number,
  min: number,
  max: number,
  step = 1,
): StrategyTemplateParam {
  return {
    name,
    label,
    description,
    default: defaultValue,
    min,
    max,
    step,
    type: 'integer',
  };
}

function numberParam(
  name: string,
  label: string,
  description: string,
  defaultValue: number,
  min: number,
  max: number,
  step: number,
): StrategyTemplateParam {
  return {
    name,
    label,
    description,
    default: defaultValue,
    min,
    max,
    step,
    type: 'number',
  };
}

function selectParam(
  name: string,
  label: string,
  description: string,
  defaultValue: string,
  options: StrategyTemplateOption[],
): StrategyTemplateParam {
  return {
    name,
    label,
    description,
    default: defaultValue,
    type: 'select',
    options,
  };
}

const SECTOR_MOMENTUM_LEADER_CORE_PARAMS: StrategyTemplateParam[] = [
  selectParam('board_universe', '板块范围', '板块候选池口径。', 'sw_l1', [
    { value: 'sw_l1', label: '申万一级行业', description: '申万 2021 一级行业历史成分。' },
    { value: 'ths_concept', label: '同花顺概念', description: '同花顺 A 股概念历史成分。' },
  ]),
  intParam('topk_sectors', '热门板块数量', '每日展示并评估的热门板块 TopK。', 10, 5, 20),
  intParam('topk_stocks', '候选股票数量', '去重后进入组合打分的股票上限。', 10, 1, 10),
  intParam('lookback_days', '回看天数', '板块与角色信号计算所需的历史热身长度。', 80, 40, 252),
  intParam('min_board_members', '最少板块成分数', '低于该成分数的板块不参与排名。', 10, 10, 30),
  numberParam('min_board_coverage', '最小成分覆盖率', '板块当日有效成员覆盖率下限。', 0.6, 0.4, 1.0, 0.05),
  intParam('max_holding_days', '最大主动持有天数', '主动持仓最长保留的交易日数。', 10, 1, 10),
  intParam('rebalance_days', '调仓周期', '信号刷新与组合调仓的交易日间隔。', 1, 1, 5),
  numberParam('max_stock_weight', '单票权重上限', '单只股票组合权重上限。', 0.1, 0.02, 0.2, 0.01),
  numberParam('max_board_weight', '单板块权重上限', '同一板块的组合权重上限。', 0.2, 0.05, 0.5, 0.05),
  numberParam('weak_market_position', '弱市最大仓位', '弱市环境下组合允许的最高总仓位。', 0.5, 0.0, 1.0, 0.05),
  numberParam('market_breadth_reduce', '降仓广度阈值', '低于该全市场广度阈值时降仓。', 0.4, 0.2, 0.8, 0.05),
  numberParam('market_breadth_pause', '暂停开仓广度阈值', '低于该全市场广度阈值时暂停新开仓。', 0.3, 0.1, 0.7, 0.05),
  numberParam('crowding_warning_quantile', '拥挤预警分位', '拥挤度进入扣分区间的起点。', 0.9, 0.7, 0.94, 0.01),
  numberParam('crowding_overheat_quantile', '拥挤过热分位', '拥挤度触发过热状态的阈值。', 0.95, 0.8, 0.99, 0.01),
  intParam('crowding_penalty_max', '拥挤最大扣分', '拥挤惩罚的最高扣分值。', 15, 0, 30),
  numberParam('launch_breadth_min', '启动期广度下限', '启动期要求的板块广度下限。', 0.3, 0.1, 0.6, 0.05),
  numberParam('launch_breadth_max', '启动期广度上限', '启动期要求的板块广度上限。', 0.6, 0.3, 0.8, 0.05),
  numberParam('launch_breadth_delta_min', '启动期广度增量下限', '启动期近 3 日广度提升下限。', 0.08, 0.0, 0.3, 0.01),
  intParam('launch_limit_count_min', '启动期涨停数下限', '启动期板块涨停家数下限。', 1, 0, 10),
  intParam('launch_limit_count_max', '启动期涨停数上限', '启动期板块涨停家数上限。', 2, 1, 15),
  numberParam('launch_limit_ratio_min', '启动期涨停率下限', '启动期涨停率下限。', 0.01, 0.0, 0.1, 0.005),
  numberParam('launch_limit_ratio_max', '启动期涨停率上限', '启动期涨停率上限。', 0.03, 0.01, 0.2, 0.005),
  numberParam('launch_amount_ratio_min', '启动期成交额占比下限', '启动期成交额占比相对中位数下限。', 1.15, 0.5, 3.0, 0.05),
  numberParam('launch_amount_ratio_max', '启动期成交额占比上限', '启动期成交额占比相对中位数上限。', 2.5, 1.0, 5.0, 0.1),
  numberParam('launch_relative_return_min', '启动期相对收益下限', '启动期 5 日相对收益下限。', 0.0, -0.1, 0.2, 0.01),
  numberParam('launch_relative_return_max', '启动期相对收益上限', '启动期 5 日相对收益上限。', 0.1, 0.02, 0.3, 0.01),
  numberParam('diffusion_breadth_min', '扩散期广度下限', '扩散期要求的板块广度下限。', 0.6, 0.3, 0.9, 0.05),
  intParam('diffusion_limit_count_min', '扩散期涨停数下限', '扩散期板块涨停家数下限。', 3, 1, 20),
  numberParam('diffusion_limit_ratio_min', '扩散期涨停率下限', '扩散期板块涨停率下限。', 0.03, 0.0, 0.2, 0.005),
  numberParam('diffusion_amount_quantile_min', '扩散期成交额分位下限', '扩散期成交额占比横截面分位下限。', 0.7, 0.4, 0.95, 0.05),
  numberParam('core_start_amount_ratio_min', '中军启动量比下限', '扩散期中军金额/5 日均额下限。', 1.2, 0.8, 3.0, 0.05),
  numberParam('overheat_breadth_min', '过热广度下限', '板块进入过热状态的广度下限。', 0.8, 0.5, 1.0, 0.05),
  numberParam('overheat_relative_return_min', '过热相对收益下限', '板块进入过热状态的 5 日相对收益下限。', 0.12, 0.05, 0.4, 0.01),
  numberParam('retreat_breadth_max', '退潮广度上限', '板块退潮判定的广度上限。', 0.4, 0.1, 0.7, 0.05),
  numberParam('retreat_breadth_delta_max', '退潮广度增量上限', '板块退潮判定的近 3 日广度变化上限。', -0.1, -0.4, 0.0, 0.01),
  numberParam('retreat_relative_return_max', '退潮相对收益上限', '板块退潮判定的 5 日相对收益上限。', -0.03, -0.2, 0.0, 0.01),
  numberParam('leader_float_mv_min', '龙头流通市值下限', '启动期龙头候选流通市值下限（元）。', 5e9, 1e9, 2e10, 1e9),
  numberParam('leader_float_mv_max', '龙头流通市值上限', '启动期龙头候选流通市值上限（元）。', 3e10, 1e10, 1e11, 5e9),
  numberParam('leader_amount_min', '龙头成交额下限', '启动期龙头候选 5 日均成交额下限（元）。', 3e8, 5e7, 2e9, 5e7),
  numberParam('leader_turnover_min', '龙头换手率下限', '启动期龙头候选换手率下限。', 0.05, 0.0, 0.2, 0.01),
  numberParam('leader_turnover_max', '龙头换手率上限', '启动期龙头候选换手率上限。', 0.25, 0.1, 0.6, 0.01),
  numberParam('leader_rps_min', '龙头 RPS 下限', '启动期龙头候选全市场 RPS20 下限。', 0.85, 0.5, 0.99, 0.01),
  numberParam('leader_amount_ratio_min', '龙头量比下限', '启动期龙头候选金额/5 日均额下限。', 1.2, 0.5, 3.0, 0.05),
  numberParam('leader_amount_ratio_max', '龙头量比上限', '启动期龙头候选金额/5 日均额上限。', 3.0, 1.0, 6.0, 0.1),
  numberParam('leader_return_3d_min', '龙头 3 日收益下限', '启动期龙头候选近 3 日收益下限。', 0.03, -0.1, 0.2, 0.01),
  numberParam('leader_return_3d_max', '龙头 3 日收益上限', '启动期龙头候选近 3 日收益上限。', 0.25, 0.05, 0.6, 0.01),
  numberParam('leader_upper_shadow_amount_ratio', '龙头长上影量比阈值', '启动期龙头高位放量长上影过滤的量比阈值。', 2.0, 1.0, 5.0, 0.1),
  numberParam('leader_upper_shadow_ratio', '龙头长上影比例阈值', '启动期龙头高位放量长上影过滤的上影比例阈值。', 0.5, 0.2, 0.9, 0.05),
  numberParam('core_float_mv_min', '中军流通市值下限', '扩散期中军候选流通市值下限（元）。', 1e10, 2e9, 1e11, 1e9),
  numberParam('core_mv_top_quantile', '中军市值分位阈值', '扩散期中军要求的板块内市值前分位阈值。', 0.2, 0.05, 0.5, 0.05),
  numberParam('core_amount_min', '中军成交额下限', '扩散期中军候选 5 日均成交额下限（元）。', 1e9, 1e8, 5e9, 1e8),
  numberParam('core_volatility_min', '中军波动率下限', '扩散期中军候选 20 日年化波动率下限。', 0.25, 0.05, 0.5, 0.05),
  numberParam('core_volatility_max', '中军波动率上限', '扩散期中军候选 20 日年化波动率上限。', 0.5, 0.2, 1.0, 0.05),
  numberParam('core_drawdown_min', '中军最大回撤下限', '扩散期中军候选 20 日最大回撤下限。', -0.12, -0.5, -0.01, 0.01),
  numberParam('core_amount_ratio_min', '中军量比下限', '扩散期中军候选金额/5 日均额下限。', 1.1, 0.5, 3.0, 0.05),
  numberParam('core_amount_ratio_max', '中军量比上限', '扩散期中军候选金额/5 日均额上限。', 2.5, 1.0, 6.0, 0.1),
  numberParam('core_return_5d_max', '中军 5 日收益上限', '扩散期中军候选近 5 日收益上限。', 0.2, 0.05, 0.6, 0.01),
  numberParam('leader_gap_down', '龙头开盘跳空下限', '启动期龙头 t+1 开盘跳空过滤下限。', -0.03, -0.15, 0.0, 0.01),
  numberParam('leader_gap_up', '龙头开盘跳空上限', '启动期龙头 t+1 开盘跳空过滤上限。', 0.05, 0.0, 0.2, 0.01),
  numberParam('core_gap_down', '中军开盘跳空下限', '扩散期中军 t+1 开盘跳空过滤下限。', -0.02, -0.15, 0.0, 0.01),
  numberParam('core_gap_up', '中军开盘跳空上限', '扩散期中军 t+1 开盘跳空过滤上限。', 0.03, 0.0, 0.15, 0.01),
  numberParam('stop_loss', '止损阈值', '收盘止损阈值，下一可交易日开盘退出。', -0.03, -0.2, -0.01, 0.01),
  numberParam('leader_trailing_stop', '龙头移动止盈', '龙头距持仓期最高收盘回撤阈值。', 0.06, 0.02, 0.2, 0.01),
  numberParam('core_trailing_stop', '中军移动止盈', '中军距持仓期最高收盘回撤阈值。', 0.05, 0.02, 0.2, 0.01),
  intParam('board_exit_rank', '板块退出排名阈值', '连续跌出该排名后触发板块退潮退出。', 20, 10, 50),
  intParam('board_exit_days', '板块退出确认天数', '板块退潮条件连续满足的确认天数。', 2, 1, 5),
  intParam('board_score_drop_exit', '板块分数回撤阈值', '板块分较入场下降达到该值时退出。', 20, 5, 50),
];

/**
 * 轻量离线 fallback（仅保留最常用的 3 个入门策略）。
 * 完整模板列表由后端动态提供，优先通过 strategyTemplateService.getTemplates() 获取。
 */
export const QLIB_STRATEGY_TEMPLATES: StrategyTemplate[] = [
  {
    id: 'standard_topk',
    name: '默认 Top-K 选股策略',
    description: '最经典的量化选股逻辑。每日截面排名，精选最具潜力的 Top-K 标的，等权持仓。',
    category: 'basic',
    difficulty: 'beginner',
    code: `"""
默认 Top-K 选股策略 (Standard Top-K Strategy)
[Native] 核心逻辑：Top-K 选股 + 零换手强制约束
"""
STRATEGY_CONFIG = {
    "class": "RedisTopkStrategy",
    "kwargs": {
        "signal": "<PRED>",
        "topk": 50,
        "n_drop": 10,
    }
}
`,
    params: [
      { name: 'topk', description: '持仓股票总数', default: 50, min: 5, max: 100 }
    ]
  },
  {
    id: 'StopLoss',
    name: '止损止盈策略',
    description: '在标准 TopK 选股基础上叠加硬性止损/止盈规则，一旦触发立即强制平仓。',
    category: 'risk_control',
    difficulty: 'beginner',
    code: `"""
止损止盈策略 (Stop-Loss / Take-Profit Strategy)
[Native] 核心逻辑：浮亏超过 stop_loss 或浮盈超过 take_profit 时强制平仓。
"""
STRATEGY_CONFIG = {
    "class": "RedisStopLossStrategy",
    "kwargs": {
        "signal": "<PRED>",
        "topk": 30,
        "n_drop": 6,
        "stop_loss": -0.08,
        "take_profit": 0.15,
    }
}
`,
    params: [
      { name: 'topk', description: '选股数量', default: 30, min: 5, max: 100 },
      { name: 'stop_loss', description: '止损阈值 (如 -0.08 = -8%)', default: -0.08, min: -0.3, max: -0.01 },
      { name: 'take_profit', description: '止盈阈值 (如 0.15 = +15%)', default: 0.15, min: 0.05, max: 0.5 }
    ]
  },
  {
    id: 'risk_guard_topk',
    name: '大盘风控 Top-K 选股策略',
    description: '在基础 Top-K 选股之上叠加基本面硬过滤、行业集中度约束与大盘下行降仓。',
    category: 'risk_control',
    difficulty: 'intermediate',
    code: `"""
大盘风控 Top-K 选股策略 (Risk-Guarded Top-K)
[Native] 核心逻辑：Top-K 选股 + 基本面硬过滤 + 大盘周期降仓。
"""
STRATEGY_CONFIG = {
    "class": "RedisRiskGuardTopkStrategy",
    "kwargs": {
        "signal": "<PRED>",
        "topk": 50,
        "n_drop": 10,
        "rebalance_days": 3,
        "market_state_symbol": "SH000300",
        "market_state_window": 20,
        "industry_cap_ratio": 0.30,
        "listed_days_min": 120,
        "turnover_rate_min": 0.5,
        "turnover_rate_max": 15.0,
        "beta_20_max": 1.8,
        "float_mv_min": 500000000,
    }
}
`,
    params: [
      { name: 'topk', description: '持仓股票总数', default: 50, min: 5, max: 200 },
      { name: 'n_drop', description: '每期替换数量', default: 10, min: 0, max: 200 },
      { name: 'rebalance_days', description: '调仓周期 (天)', default: 3, min: 1, max: 60 },
      { name: 'market_state_symbol', description: '市场状态参考指数', default: 'SH000300' },
      { name: 'market_state_window', description: '大盘状态判定窗口 (交易日)', default: 20, min: 5, max: 120 },
      { name: 'industry_cap_ratio', description: '单行业持仓上限占比', default: 0.3, min: 0.1, max: 0.6 },
      { name: 'listed_days_min', description: '上市天数下限', default: 120, min: 20, max: 500 },
      { name: 'turnover_rate_min', description: '换手率下限 (%)', default: 0.5, min: 0, max: 10 },
      { name: 'turnover_rate_max', description: '换手率上限 (%)', default: 15.0, min: 1, max: 80 },
      { name: 'beta_20_max', description: '20日 Beta 上限', default: 1.8, min: 0.5, max: 3 },
      { name: 'float_mv_min', description: '流通市值下限 (元)', default: 500000000, min: 100000000, max: 10000000000 },
    ]
  },
  {
    id: 'alpha_cross_section',
    name: '截面 Alpha 预测策略',
    description: '根据预测分自动分配资金权重，分高者重仓。',
    category: 'advanced',
    difficulty: 'intermediate',
    code: `"""
截面 Alpha 预测策略 (Cross-sectional Alpha)
[Native] 核心逻辑：按模型预测分比例进行权重分配。
"""
STRATEGY_CONFIG = {
    "class": "RedisWeightStrategy",
    "kwargs": {
        "signal": "<PRED>",
        "topk": 50,
        "min_score": 0.0,
        "max_weight": 0.05,
    }
}
`,
    params: [
      { name: 'topk', description: '参与权重的标的数量', default: 50, min: 10, max: 200 },
      { name: 'max_weight', description: '单票持仓上限 (0~1)', default: 0.05, min: 0.01, max: 0.2 }
    ]
  },
  {
    id: 'sector_momentum_leader_core',
    name: '板块动量轮动+龙头中军选股',
    description: '先筛热门板块，再按启动期龙头 / 扩散期中军规则选股，严格使用 T+1 开盘成交口径。',
    category: 'advanced',
    difficulty: 'advanced',
    code: `"""
板块动量轮动+龙头中军选股
[Native] 专用后端信号路径：按模板 ID 路由，不依赖 <PRED>。
"""
`,
    params: SECTOR_MOMENTUM_LEADER_CORE_PARAMS,
  }
];

/**
 * 按分类获取 fallback 模板
 */
export function getTemplatesByCategory(category: StrategyTemplate['category']): StrategyTemplate[] {
  return QLIB_STRATEGY_TEMPLATES.filter(t => t.category === category);
}

/**
 * 按难度获取 fallback 模板
 */
export function getTemplatesByDifficulty(difficulty: StrategyTemplate['difficulty']): StrategyTemplate[] {
  return QLIB_STRATEGY_TEMPLATES.filter(t => t.difficulty === difficulty);
}

/**
 * 按 ID 查询 fallback 模板
 */
export function getTemplateById(id: string): StrategyTemplate | undefined {
  return QLIB_STRATEGY_TEMPLATES.find(t => t.id === id);
}
