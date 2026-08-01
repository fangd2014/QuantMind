import { QlibStrategyParams, QlibStrategyType } from '../../types/backtest/qlib';
import { StrategyTemplate, QLIB_STRATEGY_TEMPLATES } from '../../data/qlibStrategyTemplates';

export const SECTOR_MOMENTUM_LEADER_CORE_ID = 'sector_momentum_leader_core';

const COMMON_PARAM_KEYS = [
  'signal',
  'rebalance_days',
  'buy_cost',
  'sell_cost',
  'dynamic_position',
  'market_state_symbol',
] as const;

/**
 * 从策略代码中解析策略类型。
 * 支持两种格式：
 *   1. "class": "RedisTopkStrategy" -> "TopkDropout"
 *   2. "class": "RedisWeightStrategy" -> "WeightStrategy"
 */
export function parseStrategyClassFromCode(code: string): string | null {
  if (!code) return null;

  // 匹配 STRATEGY_CONFIG 中的 class 字段
  const classMatch = code.match(/"class"\s*:\s*"(\w+)"/);
  if (!classMatch) return null;

  const className = classMatch[1];

  // 映射到前端策略类型
  const classToType: Record<string, string> = {
    RedisTopkStrategy: 'TopkDropout',
    RedisWeightStrategy: 'WeightStrategy',
    RedisStopLossStrategy: 'StopLoss',
    RedisMomentumStrategy: 'momentum',
    RedisAdaptiveStrategy: 'adaptive_drift',
    TopkDropoutStrategy: 'TopkDropout',
    WeightStrategy: 'WeightStrategy',
  };

  return classToType[className] || className;
}

/**
 * 根据策略代码推导是否需要 n_drop 参数。
 * 权重型策略（RedisWeightStrategy / WeightStrategy）不需要 n_drop。
 */
export function shouldShowNDrop(code: string | undefined, strategyType: string): boolean {
  // long_short_topk 不显示 n_drop
  if (strategyType === 'long_short_topk') return false;

  // 从代码解析策略类
  const parsedClass = parseStrategyClassFromCode(code || '');

  // 权重型策略不显示 n_drop
  if (parsedClass === 'WeightStrategy' || parsedClass === 'RedisWeightStrategy') {
    return false;
  }

  // 已知不需要 n_drop 的策略类型
  const noNDropTypes = ['alpha_cross_section', 'full_alpha_cross_section', 'score_weighted', 'VolatilityWeighted'];
  if (noNDropTypes.includes(strategyType)) {
    return false;
  }

  return true;
}

const resolveNDropByTopk = (topk: unknown): number | undefined => {
  const parsedTopk = Number(topk);
  if (!Number.isFinite(parsedTopk) || parsedTopk <= 0) return undefined;
  return Math.max(1, Math.round(parsedTopk * 0.2));
};

const shouldAutoPopulateNDrop = (strategyType: string, template: StrategyTemplate): boolean => {
  if (strategyType === 'long_short_topk') return false;
  const paramNames = new Set(template.params.map((p) => p.name));
  if (paramNames.has('n_drop')) return true;

  // 权重型策略（含 max_weight 且未声明 n_drop）不应展示 n_drop。
  // 例如：alpha_cross_section / full_alpha_cross_section / score_weighted。
  if (paramNames.has('max_weight')) return false;

  // TopK 轮动类模板即使未显式声明 n_drop，也沿用 20% 默认调仓比例。
  return paramNames.has('topk');
};

// -----------------------------------------------------------------------
// 基础兜底默认参数（当模板元数据不可用时使用）
// -----------------------------------------------------------------------

const HARDCODED_FALLBACK_PARAMS: Record<string, QlibStrategyParams> = {
  standard_topk: { topk: 50, n_drop: 10, signal: '<PRED>', rebalance_days: 3, enable_short_selling: false },
  alpha_cross_section: { topk: 50, signal: '<PRED>', min_score: 0.0, max_weight: 0.05, rebalance_days: 3, enable_short_selling: false },
  long_short_topk: { topk: 50, short_topk: 50, signal: '<PRED>', rebalance_days: 3, min_score: 0.0, max_weight: 0.05, long_exposure: 1.0, short_exposure: 1.0, enable_short_selling: true },
  momentum: { topk: 30, n_drop: 6, signal: '<PRED>', rebalance_days: 3, momentum_period: 20, momentum_weight: 0.3, enable_short_selling: false },
  StopLoss: { topk: 30, n_drop: 6, signal: '<PRED>', rebalance_days: 3, stop_loss: -0.10, take_profit: 0.20, enable_short_selling: false },
  VolatilityWeighted: { topk: 50, vol_lookback: 20, max_weight: 0.08, min_score: 0.0, signal: '<PRED>', rebalance_days: 3, enable_short_selling: false },
  sector_momentum_leader_core: {
    board_universe: 'sw_l1',
    topk_sectors: 10,
    topk_stocks: 10,
    lookback_days: 80,
    min_board_members: 10,
    min_board_coverage: 0.6,
    max_holding_days: 10,
    rebalance_days: 1,
    max_stock_weight: 0.1,
    max_board_weight: 0.2,
    weak_market_position: 0.5,
    market_breadth_reduce: 0.4,
    market_breadth_pause: 0.3,
    crowding_warning_quantile: 0.9,
    crowding_overheat_quantile: 0.95,
    crowding_penalty_max: 15,
    launch_breadth_min: 0.3,
    launch_breadth_max: 0.6,
    launch_breadth_delta_min: 0.08,
    launch_limit_count_min: 1,
    launch_limit_count_max: 2,
    launch_limit_ratio_min: 0.01,
    launch_limit_ratio_max: 0.03,
    launch_amount_ratio_min: 1.15,
    launch_amount_ratio_max: 2.5,
    launch_relative_return_min: 0.0,
    launch_relative_return_max: 0.1,
    diffusion_breadth_min: 0.6,
    diffusion_limit_count_min: 3,
    diffusion_limit_ratio_min: 0.03,
    diffusion_amount_quantile_min: 0.7,
    core_start_amount_ratio_min: 1.2,
    overheat_breadth_min: 0.8,
    overheat_relative_return_min: 0.12,
    retreat_breadth_max: 0.4,
    retreat_breadth_delta_max: -0.1,
    retreat_relative_return_max: -0.03,
    leader_float_mv_min: 5e9,
    leader_float_mv_max: 3e10,
    leader_amount_min: 3e8,
    leader_turnover_min: 0.05,
    leader_turnover_max: 0.25,
    leader_rps_min: 0.85,
    leader_amount_ratio_min: 1.2,
    leader_amount_ratio_max: 3.0,
    leader_return_3d_min: 0.03,
    leader_return_3d_max: 0.25,
    leader_upper_shadow_amount_ratio: 2.0,
    leader_upper_shadow_ratio: 0.5,
    core_float_mv_min: 1e10,
    core_mv_top_quantile: 0.2,
    core_amount_min: 1e9,
    core_volatility_min: 0.25,
    core_volatility_max: 0.5,
    core_drawdown_min: -0.12,
    core_amount_ratio_min: 1.1,
    core_amount_ratio_max: 2.5,
    core_return_5d_max: 0.2,
    leader_gap_down: -0.03,
    leader_gap_up: 0.05,
    core_gap_down: -0.02,
    core_gap_up: 0.03,
    stop_loss: -0.03,
    leader_trailing_stop: 0.06,
    core_trailing_stop: 0.05,
    board_exit_rank: 20,
    board_exit_days: 2,
    board_score_drop_exit: 20,
    enable_short_selling: false,
  },
  adaptive_drift: { topk: 50, n_drop: 10, signal: '<PRED>', rebalance_days: 3, dynamic_position: true, enable_short_selling: false },
  score_weighted: { topk: 50, signal: '<PRED>', min_score: 0.0, max_weight: 0.05, rebalance_days: 3, enable_short_selling: false },
  deep_time_series: { topk: 30, n_drop: 6, signal: '<PRED>', rebalance_days: 3, enable_short_selling: false },
  TopkDropout: { topk: 50, n_drop: 10, signal: '<PRED>', rebalance_days: 3, enable_short_selling: false },
  WeightStrategy: { topk: 50, signal: '<PRED>', min_score: 0.0, max_weight: 0.05, rebalance_days: 3, enable_short_selling: false },
  CustomStrategy: { topk: 50, n_drop: 10, signal: '<PRED>', rebalance_days: 3, enable_short_selling: false },
};

// -----------------------------------------------------------------------
// 运行时模板注册表（由 StrategyPicker 在加载后注入）
// -----------------------------------------------------------------------

let _runtimeTemplates: StrategyTemplate[] = QLIB_STRATEGY_TEMPLATES;

/**
 * 注入运行时动态模板列表（在 StrategyPicker 从后端加载模板后调用）。
 * 这样 strategyParams.ts 的函数能读到最新的参数元数据。
 */
export function registerRuntimeTemplates(templates: StrategyTemplate[]): void {
  if (templates && templates.length > 0) {
    _runtimeTemplates = templates;
  }
}

export function getStrategyTemplate(
  strategyType: string,
  templates?: StrategyTemplate[],
): StrategyTemplate | undefined {
  const list = templates && templates.length > 0 ? templates : _runtimeTemplates;
  return (
    list.find(t => t.id === strategyType) ||
    QLIB_STRATEGY_TEMPLATES.find(t => t.id === strategyType)
  );
}

export function resolveStrategyExecutionSettings(
  strategyType: string,
  tailTradeEnabled: boolean,
): {
  dealPrice: 'open' | 'close';
  signalLagDays: number;
  tailTradeLocked: boolean;
} {
  if (strategyType === SECTOR_MOMENTUM_LEADER_CORE_ID) {
    return {
      dealPrice: 'open',
      signalLagDays: 1,
      tailTradeLocked: true,
    };
  }

  return {
    dealPrice: tailTradeEnabled ? 'close' : 'open',
    signalLagDays: tailTradeEnabled ? 0 : 1,
    tailTradeLocked: false,
  };
}

// -----------------------------------------------------------------------
// 核心函数
// -----------------------------------------------------------------------

/**
 * 从模板元数据动态推导默认参数，未命中时回退到硬编码 fallback。
 */
export function getDefaultStrategyParams(
  strategyType: string,
  templates?: StrategyTemplate[]
): QlibStrategyParams {
  const template = getStrategyTemplate(strategyType, templates);

  if (template) {
    const defaults: QlibStrategyParams = {
      signal: '<PRED>',
      rebalance_days: 3,
      enable_short_selling: strategyType === 'long_short_topk',
    };

    for (const param of template.params) {
      const v = param.default;
      if (typeof v === 'number') {
        (defaults as Record<string, unknown>)[param.name] = v;
      } else if (typeof v === 'boolean') {
        (defaults as Record<string, unknown>)[param.name] = v;
      } else if (typeof v === 'string' && v.toLowerCase() === 'true') {
        (defaults as Record<string, unknown>)[param.name] = true;
      } else if (typeof v === 'string' && v.toLowerCase() === 'false') {
        (defaults as Record<string, unknown>)[param.name] = false;
      }
    }

    // 统一模板默认调仓比例为 20%（n_drop = topk * 20%，四舍五入且至少为 1）。
    if (shouldAutoPopulateNDrop(strategyType, template)) {
      const computedNDrop = resolveNDropByTopk(defaults.topk);
      if (computedNDrop !== undefined) {
        defaults.n_drop = computedNDrop;
      }
    }

    // 容错：后端模板元数据缺失多空敞口字段时，保持中性默认值，
    // 避免 UI 出现“空头 1.00x / 多头 0.00x”的错误初始态。
    if (strategyType === 'long_short_topk') {
      if (defaults.long_exposure === undefined || defaults.long_exposure === null) {
        defaults.long_exposure = 1.0;
      }
      if (defaults.short_exposure === undefined || defaults.short_exposure === null) {
        defaults.short_exposure = 1.0;
      }
    }

    return defaults;
  }

  // fallback 到硬编码
  return { ...(HARDCODED_FALLBACK_PARAMS[strategyType] || HARDCODED_FALLBACK_PARAMS.standard_topk) };
}

/**
 * 从模板元数据推导该策略允许的额外参数键名。
 */
function resolveTemplateParamKeys(strategyType: string, templates?: StrategyTemplate[]): string[] {
  const template = getStrategyTemplate(strategyType, templates);
  return template ? template.params.map(p => p.name) : [];
}

export function sanitizeStrategyParams(
  strategyType: string,
  params?: QlibStrategyParams | null,
  templates?: StrategyTemplate[],
  strategyCode?: string,
): QlibStrategyParams {
  const resolvedType = (strategyType || 'standard_topk') as QlibStrategyType | string;
  const defaults = getDefaultStrategyParams(resolvedType, templates);

  // 合并模板参数键 + 硬编码扩展键（向后兼容）
  const templateKeys = resolveTemplateParamKeys(resolvedType, templates);
  const allowedKeys = new Set<string>([
    ...COMMON_PARAM_KEYS,
    ...Object.keys(defaults),
    ...templateKeys,
  ]);

  const cleaned: QlibStrategyParams = {};
  for (const [key, value] of Object.entries(params || {})) {
    if (!allowedKeys.has(key) || value === undefined || value === null) continue;
    (cleaned as Record<string, unknown>)[key] = value;
  }

  const merged: QlibStrategyParams = { ...defaults, ...cleaned };
  if (resolvedType !== 'long_short_topk') {
    merged.enable_short_selling = false;
  }

  // 权重型策略移除 n_drop 参数
  if (!shouldShowNDrop(strategyCode, resolvedType)) {
    delete merged.n_drop;
  }

  return merged;
}
