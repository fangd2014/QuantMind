import { describe, expect, it } from 'vitest';
import {
  getDefaultStrategyParams,
  getStrategyTemplate,
  resolveStrategyExecutionSettings,
  sanitizeStrategyParams,
  SECTOR_MOMENTUM_LEADER_CORE_ID,
} from '../strategyParams';

const selectMeta = (
  name: string,
  defaultValue: string,
  options: Array<{ value: string | number; label: string; description?: string }>,
) => ({
  name,
  default: defaultValue,
  min: undefined,
  max: undefined,
  step: undefined,
  options,
});

const intMeta = (
  name: string,
  defaultValue: number,
  min: number,
  max: number,
  step = 1,
) => ({
  name,
  default: defaultValue,
  min,
  max,
  step,
  options: undefined,
});

const numberMeta = (
  name: string,
  defaultValue: number,
  min: number,
  max: number,
  step: number,
) => ({
  name,
  default: defaultValue,
  min,
  max,
  step,
  options: undefined,
});

const EXPECTED_SECTOR_PARAM_METADATA = [
  selectMeta('board_universe', 'sw_l1', [
    { value: 'sw_l1', label: '申万一级行业', description: '申万 2021 一级行业历史成分。' },
    { value: 'ths_concept', label: '同花顺概念', description: '同花顺 A 股概念历史成分。' },
  ]),
  intMeta('topk_sectors', 10, 5, 20),
  intMeta('topk_stocks', 10, 1, 10),
  intMeta('lookback_days', 80, 40, 252),
  intMeta('min_board_members', 10, 10, 30),
  numberMeta('min_board_coverage', 0.6, 0.4, 1.0, 0.05),
  intMeta('max_holding_days', 10, 1, 10),
  intMeta('rebalance_days', 1, 1, 5),
  numberMeta('max_stock_weight', 0.1, 0.02, 0.2, 0.01),
  numberMeta('max_board_weight', 0.2, 0.05, 0.5, 0.05),
  numberMeta('weak_market_position', 0.5, 0.0, 1.0, 0.05),
  numberMeta('market_breadth_reduce', 0.4, 0.2, 0.8, 0.05),
  numberMeta('market_breadth_pause', 0.3, 0.1, 0.7, 0.05),
  numberMeta('crowding_warning_quantile', 0.9, 0.7, 0.94, 0.01),
  numberMeta('crowding_overheat_quantile', 0.95, 0.8, 0.99, 0.01),
  intMeta('crowding_penalty_max', 15, 0, 30),
  numberMeta('launch_breadth_min', 0.3, 0.1, 0.6, 0.05),
  numberMeta('launch_breadth_max', 0.6, 0.3, 0.8, 0.05),
  numberMeta('launch_breadth_delta_min', 0.08, 0.0, 0.3, 0.01),
  intMeta('launch_limit_count_min', 1, 0, 10),
  intMeta('launch_limit_count_max', 2, 1, 15),
  numberMeta('launch_limit_ratio_min', 0.01, 0.0, 0.1, 0.005),
  numberMeta('launch_limit_ratio_max', 0.03, 0.01, 0.2, 0.005),
  numberMeta('launch_amount_ratio_min', 1.15, 0.5, 3.0, 0.05),
  numberMeta('launch_amount_ratio_max', 2.5, 1.0, 5.0, 0.1),
  numberMeta('launch_relative_return_min', 0.0, -0.1, 0.2, 0.01),
  numberMeta('launch_relative_return_max', 0.1, 0.02, 0.3, 0.01),
  numberMeta('diffusion_breadth_min', 0.6, 0.3, 0.9, 0.05),
  intMeta('diffusion_limit_count_min', 3, 1, 20),
  numberMeta('diffusion_limit_ratio_min', 0.03, 0.0, 0.2, 0.005),
  numberMeta('diffusion_amount_quantile_min', 0.7, 0.4, 0.95, 0.05),
  numberMeta('core_start_amount_ratio_min', 1.2, 0.8, 3.0, 0.05),
  numberMeta('overheat_breadth_min', 0.8, 0.5, 1.0, 0.05),
  numberMeta('overheat_relative_return_min', 0.12, 0.05, 0.4, 0.01),
  numberMeta('retreat_breadth_max', 0.4, 0.1, 0.7, 0.05),
  numberMeta('retreat_breadth_delta_max', -0.1, -0.4, 0.0, 0.01),
  numberMeta('retreat_relative_return_max', -0.03, -0.2, 0.0, 0.01),
  numberMeta('leader_float_mv_min', 5e9, 1e9, 2e10, 1e9),
  numberMeta('leader_float_mv_max', 3e10, 1e10, 1e11, 5e9),
  numberMeta('leader_amount_min', 3e8, 5e7, 2e9, 5e7),
  numberMeta('leader_turnover_min', 0.05, 0.0, 0.2, 0.01),
  numberMeta('leader_turnover_max', 0.25, 0.1, 0.6, 0.01),
  numberMeta('leader_rps_min', 0.85, 0.5, 0.99, 0.01),
  numberMeta('leader_amount_ratio_min', 1.2, 0.5, 3.0, 0.05),
  numberMeta('leader_amount_ratio_max', 3.0, 1.0, 6.0, 0.1),
  numberMeta('leader_return_3d_min', 0.03, -0.1, 0.2, 0.01),
  numberMeta('leader_return_3d_max', 0.25, 0.05, 0.6, 0.01),
  numberMeta('leader_upper_shadow_amount_ratio', 2.0, 1.0, 5.0, 0.1),
  numberMeta('leader_upper_shadow_ratio', 0.5, 0.2, 0.9, 0.05),
  numberMeta('core_float_mv_min', 1e10, 2e9, 1e11, 1e9),
  numberMeta('core_mv_top_quantile', 0.2, 0.05, 0.5, 0.05),
  numberMeta('core_amount_min', 1e9, 1e8, 5e9, 1e8),
  numberMeta('core_volatility_min', 0.25, 0.05, 0.5, 0.05),
  numberMeta('core_volatility_max', 0.5, 0.2, 1.0, 0.05),
  numberMeta('core_drawdown_min', -0.12, -0.5, -0.01, 0.01),
  numberMeta('core_amount_ratio_min', 1.1, 0.5, 3.0, 0.05),
  numberMeta('core_amount_ratio_max', 2.5, 1.0, 6.0, 0.1),
  numberMeta('core_return_5d_max', 0.2, 0.05, 0.6, 0.01),
  numberMeta('leader_gap_down', -0.03, -0.15, 0.0, 0.01),
  numberMeta('leader_gap_up', 0.05, 0.0, 0.2, 0.01),
  numberMeta('core_gap_down', -0.02, -0.15, 0.0, 0.01),
  numberMeta('core_gap_up', 0.03, 0.0, 0.15, 0.01),
  numberMeta('stop_loss', -0.03, -0.2, -0.01, 0.01),
  numberMeta('leader_trailing_stop', 0.06, 0.02, 0.2, 0.01),
  numberMeta('core_trailing_stop', 0.05, 0.02, 0.2, 0.01),
  intMeta('board_exit_rank', 20, 10, 50),
  intMeta('board_exit_days', 2, 1, 5),
  intMeta('board_score_drop_exit', 20, 5, 50),
] as const;

describe('strategyParams', () => {
  it('returns sector momentum leader core defaults from template metadata', () => {
    const params = getDefaultStrategyParams(SECTOR_MOMENTUM_LEADER_CORE_ID);

    expect(params.board_universe).toBe('sw_l1');
    expect(params.topk_sectors).toBe(10);
    expect(params.topk_stocks).toBe(10);
    expect(params.max_holding_days).toBe(10);
    expect(params.leader_gap_down).toBe(-0.03);
    expect(params.core_gap_up).toBe(0.03);
  });

  it('matches all 68 sector metadata entries against an independent expectation table', () => {
    const template = getStrategyTemplate(SECTOR_MOMENTUM_LEADER_CORE_ID);
    expect(template?.name).toBe('板块动量轮动+龙头中军选股');

    const actualParams = template?.params ?? [];
    expect(actualParams).toHaveLength(68);
    expect(actualParams.map((param) => param.name)).toEqual(
      EXPECTED_SECTOR_PARAM_METADATA.map((param) => param.name)
    );

    const actualByName = new Map(actualParams.map((param) => [param.name, param]));
    for (const expectedParam of EXPECTED_SECTOR_PARAM_METADATA) {
      const actual = actualByName.get(expectedParam.name);
      expect(actual, `missing metadata for ${expectedParam.name}`).toBeDefined();
      expect(actual?.default).toEqual(expectedParam.default);
      expect(actual?.min).toBe(expectedParam.min);
      expect(actual?.max).toBe(expectedParam.max);
      expect(actual?.step).toBe(expectedParam.step);
      expect(actual?.options).toEqual(expectedParam.options);
    }
  });

  it('sanitizes unknown keys while preserving valid sector parameters', () => {
    const sanitized = sanitizeStrategyParams(SECTOR_MOMENTUM_LEADER_CORE_ID, {
      board_universe: 'ths_concept',
      topk_sectors: 12,
      rebalance_days: 4,
      leader_gap_down: -0.05,
      unknown_param: 999,
    });

    expect(sanitized.board_universe).toBe('ths_concept');
    expect(sanitized.topk_sectors).toBe(12);
    expect(sanitized.rebalance_days).toBe(4);
    expect(sanitized.leader_gap_down).toBe(-0.05);
    expect('unknown_param' in sanitized).toBe(false);
  });

  it('forces sector strategy execution to next-open semantics', () => {
    expect(resolveStrategyExecutionSettings(SECTOR_MOMENTUM_LEADER_CORE_ID, true)).toEqual({
      dealPrice: 'open',
      signalLagDays: 1,
      tailTradeLocked: true,
    });

    expect(resolveStrategyExecutionSettings('standard_topk', true)).toEqual({
      dealPrice: 'close',
      signalLagDays: 0,
      tailTradeLocked: false,
    });
  });
});
