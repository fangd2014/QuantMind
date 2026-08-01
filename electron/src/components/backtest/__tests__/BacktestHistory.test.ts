import { describe, expect, it } from 'vitest';
import { resolveStrategyName } from '../BacktestHistory';

describe('BacktestHistory helpers', () => {
  it('renders sector strategy history rows with the Chinese strategy name', () => {
    expect(
      resolveStrategyName({
        backtest_id: 'bt-1',
        created_at: '2026-08-01T09:00:00Z',
        strategy_name: 'sector_momentum_leader_core',
      } as any)
    ).toBe('板块动量轮动+龙头中军选股');
  });
});
