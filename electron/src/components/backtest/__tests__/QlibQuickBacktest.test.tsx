import React from 'react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QlibQuickBacktest } from '../QlibQuickBacktest';

const runBacktestMock = vi.fn();

vi.mock('../../../stores/backtestCenterStore', () => ({
  useBacktestCenterStore: vi.fn((selector: (state: any) => unknown) =>
    selector({
      backtestConfig: {},
      activeModule: 'quick-backtest',
    })
  ),
}));

vi.mock('../../../store', () => ({
  useAppSelector: vi.fn(() => 'CN'),
}));

vi.mock('../../../features/auth/services/authService', () => ({
  authService: {
    getStoredUser: vi.fn(() => ({ id: 'user-1' })),
  },
}));

vi.mock('../../../services/backtestService', () => ({
  backtestService: {
    getQlibDataRange: vi.fn(async () => ({ exists: false })),
    runBacktest: runBacktestMock,
    pollStatus: vi.fn(),
    logError: vi.fn(async () => undefined),
  },
}));

vi.mock('../../../services/strategyManagementService', () => ({
  strategyManagementService: {
    getStrategy: vi.fn(),
  },
}));

vi.mock('../QlibResultComponents', () => ({
  QlibResultDisplay: () => <div data-testid="qlib-result-display" />,
  ErrorLogModal: () => null,
}));

vi.mock('../StrategyPicker', () => ({
  StrategyPicker: ({ onStrategySelected }: any) => (
    <button
      type="button"
      onClick={() =>
        onStrategySelected('', {
          id: 'sector_momentum_leader_core',
          name: '板块动量轮动+龙头中军选股',
          source: 'template',
          code: '',
          description: 'mock template',
          is_qlib_format: true,
          language: 'qlib',
        }, {
          board_universe: 'sw_l1',
          topk_sectors: 10,
          max_holding_days: 10,
        })
      }
    >
      select sector strategy
    </button>
  ),
}));

vi.mock('../QlibStrategyConfigurator', () => ({
  QlibStrategyConfigurator: ({ strategyType, template }: any) => (
    <div>
      <div data-testid="config-strategy-type">{strategyType}</div>
      <div data-testid="config-template-id">{template?.id || 'missing-template'}</div>
    </div>
  ),
}));

describe('QlibQuickBacktest', () => {
  beforeEach(() => {
    window.localStorage.clear();
    runBacktestMock.mockReset();
    runBacktestMock.mockResolvedValue({
      status: 'completed',
      backtest_id: 'bt-1',
      created_at: '2026-08-01T09:30:00Z',
    });
  });

  it('passes sector template metadata into configurator, hides legacy rebalance control, and forces safe run payload', async () => {
    const user = userEvent.setup();
    window.localStorage.setItem('backtest_tail_trade_mode', '1');

    render(<QlibQuickBacktest />);

    expect(screen.getByRole('button', { name: /尾盘交易/i })).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'select sector strategy' }));

    await waitFor(() => {
      expect(screen.getByTestId('config-strategy-type')).toHaveTextContent('sector_momentum_leader_core');
      expect(screen.getByTestId('config-template-id')).toHaveTextContent('sector_momentum_leader_core');
    });

    expect(screen.queryByRole('button', { name: /尾盘交易/i })).not.toBeInTheDocument();
    expect(screen.queryByText('调仓周期 (Rebalance)')).not.toBeInTheDocument();
    expect(screen.getByText('固定 T+1 开盘成交')).toBeInTheDocument();
    expect(
      screen.getByText(/固定使用 `deal_price=open` 与 `signal_lag_days=1`/)
    ).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /立即执行回测/i }));

    await waitFor(() => {
      expect(runBacktestMock).toHaveBeenCalledTimes(1);
    });

    expect(runBacktestMock).toHaveBeenCalledWith(
      expect.objectContaining({
        strategy_type: 'sector_momentum_leader_core',
        deal_price: 'open',
        signal_lag_days: 1,
        strategy_params: expect.objectContaining({
          board_universe: 'sw_l1',
          topk_sectors: 10,
          max_holding_days: 10,
          rebalance_days: 1,
        }),
      })
    );
  });
});
