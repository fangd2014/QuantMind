import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { StrategyManagementModule } from '../StrategyManagementModule';

const navigate = vi.fn();
const loadStrategies = vi.fn();

vi.mock('react-router-dom', () => ({
  useNavigate: () => navigate,
}));

vi.mock('react-redux', () => ({
  useDispatch: () => vi.fn(),
}));

vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, layout: _layout, ...props }: React.HTMLAttributes<HTMLDivElement> & { layout?: boolean }) => (
      <div {...props}>{children}</div>
    ),
  },
}));

vi.mock('../../../services/strategyManagementService', () => ({
  strategyManagementService: {
    loadStrategies: (...args: unknown[]) => loadStrategies(...args),
    deleteStrategy: vi.fn(),
  },
}));

vi.mock('../../../stores/backtestCenterStore', () => ({
  useBacktestCenterStore: () => ({ setActiveModule: vi.fn() }),
}));

vi.mock('../../../store/slices/aiStrategySlice', () => ({
  setCurrentTab: vi.fn(),
}));

describe('StrategyManagementModule', () => {
  beforeEach(() => {
    navigate.mockReset();
    loadStrategies.mockResolvedValue([
      {
        id: 'sys_sector_momentum_leader_core',
        name: '板块动量轮动+龙头中军选股',
        status: 'draft',
        is_system: true,
        created_at: '2026-08-02T00:00:00Z',
      },
    ]);
  });

  it('marks system strategies as built-in and exposes view-only actions', async () => {
    const user = userEvent.setup();
    render(<StrategyManagementModule />);

    await screen.findByText('板块动量轮动+龙头中军选股');
    expect(screen.getByText('内置')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /查看/ })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /编辑/ })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /删除/ })).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /查看/ }));
    await waitFor(() => {
      expect(navigate).toHaveBeenCalledWith(
        '/ai-ide?strategyId=sys_sector_momentum_leader_core'
      );
    });
  });
});
