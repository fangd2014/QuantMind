import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import CloudStrategyManagement from '../CloudStrategyManagement';

const loadStrategies = vi.fn();

vi.mock('../../../../services/strategyManagementService', () => ({
  strategyManagementService: {
    loadStrategies: (...args: unknown[]) => loadStrategies(...args),
    deleteStrategy: vi.fn(),
    syncTemplates: vi.fn(),
  },
}));

vi.mock('../../../auth/hooks', () => ({
  useAuth: () => ({ user: { id: 'test-user' } }),
}));

describe('CloudStrategyManagement', () => {
  beforeEach(() => {
    loadStrategies.mockResolvedValue([
      {
        id: 'sys_sector_momentum_leader_core',
        name: '板块动量轮动+龙头中军选股',
        strategy_type: 'sector_momentum_leader_core',
        status: 'draft',
        is_system: true,
        created_at: '2026-08-02T00:00:00Z',
      },
    ]);
  });

  it('labels system strategies as built-in and hides delete', async () => {
    render(<CloudStrategyManagement />);

    await screen.findByText('板块动量轮动+龙头中军选股');
    expect(screen.getByText('内置')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'delete' })).not.toBeInTheDocument();
  });
});
