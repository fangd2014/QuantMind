import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { QlibExpertBacktest } from '../QlibExpertBacktest';

const extractConfig = vi.fn();
const saveStrategy = vi.fn();

vi.mock('@monaco-editor/react', () => ({
  default: ({ value }: { value?: string }) => (
    <textarea aria-label="策略代码" readOnly value={value || ''} />
  ),
}));

vi.mock('../../../services/strategyManagementService', () => ({
  strategyManagementService: {
    extractConfig: (...args: unknown[]) => extractConfig(...args),
    saveStrategy: (...args: unknown[]) => saveStrategy(...args),
    readLocalFile: vi.fn(),
  },
}));

vi.mock('../../../services/backtestService', () => ({
  backtestService: {
    getQlibDataRange: vi.fn(async () => ({ exists: false })),
    runBacktest: vi.fn(),
    pollStatus: vi.fn(),
  },
}));

vi.mock('../MultiStockCodeInput', () => ({
  MultiStockCodeInput: () => null,
}));

vi.mock('../QlibResultComponents', () => ({
  QlibBacktestResultDisplay: () => null,
  QlibResultDisplay: () => null,
  ErrorLogModal: () => null,
}));

describe('QlibExpertBacktest 保存到个人中心', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    extractConfig.mockResolvedValue({ topk: 50, n_drop: 5 });
    saveStrategy.mockResolvedValue({ id: '303' });
    vi.spyOn(window, 'alert').mockImplementation(() => undefined);
  });

  it('先提取配置，再把专家策略保存为个人策略', async () => {
    const user = userEvent.setup();
    render(<QlibExpertBacktest />);

    await user.click(screen.getByRole('button', { name: '保存到个人中心' }));
    await user.click(screen.getByRole('button', { name: '确认保存' }));

    await waitFor(() => {
      expect(extractConfig).toHaveBeenCalledTimes(1);
      expect(saveStrategy).toHaveBeenCalledTimes(1);
    });

    expect(String(extractConfig.mock.calls[0][0])).toContain('STRATEGY_CONFIG');
    expect(saveStrategy).toHaveBeenCalledWith(
      expect.objectContaining({
        source: 'personal',
        tags: ['ExpertMode'],
        parameters: { topk: 50, n_drop: 5 },
      })
    );
    expect(window.alert).toHaveBeenCalledWith(
      '策略已成功保存至个人中心（已通过配置合规性验证）'
    );
  });
});
