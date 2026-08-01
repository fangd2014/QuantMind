import React from 'react';
import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QlibStrategyConfigurator } from '../QlibStrategyConfigurator';
import { getDefaultStrategyParams, getStrategyTemplate, SECTOR_MOMENTUM_LEADER_CORE_ID } from '../../../shared/qlib/strategyParams';

describe('QlibStrategyConfigurator', () => {
  it('renders sector strategy metadata labels, descriptions, ranges and select options', () => {
    render(
      <QlibStrategyConfigurator
        strategyType={SECTOR_MOMENTUM_LEADER_CORE_ID}
        params={getDefaultStrategyParams(SECTOR_MOMENTUM_LEADER_CORE_ID)}
        onChange={vi.fn()}
        template={getStrategyTemplate(SECTOR_MOMENTUM_LEADER_CORE_ID)}
      />
    );

    expect(screen.getByLabelText('板块范围')).toBeInTheDocument();
    expect(screen.getByRole('option', { name: '申万一级行业' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: '同花顺概念' })).toBeInTheDocument();
    expect(screen.getByLabelText('龙头开盘跳空下限')).toHaveValue(-0.03);
    expect(screen.getByText('启动期龙头 t+1 开盘跳空过滤下限。')).toBeInTheDocument();
    expect(screen.getAllByText(/范围 -0.15 ~ 0，步长 0.01/).length).toBeGreaterThan(0);
  });

  it('emits sector metadata-backed value changes', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const template = getStrategyTemplate(SECTOR_MOMENTUM_LEADER_CORE_ID);

    const Harness = () => {
      const [params, setParams] = React.useState(getDefaultStrategyParams(SECTOR_MOMENTUM_LEADER_CORE_ID));
      return (
        <QlibStrategyConfigurator
          strategyType={SECTOR_MOMENTUM_LEADER_CORE_ID}
          params={params}
          onChange={(next) => {
            setParams(next);
            onChange(next);
          }}
          template={template}
        />
      );
    };

    render(<Harness />);

    await user.selectOptions(screen.getByLabelText('板块范围'), 'ths_concept');
    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ board_universe: 'ths_concept' })
    );

    const rebalanceInput = screen.getByLabelText('调仓周期');
    fireEvent.change(rebalanceInput, { target: { value: '2' } });
    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ rebalance_days: 2 })
    );
    expect(rebalanceInput).toHaveValue(2);

    fireEvent.change(rebalanceInput, { target: { value: '4' } });
    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ rebalance_days: 4 })
    );
    expect(rebalanceInput).toHaveValue(4);

    const gapInput = screen.getByLabelText('龙头开盘跳空下限');
    fireEvent.change(gapInput, { target: { value: '-0.05' } });
    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ leader_gap_down: -0.05 })
    );
  });
});
