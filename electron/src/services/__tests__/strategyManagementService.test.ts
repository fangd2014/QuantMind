import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  post: vi.fn(),
  requestUse: vi.fn(),
  responseUse: vi.fn(),
}));

vi.mock('axios', () => ({
  default: {
    create: () => ({
      post: mocks.post,
      interceptors: {
        request: { use: mocks.requestUse },
        response: { use: mocks.responseUse },
      },
    }),
  },
  __esModule: true,
}));

vi.mock('../../features/auth/services/authService', () => ({
  authService: {
    getAccessToken: vi.fn(() => null),
    getTenantId: vi.fn(() => 'default'),
    handle401Error: vi.fn(),
  },
}));

import '../strategyManagementService';

describe('StrategyManagementService routing', () => {
  beforeEach(() => {
    window.history.replaceState({}, '', '/quantmind/#/backtest');
    window.localStorage.clear();
  });

  it('routes expert strategy extraction and saving to the engine service', () => {
    const applyRequestConfig = mocks.requestUse.mock.calls[0][0];
    const config = applyRequestConfig({ headers: {} });

    expect(config.baseURL).toBe('/quantmind-engine');
  });
});
