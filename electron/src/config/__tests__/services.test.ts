import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

describe('web deployment service routing', () => {
  beforeEach(() => {
    vi.resetModules();
    vi.stubEnv('VITE_API_BASE_URL', '');
    vi.stubEnv('VITE_WS_BASE_URL', '');
    window.localStorage.clear();
    window.history.replaceState({}, '', '/quantmind/#/auth/login');
    delete (window as any).electronAPI;
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it('keeps QuantMind API and websocket traffic inside their nginx namespaces', async () => {
    window.localStorage.setItem('quantmind_server_url', 'http://stale-host:8000');

    const { SERVICE_ENDPOINTS, SERVICE_URLS } = await import('../services');

    expect(SERVICE_URLS.API_GATEWAY).toBe('/quantmind-api');
    expect(SERVICE_ENDPOINTS.USER_SERVICE).toBe('/quantmind-api/api/v1');
    expect(SERVICE_URLS.WEBSOCKET_MARKET).toBe(
      'ws://localhost:3000/quantmind-ws/api/v1/ws/market',
    );
  });

  it('does not let build-time service overrides escape the QuantMind nginx namespace', async () => {
    vi.stubEnv('VITE_API_BASE_URL', '/api/v1');
    vi.stubEnv('VITE_USER_API_URL', '/api/v1');
    vi.stubEnv('VITE_API_GATEWAY_URL', '/api/v1');
    vi.stubEnv('VITE_WS_BASE_URL', 'ws://stale-host:8003/api/v1/ws/market');

    const { SERVICE_ENDPOINTS, SERVICE_URLS } = await import('../services');

    expect(SERVICE_URLS.API_GATEWAY).toBe('/quantmind-api');
    expect(SERVICE_ENDPOINTS.USER_SERVICE).toBe('/quantmind-api/api/v1');
    expect(SERVICE_URLS.WEBSOCKET_MARKET).toBe(
      'ws://localhost:3000/quantmind-ws/api/v1/ws/market',
    );
  });

  it('splits a relative nginx API namespace without dropping the application prefix', async () => {
    const { splitApiServiceUrl } = await import('../services');

    expect(splitApiServiceUrl('/quantmind-api/api/v1')).toEqual({
      baseURL: '/quantmind-api',
      apiPrefix: '/api/v1',
    });
    expect(splitApiServiceUrl('/api/v1')).toEqual({
      baseURL: '',
      apiPrefix: '/api/v1',
    });
  });
});
