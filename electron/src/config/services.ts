/**
 * 统一服务端口配置
 * 所有服务端口的唯一配置来源
 */
export const SERVICE_PORTS = {
  // 前端服务
  FRONTEND_DEV: 3000,

  // 后端服务 (统一通过网关 8000)
  API_GATEWAY: 8000,
  MARKET_DATA: 8000,    // 原 8002
  DATA_SERVICE: 8000,   // 原 8002
  USER_SERVICE: 8000,   // 原 8011
  AI_STRATEGY: 8000,    // 原 8007
  STOCK_QUERY: 8000,    // 原 8010
  TRADING: 8000,        // 原 8004
  QLIB_SERVICE: 8000, // Qlib快速回测服务（收敛至网关）

  // WebSocket服务
  WEBSOCKET_MARKET: 8003,

  // 数据库
  REDIS: 6379,
} as const;

const ENV: Record<string, any> = typeof import.meta !== 'undefined' ? (import.meta as any).env || {} : {};

// 动态服务器配置（桌面端用户设置）
let dynamicServerUrl: string | null = null;
const SERVER_URL_STORAGE_KEY = 'quantmind_server_url';

function readPersistedServerUrl(): string | null {
  if (typeof window === 'undefined') return null;
  try {
    return localStorage.getItem(SERVER_URL_STORAGE_KEY)?.trim() || null;
  } catch {
    return null;
  }
}

function persistServerUrl(url: string | null): void {
  if (typeof window === 'undefined') return;
  try {
    if (url) {
      localStorage.setItem(SERVER_URL_STORAGE_KEY, url);
    } else {
      localStorage.removeItem(SERVER_URL_STORAGE_KEY);
    }
  } catch {
    // ignore storage failures
  }
}

/**
 * 检测是否为 Electron 桌面环境
 */
export function isElectronEnv(): boolean {
  return typeof window !== 'undefined' && typeof (window as any).electronAPI === 'object';
}

/**
 * 初始化动态服务器配置（桌面端启动时调用）
 */
export async function initDynamicServerUrl(): Promise<void> {
  const persisted = readPersistedServerUrl();
  if (persisted) {
    dynamicServerUrl = persisted;
    return;
  }

  const electronApi = typeof window !== 'undefined' ? (window as any).electronAPI : null;
  if (isElectronEnv() && typeof electronApi?.getServerUrl === 'function') {
    try {
      const url = await electronApi.getServerUrl();
      if (url && typeof url === 'string') {
        dynamicServerUrl = url.replace(/\/+$/, '');
        persistServerUrl(dynamicServerUrl);
      }
    } catch (e) {
      console.warn('[services] Failed to get server URL from config:', e);
    }
  }
}

/**
 * 设置动态服务器配置（用户设置后调用）
 */
export function setDynamicServerUrl(url: string): void {
  dynamicServerUrl = url ? url.replace(/\/+$/, '') : null;
  persistServerUrl(dynamicServerUrl);
}

/**
 * 获取当前动态服务器配置
 */
export function getDynamicServerUrl(): string | null {
  if (!isElectronEnv()) return null;
  return dynamicServerUrl || readPersistedServerUrl();
}

const HOST = ENV.VITE_SERVICE_HOST || '';
const HTTP_PROTOCOL = ENV.VITE_HTTP_PROTOCOL || 'http';
const WS_PROTOCOL = HTTP_PROTOCOL === 'https' ? 'wss' : 'ws';

export function normalizeBaseUrl(url: string): string {
  if (!url) return url;
  let normalized = url.replace(/\/+$/, '');
  if (normalized.endsWith('/api/v1')) {
    normalized = normalized.slice(0, -'/api/v1'.length);
  }
  return normalized;
}

export function splitApiServiceUrl(url: string): { baseURL: string; apiPrefix: string } {
  const normalized = String(url || '').replace(/\/+$/, '');
  if (!normalized) return { baseURL: '', apiPrefix: '' };

  if (normalized.startsWith('/')) {
    const apiPrefixIndex = normalized.indexOf(API_PATHS.V1);
    if (apiPrefixIndex >= 0) {
      return {
        baseURL: normalized.slice(0, apiPrefixIndex),
        apiPrefix: normalized.slice(apiPrefixIndex),
      };
    }
    return { baseURL: normalized, apiPrefix: '' };
  }

  try {
    const parsed = new URL(normalized);
    let apiPrefix = parsed.pathname.replace(/\/$/, '');
    if (apiPrefix === '/') apiPrefix = '';
    parsed.pathname = '';
    parsed.search = '';
    parsed.hash = '';
    return { baseURL: parsed.toString().replace(/\/$/, ''), apiPrefix };
  } catch {
    return { baseURL: normalized, apiPrefix: '' };
  }
}

const API_BASE = normalizeBaseUrl(ENV.VITE_API_BASE_URL || '');

function getWebDeploymentBaseUrl(): string {
  if (typeof window === 'undefined') return '';
  const pathname = window.location.pathname || '';
  if (pathname.startsWith('/quantmind/') || pathname === '/quantmind') {
    return '/quantmind-api';
  }
  return '';
}

function getWebEngineDeploymentBaseUrl(): string {
  if (typeof window === 'undefined') return '';
  const pathname = window.location.pathname || '';
  if (pathname.startsWith('/quantmind/') || pathname === '/quantmind') {
    return '/quantmind-engine';
  }
  return '';
}

/**
 * 获取基础 URL（优先使用动态配置）
 */
function getBaseUrl(): string {
  // 子路径 Web 部署的 Nginx 命名空间是不可覆盖的运行时约束。
  // 必须优先于构建变量和桌面端持久化配置，否则认证请求会落入
  // 同一 Nginx 上其他应用的 /api 路由并随机表现为 400/502。
  const webDeploymentBaseUrl = getWebDeploymentBaseUrl();
  if (webDeploymentBaseUrl) return webDeploymentBaseUrl;

  // 仅桌面端允许使用持久化服务器地址。Web 端必须保持同域，避免旧配置
  // 把 /quantmind 下的认证请求错误发送到同机其他应用的 /api 路由。
  if (isElectronEnv()) {
    if (dynamicServerUrl) return dynamicServerUrl;
    const persisted = readPersistedServerUrl();
    if (persisted) return persisted;
  }
  return API_BASE || getWebDeploymentBaseUrl();
}

// WebSocket URL 构建
const getWebSocketUrl = () => {
  if (typeof window !== 'undefined' && getWebDeploymentBaseUrl()) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}/quantmind-ws/api/v1/ws/market`;
  }

  const persisted = getDynamicServerUrl();
  // 桌面端使用动态配置
  if (persisted) {
    return `${persisted.replace(/^http/, 'ws')}/api/v1/ws/market`;
  }
  const configured = ENV.VITE_WS_BASE_URL || ENV.VITE_WEBSOCKET_MARKET_URL;
  if (configured) {
    return configured;
  }
  const gateway = getBaseUrl();
  if (gateway) {
    return `${gateway.replace(/^http/, 'ws')}/api/v1/ws/market`;
  }
  // Web 部署使用相对路径，通过 Nginx 代理
  if (typeof window !== 'undefined') {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}/ws/api/v1/ws/market`;
  }
  return '';
};

export const SERVICE_URLS = {
  get API_GATEWAY() { return getWebDeploymentBaseUrl() || normalizeBaseUrl(ENV.VITE_API_GATEWAY_URL) || getBaseUrl(); },
  get MARKET_DATA() { return getWebDeploymentBaseUrl() || normalizeBaseUrl(ENV.VITE_MARKET_DATA_API_URL) || getBaseUrl(); },
  get DATA_SERVICE() { return getWebDeploymentBaseUrl() || normalizeBaseUrl(ENV.VITE_DATA_SERVICE_API_URL) || getBaseUrl(); },
  get USER_SERVICE() { return getWebDeploymentBaseUrl() || normalizeBaseUrl(ENV.VITE_USER_API_URL) || getBaseUrl(); },
  get AI_STRATEGY() { return getWebDeploymentBaseUrl() || normalizeBaseUrl(ENV.VITE_AI_STRATEGY_API_URL) || getBaseUrl(); },
  get STOCK_QUERY() { return getWebDeploymentBaseUrl() || normalizeBaseUrl(ENV.VITE_STOCK_QUERY_API_URL) || getBaseUrl(); },
  get TRADING() { return getWebDeploymentBaseUrl() || normalizeBaseUrl(ENV.VITE_TRADING_API_URL) || getBaseUrl(); },
  get QLIB_SERVICE() { return getWebDeploymentBaseUrl() || normalizeBaseUrl(ENV.VITE_QLIB_SERVICE_URL) || getBaseUrl(); },
  get ENGINE_SERVICE() { return getWebEngineDeploymentBaseUrl() || normalizeBaseUrl(ENV.VITE_ENGINE_SERVICE_URL) || getBaseUrl(); },
  get WEBSOCKET_MARKET() { return getWebSocketUrl(); },
} as const;

// API路径配置
export const API_PATHS = {
  V1: '/api/v1',
  HEALTH: '/health',
  STRATEGIES: '/strategies',
  MARKET_DATA: '/market-data',
  USER: '/user',
  FILES: '/files',
} as const;

// 完整的服务端点配置
export const SERVICE_ENDPOINTS = {
  get API_GATEWAY() { return `${SERVICE_URLS.API_GATEWAY}${API_PATHS.V1}`; },
  get AI_STRATEGY() { return `${SERVICE_URLS.AI_STRATEGY}${API_PATHS.V1}`; },
  get DATA_SERVICE() { return `${SERVICE_URLS.DATA_SERVICE}${API_PATHS.V1}`; },
  get USER_SERVICE() { return `${SERVICE_URLS.USER_SERVICE}${API_PATHS.V1}`; },
  get QLIB_SERVICE() { return `${SERVICE_URLS.QLIB_SERVICE}${API_PATHS.V1}`; },
  get STOCK_QUERY() { return `${SERVICE_URLS.STOCK_QUERY}${API_PATHS.V1}`; },
  get TRADING() { return `${SERVICE_URLS.TRADING}${API_PATHS.V1}`; },
} as const;

export default {
  PORTS: SERVICE_PORTS,
  URLS: SERVICE_URLS,
  PATHS: API_PATHS,
  ENDPOINTS: SERVICE_ENDPOINTS,
};
