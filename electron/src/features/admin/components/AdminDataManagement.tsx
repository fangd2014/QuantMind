import React, { useEffect, useMemo, useState, useRef, useCallback } from 'react';
import { Alert, Button, Card, Col, Descriptions, Input, Row, Space, Spin, Statistic, Table, Tag, message, Typography, Progress, Divider, Tooltip, Empty, Tabs } from 'antd';
import {
    DatabaseOutlined,
    ReloadOutlined,
    CloudSyncOutlined,
    CheckCircleFilled,
    WarningFilled,
    FileTextOutlined,
    ThunderboltOutlined,
    CompassOutlined,
    LineChartOutlined,
    InfoCircleOutlined,
    CodeOutlined,
    SafetyCertificateOutlined,
    UserOutlined,
    SyncOutlined,
    CloudDownloadOutlined,
    GlobalOutlined,
    StockOutlined,
    FundOutlined,
    CloudServerOutlined,
    BlockOutlined,
} from '@ant-design/icons';
import dayjs from 'dayjs';
import { adminService } from '../services/adminService';
import { dataPlatformService } from '../services/dataPlatformService';
import { AdminQuantDBPanel } from './AdminQuantDBPanel';
import { AdminQuantMarketPanel } from './AdminQuantMarketPanel';
import {
    AdminFeatureSnapshotsOlderSample,
    AdminFeatureSnapshotsInvalidSample,
    AdminDataStatusResult,
} from '../types';
import { MARKET_CONFIG } from './data-management/constants';
import { MultiMarketSection } from './data-management/MultiMarketSection';
import { AdminRightColumn } from './data-management/AdminRightColumn';

const { Title, Text, Paragraph } = Typography;

const getRequestErrorMessage = (err: any): string => {
    const detail = err?.response?.data?.detail;
    if (typeof detail === 'string' && detail.trim()) return detail;
    if (Array.isArray(detail) && detail.length > 0) {
        const first = detail[0];
        return first?.msg || JSON.stringify(first);
    }
    return err?.message || '未知错误';
};

export const AdminDataManagement: React.FC = () => {
    const [loading, setLoading] = useState(false);
    const [data, setData] = useState<AdminDataStatusResult | null>(null);
    const [dailySyncLoading, setDailySyncLoading] = useState(false);
    const [syncStatus, setSyncStatus] = useState<any>(null);
    const [syncStatusLoading, setSyncStatusLoading] = useState(false);

    // Alpha Agent 市场数据状态（提前声明，供 loadDataStatus 引用）
    const [marketsData, setMarketsData] = useState<any[]>([]);
    const [marketsLoading, setMarketsLoading] = useState(false);
    const [selectedMarket, setSelectedMarket] = useState<string>('a_share');
    const [marketSyncing, setMarketSyncing] = useState<string | null>(null);

    const loadDataStatus = async (refresh = false, market?: string) => {
        setLoading(true);
        try {
            const resp = await adminService.getDataStatus(refresh, market || selectedMarket);
            setData(resp);
            if (refresh) {
                message.success(resp.message || '后台扫描任务已启动，请稍后刷新查看最新状态');
            }
        } catch (err: unknown) {
            const msg = err instanceof Error ? err.message : '未知错误';
            message.error(`数据状态同步失败: ${msg}`);
        } finally {
            setLoading(false);
        }
    };

    const initialRefreshRef = useRef(false);

    const loadSyncStatus = useCallback(async () => {
        setSyncStatusLoading(true);
        try {
            const resp = await adminService.getSyncStatus();
            setSyncStatus(resp?.data || resp);
        } catch {
            // silent
        } finally {
            setSyncStatusLoading(false);
        }
    }, []);

    useEffect(() => {
        if (!initialRefreshRef.current) {
            loadDataStatus(true, selectedMarket);
            loadSyncStatus();
            initialRefreshRef.current = true;
        }
    }, []);

    // 切换市场时重新加载数据状态
    useEffect(() => {
        if (initialRefreshRef.current) {
            loadDataStatus(false, selectedMarket);
        }
    }, [selectedMarket]);

    const qlib = data?.qlib_data;
    const snapshots = data?.feature_snapshots;
    const checkedAt = data?.checked_at ? dayjs(data.checked_at).format('HH:mm:ss') : '—';
    const olderSamples = snapshots?.topn_samples?.older_samples || [];
    const invalidSamples = snapshots?.topn_samples?.invalid_samples || [];
    const sampleSize = snapshots?.topn_samples?.sample_size || 20;

    const coverageRate = useMemo(() => {
        const c = snapshots?.latest_date_coverage;
        if (!c) return 0;
        const total = c.at_target_count + c.older_count + c.invalid_count;
        if (total <= 0) return 0;
        return Math.round((c.at_target_count / total) * 10000) / 100;
    }, [snapshots]);

    const olderColumns = [
        {
            title: '股票代码',
            dataIndex: 'symbol',
            key: 'symbol',
            width: 100,
            render: (v: string) => <span className="font-mono font-black text-indigo-600">{v}</span>,
        },
        {
            title: '最新日期',
            dataIndex: 'last_date',
            key: 'last_date',
            width: 120,
            render: (v: string) => <Text className="font-mono text-slate-500">{v}</Text>
        },
        {
            title: '滞后天数',
            dataIndex: 'lag_days',
            key: 'lag_days',
            width: 100,
            align: 'right' as const,
            render: (v: number) => (
                <Tag color={v > 60 ? '#f43f5e' : v > 10 ? '#f59e0b' : '#10b981'} className="m-0 border-none font-bold rounded-lg px-2">
                    {v}天
                </Tag>
            ),
        },
    ];

    const invalidColumns = [
        {
            title: '股票代码',
            dataIndex: 'symbol',
            key: 'symbol',
            width: 100,
            render: (v: string) => <span className="font-mono font-black text-rose-600">{v}</span>,
        },
        {
            title: '原因',
            dataIndex: 'reason',
            key: 'reason',
            render: (v: string) => <Tag color="error" className="m-0 border-none rounded-md px-2 text-[11px] font-bold uppercase tracking-tight">{v}</Tag>,
        },
        {
            title: '文件路径',
            dataIndex: 'file',
            key: 'file',
            ellipsis: true as const,
            render: (v?: string) => <Text className="text-slate-400 font-mono text-[10px] italic">{v || '—'}</Text>,
        },
    ];

    const handleUpdateFeatureParquet = async (rebuild = false) => {
        setParquetLoading(true);
        setParquetResult(null);
        try {
            const resp = await adminService.updateFeatureParquet(rebuild);
            setParquetResult(resp);
            if (resp.success) {
                message.success(rebuild ? '特征快照已全量重建' : '特征快照已更新');
                await loadDataStatus(false, selectedMarket);
            } else {
                message.error('特征更新失败，请查看执行日志');
            }
        } catch (err: unknown) {
            const msg = err instanceof Error ? err.message : '未知错误';
            message.error(`更新失败: ${msg}`);
        } finally {
            setParquetLoading(false);
        }
    };

    const handleUpdateMarketFeatures = async (market: string, rebuild = false) => {
        setParquetLoading(true);
        setParquetResult(null);
        try {
            const resp = await adminService.updateMarketFeatures(market, rebuild);
            setParquetResult(resp);
            if (resp.success) {
                message.success(rebuild ? `${market} 特征已全量重建` : `${market} 特征已更新`);
                await loadDataStatus(false, market);
            } else {
                message.error('特征更新失败，请查看执行日志');
            }
        } catch (err: unknown) {
            const msg = err instanceof Error ? err.message : '未知错误';
            message.error(`更新失败: ${msg}`);
        } finally {
            setParquetLoading(false);
        }
    };

    const handleSyncFundamentals = async (market = 'ALL') => {
        setFundamentalsLoading(true);
        setFundamentalsResult(null);
        try {
            const resp = await adminService.syncFundamentals(market);
            setFundamentalsResult(resp);
            if (resp?.success) {
                message.success(`基本面数据同步完成 (${market})`);
            } else {
                message.error('基本面数据同步失败');
            }
        } catch (err: unknown) {
            const msg = err instanceof Error ? err.message : '未知错误';
            message.error(`同步失败: ${msg}`);
        } finally {
            setFundamentalsLoading(false);
        }
    };

    const [syncTaskId, setSyncTaskId] = useState<string | null>(null);
    const [syncTaskProgress, setSyncTaskProgress] = useState<string>('');
    const [syncStepProgress, setSyncStepProgress] = useState<{ step: string; detail: string; pct: number; current: number; total: number } | null>(null);
    const [parquetLoading, setParquetLoading] = useState(false);
    const [parquetResult, setParquetResult] = useState<any>(null);
    const [fundamentalsLoading, setFundamentalsLoading] = useState(false);
    const [fundamentalsResult, setFundamentalsResult] = useState<any>(null);

    // 当前选中的市场数据
    const currentMarket = marketsData.find(x => x.market_id === selectedMarket);
    const currentMarketCfg = MARKET_CONFIG[selectedMarket] || { label: 'A股', icon: <StockOutlined />, color: '#ef4444', gradient: 'from-red-500 to-orange-500' };

    const loadMarketsData = useCallback(async () => {
        setMarketsLoading(true);
        try {
            const resp = await adminService.getAlphaAgentMarkets();
            if (resp?.success && resp.data?.markets) {
                setMarketsData(resp.data.markets);
            }
        } catch {
            // silent
        } finally {
            setMarketsLoading(false);
        }
    }, []);

    const handleSyncMarket = async (marketId: string, force = false) => {
        setMarketSyncing(marketId);
        try {
            const resp = await adminService.syncAlphaAgentMarket(marketId, force);
            if (resp?.success) {
                const d = resp.data;
                if (d.status === 'already_ready') {
                    message.info(d.message || `${marketId} 数据已就绪`);
                } else if (d.status === 'completed') {
                    message.success(d.message || `${marketId} 数据同步完成`);
                } else if (d.status === 'submitted' && d.task_id) {
                    message.info(d.message || `${marketId} 数据同步任务已提交`);
                    let completed = false;
                    for (let attempt = 0; attempt < 600; attempt += 1) {
                        await new Promise((resolve) => setTimeout(resolve, 3000));
                        const statusResp = await adminService.getDailySyncTaskStatus(d.task_id);
                        const task = statusResp?.data;
                        if (task?.status === 'SUCCESS') {
                            message.success(`${marketId} 数据同步完成`);
                            completed = true;
                            break;
                        }
                        if (task?.status === 'FAILURE') {
                            throw new Error(task.error || '后台同步任务执行失败');
                        }
                    }
                    if (!completed) {
                        throw new Error('同步任务等待超时，请稍后刷新状态');
                    }
                } else if (d.status === 'skipped') {
                    message.warning(d.message || `${marketId} 已跳过`);
                }
                await loadMarketsData();
            } else {
                message.error(resp?.data?.message || '同步失败');
            }
        } catch (err: any) {
            message.error(`同步失败: ${getRequestErrorMessage(err)}`);
        } finally {
            setMarketSyncing(null);
        }
    };

    useEffect(() => {
        loadMarketsData();
    }, [loadMarketsData]);

    const handleDailySync = async () => {
        setDailySyncLoading(true);
        setSyncTaskProgress('提交任务...');
        setSyncStepProgress(null);
        try {
            const resp = await adminService.triggerDailySync({ calibrate: true });
            if (resp?.success && resp.data?.task_id) {
                const taskId = resp.data.task_id;
                setSyncTaskId(taskId);
                setSyncTaskProgress('任务已提交，等待执行...');
                message.info(`同步任务已提交 (${taskId.slice(0, 8)}...)，后台执行中`);

                // 轮询任务状态 + 步骤进度
                const pollInterval = setInterval(async () => {
                    try {
                        // 同时查 Celery 任务状态和步骤进度
                        const [statusResp, progressResp] = await Promise.all([
                            adminService.getDailySyncTaskStatus(taskId),
                            adminService.getSyncProgress(),
                        ]);

                        // 更新步骤进度
                        const prog = progressResp?.data;
                        if (prog && prog.step !== 'idle') {
                            setSyncStepProgress(prog);
                        }

                        const d = statusResp?.data;
                        if (!d) return;

                        if (d.status === 'SUCCESS') {
                            clearInterval(pollInterval);
                            const r = d.result || {};
                            if (r.status === 'skipped') {
                                message.warning(r.reason || '已有同步任务在运行');
                            } else {
                                message.success(
                                    `同步完成: investment_data=${r.investment_data_synced || 0}, baostock=${r.baostock_synced || 0}, akshare=${r.akshare_synced || 0}, eltdx=${r.eltdx_synced || 0}`
                                );
                            }
                            setDailySyncLoading(false);
                            setSyncTaskId(null);
                            setSyncTaskProgress('');
                            setSyncStepProgress(null);
                            await loadSyncStatus();
                            await loadDataStatus(false, selectedMarket);
                        } else if (d.status === 'FAILURE') {
                            clearInterval(pollInterval);
                            const errMsg = d.error && d.error !== `engine.tasks.daily_data_sync`
                                ? d.error
                                : '任务执行异常，请查看后端日志';
                            message.error(`同步失败: ${errMsg}`);
                            setDailySyncLoading(false);
                            setSyncTaskId(null);
                            setSyncTaskProgress('');
                            setSyncStepProgress(null);
                        } else {
                            // PENDING / STARTED
                            setSyncTaskProgress(d.status === 'STARTED' ? '同步执行中...' : '等待队列...');
                        }
                    } catch {
                        // polling error, continue
                    }
                }, 3000);

                // 超时保护: 30 分钟后停止轮询
                setTimeout(() => {
                    clearInterval(pollInterval);
                    if (dailySyncLoading) {
                        message.warning('同步任务超时，请手动检查状态');
                        setDailySyncLoading(false);
                        setSyncTaskId(null);
                        setSyncTaskProgress('');
                        setSyncStepProgress(null);
                    }
                }, 30 * 60 * 1000);
            } else {
                message.error('任务提交失败');
                setDailySyncLoading(false);
            }
        } catch (err: unknown) {
            const msg = err instanceof Error ? err.message : '未知错误';
            message.error(`提交失败: ${msg}`);
            setDailySyncLoading(false);
            setSyncTaskProgress('');
            setSyncStepProgress(null);
        }
    };

    return (
        <div className="pb-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
            <Tabs
                defaultActiveKey="overview"
                items={[
                    {
                        key: 'overview',
                        label: <span className="font-bold">数据概览</span>,
                        children: (
                            <div className="space-y-10">
                                {/* Header Section */}
                                <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-6">
                                    <div>
                                        <Title level={1} className="!m-0 !font-black !text-4xl !tracking-tighter !text-slate-900 uppercase">
                                            数据管理
                                        </Title>
                                        <div className="flex items-center mt-2 space-x-3">
                                            <Tag className="rounded-full bg-slate-100 border-none text-slate-500 font-bold px-3">
                                                节点: QUANT-OSS-01
                                            </Tag>
                                            <Text className="text-slate-400 font-medium text-sm flex items-center">
                                                <InfoCircleOutlined className="mr-1.5" />
                                                最后扫描时间: <span className="text-indigo-500 font-bold ml-1">{checkedAt}</span>
                                            </Text>
                                        </div>
                                    </div>
                                    <Space size="middle">
                                        <Button
                                            type="primary"
                                            icon={<ThunderboltOutlined />}
                                            className="rounded-2xl h-11 px-8 bg-indigo-600 border-none font-bold shadow-lg shadow-indigo-100"
                                            loading={loading}
                                            onClick={() => loadDataStatus(true, selectedMarket)}
                                        >
                                            强制深度扫描
                    </Button>
                    <Button
                        icon={<ReloadOutlined />}
                        className="rounded-2xl h-11 px-8 border-slate-200 text-slate-600 font-bold hover:bg-slate-50 transition-all"
                        loading={loading}
                        onClick={() => loadDataStatus(false, selectedMarket)}
                    >
                        刷新
                    </Button>
                </Space>
            </div>

            {/* Alpha Agent 市场数据管理 */}
            <MultiMarketSection
                marketsData={marketsData}
                selectedMarket={selectedMarket}
                marketsLoading={marketsLoading}
                marketSyncing={marketSyncing}
                onSelectMarket={setSelectedMarket}
                onReloadMarkets={loadMarketsData}
                onSyncMarket={handleSyncMarket}
            />

            {/* Quick Stats Grid — 根据选中市场切换 */}
            <Row gutter={[24, 24]}>
                {selectedMarket === 'a_share' ? (
                    <>
                        <Col xs={24} sm={12} lg={6}>
                            <Card className="rounded-[2rem] border-none shadow-xl shadow-slate-200/40 bg-white group overflow-hidden">
                                <div className="absolute top-0 right-0 w-24 h-24 bg-blue-500 opacity-[0.03] rounded-bl-[4rem]" />
                                <Statistic
                                    title={<span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Qlib 日历最后日期</span>}
                                    value={qlib?.calendar_last_date || '—'}
                                    valueStyle={{ fontWeight: 900, letterSpacing: '-0.02em', color: '#1e293b' }}
                                    prefix={<CompassOutlined className="text-blue-500 mr-2" />}
                                />
                            </Card>
                        </Col>
                        <Col xs={24} sm={12} lg={6}>
                            <Card className="rounded-[2rem] border-none shadow-xl shadow-slate-200/40 bg-white group overflow-hidden">
                                <div className="absolute top-0 right-0 w-24 h-24 bg-indigo-500 opacity-[0.03] rounded-bl-[4rem]" />
                                <Statistic
                                    title={<span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">快照最新日期</span>}
                                    value={snapshots?.max_date || '—'}
                                    valueStyle={{ fontWeight: 900, letterSpacing: '-0.02em', color: '#1e293b' }}
                                    prefix={<LineChartOutlined className="text-indigo-500 mr-2" />}
                                />
                            </Card>
                        </Col>
                        <Col xs={24} sm={12} lg={6}>
                            <Card className="rounded-[2rem] border-none shadow-xl shadow-slate-200/40 bg-white group overflow-hidden">
                                <div className="absolute top-0 right-0 w-24 h-24 bg-emerald-500 opacity-[0.03] rounded-bl-[4rem]" />
                                <Statistic
                                    title={<span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Parquet 文件总数</span>}
                                    value={snapshots?.file_count ?? 0}
                                    suffix="个"
                                    valueStyle={{ fontWeight: 900, letterSpacing: '-0.02em', color: '#1e293b' }}
                                    prefix={<DatabaseOutlined className="text-emerald-500 mr-2" />}
                                />
                            </Card>
                        </Col>
                        <Col xs={24} sm={12} lg={6}>
                            <Card className="rounded-[2rem] border-none shadow-xl shadow-slate-200/40 bg-white group overflow-hidden">
                                <div className="flex flex-col">
                                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-2">覆盖率</span>
                                    <div className="flex items-center space-x-4">
                                        <Progress
                                            type="circle"
                                            percent={coverageRate}
                                            size={48}
                                            strokeWidth={12}
                                            strokeColor={{ '0%': '#6366f1', '100%': '#10b981' }}
                                            format={() => <span className="text-[10px] font-black text-slate-700">{Math.round(coverageRate)}%</span>}
                                        />
                                        <div>
                                            <div className="text-2xl font-black text-slate-800 tracking-tight">{coverageRate}%</div>
                                            <div className="text-[10px] font-bold text-emerald-500">良好</div>
                                        </div>
                                    </div>
                                </div>
                            </Card>
                        </Col>
                    </>
                ) : (
                    <>
                        <Col xs={24} sm={12} lg={6}>
                            <Card className="rounded-[2rem] border-none shadow-xl shadow-slate-200/40 bg-white group overflow-hidden">
                                <div className="absolute top-0 right-0 w-24 h-24 bg-blue-500 opacity-[0.03] rounded-bl-[4rem]" />
                                <Statistic
                                    title={<span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">标的数量</span>}
                                    value={currentMarket?.h5_info?.symbols ?? '—'}
                                    suffix="只"
                                    valueStyle={{ fontWeight: 900, letterSpacing: '-0.02em', color: '#1e293b' }}
                                    prefix={<StockOutlined className="text-blue-500 mr-2" />}
                                />
                            </Card>
                        </Col>
                        <Col xs={24} sm={12} lg={6}>
                            <Card className="rounded-[2rem] border-none shadow-xl shadow-slate-200/40 bg-white group overflow-hidden">
                                <div className="absolute top-0 right-0 w-24 h-24 bg-indigo-500 opacity-[0.03] rounded-bl-[4rem]" />
                                <Statistic
                                    title={<span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">数据行数</span>}
                                    value={currentMarket?.h5_info?.rows?.toLocaleString() || '—'}
                                    valueStyle={{ fontWeight: 900, letterSpacing: '-0.02em', color: '#1e293b' }}
                                    prefix={<DatabaseOutlined className="text-indigo-500 mr-2" />}
                                />
                            </Card>
                        </Col>
                        <Col xs={24} sm={12} lg={6}>
                            <Card className="rounded-[2rem] border-none shadow-xl shadow-slate-200/40 bg-white group overflow-hidden">
                                <div className="absolute top-0 right-0 w-24 h-24 bg-emerald-500 opacity-[0.03] rounded-bl-[4rem]" />
                                <Statistic
                                    title={<span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">时间范围</span>}
                                    value={currentMarket?.h5_info ? `${currentMarket.h5_info.start_date} ~ ${currentMarket.h5_info.end_date}` : '—'}
                                    valueStyle={{ fontWeight: 900, fontSize: 14, letterSpacing: '-0.02em', color: '#1e293b' }}
                                    prefix={<CompassOutlined className="text-emerald-500 mr-2" />}
                                />
                            </Card>
                        </Col>
                        <Col xs={24} sm={12} lg={6}>
                            <Card className="rounded-[2rem] border-none shadow-xl shadow-slate-200/40 bg-white group overflow-hidden">
                                <div className="absolute top-0 right-0 w-24 h-24 bg-amber-500 opacity-[0.03] rounded-bl-[4rem]" />
                                <Statistic
                                    title={<span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">数据状态</span>}
                                    value={currentMarket?.data_ready ? '已就绪' : '未就绪'}
                                    valueStyle={{ fontWeight: 900, letterSpacing: '-0.02em', color: currentMarket?.data_ready ? '#10b981' : '#f59e0b' }}
                                    prefix={currentMarket?.data_ready ? <CheckCircleFilled className="text-emerald-500 mr-2" /> : <WarningFilled className="text-amber-500 mr-2" />}
                                />
                            </Card>
                        </Col>
                    </>
                )}
            </Row>

            {/* Main Content Area */}
            <Row gutter={[32, 32]}>
                <Col span={24} lg={15} className="space-y-8">
                    {/* Qlib Section — 根据选中市场切换 */}
                    <Card
                        title={
                            <div className="flex items-center space-x-3 py-1">
                                <div className="w-8 h-8 rounded-lg bg-blue-50 flex items-center justify-center text-blue-600">
                                    <DatabaseOutlined />
                                </div>
                                <span className="font-black text-slate-800 tracking-tight text-lg uppercase">
                                    Qlib 基础设施详情 <span className="text-indigo-400 text-sm ml-2">{currentMarketCfg.label}</span>
                                </span>
                            </div>
                        }
                        className="rounded-[2.5rem] border-none shadow-2xl shadow-slate-200/30"
                        styles={{ body: { padding: '32px' } }}
                    >
                        {selectedMarket === 'a_share' ? (
                            !qlib?.exists ? (
                                <Alert
                                    type="error"
                                    showIcon
                                    message={<span className="font-bold">Qlib 目录不存在</span>}
                                    description={<span className="text-xs italic opacity-70">{qlib?.qlib_dir || '路径未定义'}</span>}
                                    className="rounded-2xl"
                                />
                            ) : (
                                <div className="grid grid-cols-2 md:grid-cols-3 gap-y-8 gap-x-12">
                                    {[
                                        { label: 'Qlib 路径', value: qlib.qlib_dir, span: 3, full: true },
                                        { label: '日历总天数', value: qlib.calendar_total_days },
                                        { label: '日历区间', value: `${qlib.calendar_start_date} → ${qlib.calendar_last_date}`, span: 2 },
                                        { label: '标的总数', value: qlib.instruments?.total, highlight: true },
                                        { label: '特征目录数', value: qlib.feature_dirs_total },
                                        { label: '交易所分布', value: `SH: ${qlib.instruments?.sh} | SZ: ${qlib.instruments?.sz} | BJ: ${qlib.instruments?.bj}`, span: 3, italic: true }
                                    ].map((item, i) => (
                                        <div key={i} className={`flex flex-col space-y-1 ${item.span === 3 ? 'col-span-full' : item.span === 2 ? 'col-span-2' : ''}`}>
                                            <Text className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">{item.label}</Text>
                                            <Text className={`text-slate-800 ${item.full ? 'font-mono text-xs break-all' : 'font-black text-lg'} ${item.highlight ? 'text-indigo-600' : ''} ${item.italic ? 'italic text-slate-500' : ''}`}>
                                                {item.value ?? '—'}
                                            </Text>
                                        </div>
                                    ))}
                                </div>
                            )
                        ) : (
                            /* 其他市场: 显示该市场的 Qlib 详情 */
                            (() => {
                                const h5 = currentMarket?.h5_info;
                                const qlibInfo = currentMarket?.qlib_info;
                                // Qlib 路径优先使用后端解析值，回退到本地兜底
                                const qlibPaths: Record<string, string> = {
                                    crypto: '/app/db/qlib_data/crypto_data',
                                    hong_kong: '/app/db/qlib_data/hk_data',
                                    us_stock: '/app/db/qlib_data/us_data',
                                    futures: '/app/db/qlib_data/futures_data',
                                };
                                const qlibDir = qlibInfo?.qlib_dir || qlibPaths[selectedMarket] || '—';
                                if (!h5 && !currentMarket?.data_ready) {
                                    return (
                                        <div className="text-center py-8">
                                            <WarningFilled className="text-amber-400 text-3xl mb-3" />
                                            <div className="text-slate-500 font-bold">数据未下载</div>
                                            <div className="text-xs text-slate-400 mt-1">请先点击上方「开始同步」下载数据</div>
                                        </div>
                                    );
                                }
                                return (
                                    <div className="grid grid-cols-2 md:grid-cols-3 gap-y-8 gap-x-12">
                                        {[
                                            { label: 'Qlib 路径', value: qlibDir, span: 3, full: true },
                                            { label: '日历文件', value: qlibInfo?.calendar_files?.join(', ') || '—' },
                                            { label: '数据区间', value: h5 ? `${h5.start_date} → ${h5.end_date}` : '—', span: 2 },
                                            { label: '标的总数', value: h5?.symbols, highlight: true },
                                            { label: '特征目录数', value: qlibInfo?.feature_dirs ?? '—' },
                                            { label: '数据行数', value: h5?.rows?.toLocaleString() || '—', span: 3, italic: true }
                                        ].map((item, i) => (
                                            <div key={i} className={`flex flex-col space-y-1 ${item.span === 3 ? 'col-span-full' : item.span === 2 ? 'col-span-2' : ''}`}>
                                                <Text className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">{item.label}</Text>
                                                <Text className={`text-slate-800 ${item.full ? 'font-mono text-xs break-all' : 'font-black text-lg'} ${item.highlight ? 'text-indigo-600' : ''} ${item.italic ? 'italic text-slate-500' : ''}`}>
                                                    {item.value ?? '—'}
                                                </Text>
                                            </div>
                                        ))}
                                    </div>
                                );
                            })()
                        )}
                    </Card>

                    {/* Snapshots Section */}
                    <Card
                        title={
                            <div className="flex items-center space-x-3 py-1">
                                <div className="w-8 h-8 rounded-lg bg-indigo-50 flex items-center justify-center text-indigo-600">
                                    <FileTextOutlined />
                                </div>
                                <span className="font-black text-slate-800 tracking-tight text-lg uppercase">
                                    特征快照分析 <span className="text-indigo-400 text-sm ml-2">{currentMarketCfg.label}</span>
                                </span>
                            </div>
                        }
                        className="rounded-[2.5rem] border-none shadow-2xl shadow-slate-200/30"
                        styles={{ body: { padding: '32px' } }}
                    >
                        {!snapshots?.exists ? (
                            <Empty description="暂无快照数据" image={Empty.PRESENTED_IMAGE_SIMPLE} />
                        ) : (
                            <div className="space-y-10">
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-y-8">
                                    {(() => {
                                        const integrityStatus = snapshots.integrity_status as ('ok' | 'warning' | 'error' | undefined);
                                        const integrityIssues = (snapshots.integrity_issues as string[] | undefined) || [];
                                        const hasError = snapshots.error || integrityStatus === 'error';
                                        const integrityLabel = hasError
                                            ? '异常'
                                            : integrityStatus === 'warning'
                                                ? `警告 (${integrityIssues.length})`
                                                : '正常';
                                        const integrityColor = hasError
                                            ? 'text-rose-500'
                                            : integrityStatus === 'warning'
                                                ? 'text-amber-500'
                                                : 'text-emerald-500';
                                        return [
                                            { label: '总行数', value: snapshots.total_rows?.toLocaleString(), color: 'text-indigo-600' },
                                            { label: '扫描成功', value: snapshots.scanned_files, color: 'text-emerald-500' },
                                            { label: '扫描失败', value: snapshots.failed_files, color: 'text-rose-500' },
                                            {
                                                label: '数据完整性',
                                                value: integrityLabel,
                                                color: integrityColor,
                                                title: integrityIssues.length ? integrityIssues.join('；') : undefined,
                                            },
                                        ];
                                    })().map((item, i) => (
                                        <div key={i} className="flex flex-col" title={(item as any).title}>
                                            <Text className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">{item.label}</Text>
                                            <Text className={`font-black text-xl tracking-tighter ${item.color}`}>{item.value ?? '—'}</Text>
                                        </div>
                                    ))}
                                </div>

                                {snapshots.suggested_periods && (
                                    <div className="p-6 rounded-3xl bg-slate-50 border border-slate-100">
                                        <div className="flex items-center space-x-2 mb-4">
                                            <CompassOutlined className="text-slate-400" />
                                            <span className="text-xs font-black text-slate-600 uppercase tracking-widest">推荐训练区间（全局）</span>
                                        </div>
                                        <div className="flex flex-wrap gap-4">
                                            {Object.entries(snapshots.suggested_periods).map(([key, period]: [string, any]) => (
                                                <div key={key} className="flex-1 min-w-[140px] p-4 bg-white rounded-2xl shadow-sm border border-slate-100">
                                                    <Text className="text-[10px] font-bold text-slate-400 uppercase block mb-1">{key} 集</Text>
                                                    <Text className="font-mono text-[11px] font-black text-slate-700">{period[0]} ~ {period[1]}</Text>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {snapshots?.metadata_files && snapshots.metadata_files.length > 0 && (
                                    <div className="space-y-4">
                                        <div className="flex items-center justify-between px-1">
                                            <Text className="text-[10px] font-black text-slate-400 uppercase tracking-widest">年度快照详情</Text>
                                            <Tag className="m-0 border-none bg-indigo-50 text-indigo-400 text-[9px] font-bold rounded-md">Total: {snapshots.metadata_files.length}</Tag>
                                        </div>
                                        <div className="space-y-3 max-h-[420px] overflow-y-auto pr-2 custom-scrollbar">
                                            {snapshots.metadata_files.map((m: any, idx: number) => (
                                                <div key={idx} className="group bg-white rounded-2xl p-4 border border-slate-100 flex items-center justify-between hover:border-indigo-200 hover:shadow-md transition-all duration-300">
                                                    <div className="flex items-center space-x-4">
                                                        <div className="w-10 h-10 rounded-xl bg-slate-50 flex items-center justify-center group-hover:bg-indigo-50 transition-colors">
                                                            <span className="text-slate-400 font-black text-xs group-hover:text-indigo-500">
                                                                {m.year ?? m.filename?.replace('model_features_', '').replace('.parquet', '').slice(0, 8)}
                                                            </span>
                                                        </div>
                                                        <div className="space-y-1">
                                                            <div className="text-xs font-black text-slate-700 tracking-tight">
                                                                {m.start_date} <span className="text-slate-300 mx-1">/</span> {m.end_date}
                                                            </div>
                                                            <div className="flex items-center space-x-3 text-[10px] text-slate-400 font-medium">
                                                                <span className="flex items-center"><DatabaseOutlined className="mr-1 text-[9px]" /> {m.row_count.toLocaleString()} 样本</span>
                                                                <span className="flex items-center"><UserOutlined className="mr-1 text-[9px]" /> {m.symbol_count} 标的</span>
                                                            </div>
                                                        </div>
                                                    </div>
                                                    <div className="text-right">
                                                        <div className="text-[10px] font-black text-indigo-500 uppercase tracking-tight">{m.feature_dim} Features</div>
                                                        <div className="text-[8px] text-slate-300 font-mono mt-1 uppercase">{m.filename.split('.').slice(0, 2).join('.')}</div>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}
                    </Card>
                </Col>

                <AdminRightColumn
                    selectedMarket={selectedMarket}
                    currentMarketCfg={currentMarketCfg}
                    currentMarket={currentMarket}
                    dailySyncLoading={dailySyncLoading}
                    parquetLoading={parquetLoading}
                    fundamentalsLoading={fundamentalsLoading}
                    marketSyncing={marketSyncing}
                    syncTaskId={syncTaskId}
                    syncTaskProgress={syncTaskProgress}
                    syncStepProgress={syncStepProgress}
                    parquetResult={parquetResult}
                    fundamentalsResult={fundamentalsResult}
                    syncStatus={syncStatus}
                    syncStatusLoading={syncStatusLoading}
                    snapshots={snapshots}
                    olderSamples={olderSamples}
                    invalidSamples={invalidSamples}
                    sampleSize={sampleSize}
                    olderColumns={olderColumns}
                    invalidColumns={invalidColumns}
                    onReloadSyncStatus={loadSyncStatus}
                    onDailySync={handleDailySync}
                    onUpdateFeatureParquet={handleUpdateFeatureParquet}
                    onUpdateMarketFeatures={handleUpdateMarketFeatures}
                    onSyncFundamentals={handleSyncFundamentals}
                    onSyncMarket={handleSyncMarket}
                />
            </Row>
                            </div>
                        ),
                    },
                    {
                        key: 'quantdb',
                        label: <span className="font-bold flex items-center"><CloudServerOutlined className="mr-1" />QuantDB A股</span>,
                        children: <AdminQuantDBPanel />,
                    },
                    {
                        key: 'quantus',
                        label: <span className="font-bold flex items-center"><FundOutlined className="mr-1" />QuantUS 美股</span>,
                        children: <AdminQuantMarketPanel market="quantus" marketLabel="QuantUS 美股" color="#1677ff" />,
                    },
                    {
                        key: 'quanthk',
                        label: <span className="font-bold flex items-center"><GlobalOutlined className="mr-1" />QuantHK 港股</span>,
                        children: <AdminQuantMarketPanel market="quanthk" marketLabel="QuantHK 港股" color="#722ed1" />,
                    },
                    {
                        key: 'quantbc',
                        label: <span className="font-bold flex items-center"><BlockOutlined className="mr-1" />QuantBC 区块链</span>,
                        children: <AdminQuantMarketPanel market="quantbc" marketLabel="QuantBC 区块链" color="#13c2c2" />,
                    },
                    {
                        key: 'quantfutures',
                        label: <span className="font-bold flex items-center"><StockOutlined className="mr-1" />QuantFutures 期货</span>,
                        children: <AdminQuantMarketPanel market="quantfutures" marketLabel="QuantFutures 期货" color="#fa8c16" />,
                    },
                ]}
            />
        </div>
    );
};
