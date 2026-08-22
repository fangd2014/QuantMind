import React from 'react';
import {
    Button,
    Card,
    Empty,
    Space,
    Tag,
    Typography,
} from 'antd';
import {
    CheckCircleFilled,
    DatabaseOutlined,
    GlobalOutlined,
    ReloadOutlined,
    SyncOutlined,
    WarningFilled,
} from '@ant-design/icons';
import { MARKET_CONFIG } from './constants';

const { Text } = Typography;

interface MultiMarketSectionProps {
    marketsData: any[];
    selectedMarket: string;
    marketsLoading: boolean;
    marketSyncing: string | null;
    onSelectMarket: (marketId: string) => void;
    onReloadMarkets: () => void;
    onSyncMarket: (marketId: string, dataReady: boolean) => void;
}

export const MultiMarketSection: React.FC<MultiMarketSectionProps> = ({
    marketsData,
    selectedMarket,
    marketsLoading,
    marketSyncing,
    onSelectMarket,
    onReloadMarkets,
    onSyncMarket,
}) => {
    return (
        <Card
            className="rounded-[2.5rem] border-none shadow-2xl shadow-slate-200/30"
            styles={{ body: { padding: '32px' } }}
        >
            <div className="flex items-center justify-between mb-6">
                <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                        <GlobalOutlined className="text-white text-xl" />
                    </div>
                    <div>
                        <span className="text-slate-800 font-black text-xl uppercase tracking-tight block">
                            多市场数据
                        </span>
                        <span className="text-[10px] text-slate-400 font-bold uppercase tracking-widest">
                            Alpha Agent 因子挖掘数据源
                        </span>
                    </div>
                </div>
                <Button
                    type="text"
                    size="small"
                    icon={<ReloadOutlined spin={marketsLoading} />}
                    onClick={onReloadMarkets}
                    className="text-slate-400"
                />
            </div>

            {/* 市场标签选择器 */}
            <div className="flex flex-wrap gap-3 mb-6">
                {marketsData.map((m) => {
                    const cfg = MARKET_CONFIG[m.market_id] || {
                        label: m.market_name,
                        icon: <DatabaseOutlined />,
                        color: '#6366f1',
                        gradient: 'from-indigo-500 to-purple-500',
                    };
                    const isActive = selectedMarket === m.market_id;
                    return (
                        <button
                            key={m.market_id}
                            onClick={() => onSelectMarket(m.market_id)}
                            className={`
                                relative flex items-center gap-2.5 px-5 py-3 rounded-2xl font-bold text-sm
                                transition-all duration-300 cursor-pointer border-none outline-none
                                ${isActive
                                    ? `bg-gradient-to-r ${cfg.gradient} text-white shadow-lg scale-[1.02]`
                                    : 'bg-slate-50 text-slate-600 hover:bg-slate-100 hover:scale-[1.01]'
                                }
                            `}
                        >
                            <span className="text-base">{cfg.icon}</span>
                            <span className="tracking-tight">{cfg.label}</span>
                            {m.data_ready ? (
                                <span
                                    className={`w-2 h-2 rounded-full ${isActive ? 'bg-white/80' : 'bg-emerald-400'} animate-pulse`}
                                />
                            ) : (
                                <span
                                    className={`w-2 h-2 rounded-full ${isActive ? 'bg-white/40' : 'bg-slate-300'}`}
                                />
                            )}
                        </button>
                    );
                })}
            </div>

            {/* 选中市场的详情 */}
            <MultiMarketDetail
                marketsData={marketsData}
                selectedMarket={selectedMarket}
                marketSyncing={marketSyncing}
                onSyncMarket={onSyncMarket}
            />
        </Card>
    );
};

interface MultiMarketDetailProps {
    marketsData: any[];
    selectedMarket: string;
    marketSyncing: string | null;
    onSyncMarket: (marketId: string, dataReady: boolean) => void;
}

const MultiMarketDetail: React.FC<MultiMarketDetailProps> = ({
    marketsData,
    selectedMarket,
    marketSyncing,
    onSyncMarket,
}) => {
    const m = marketsData.find((x) => x.market_id === selectedMarket);
    if (!m)
        return (
            <Empty
                description="暂无市场数据"
                image={Empty.PRESENTED_IMAGE_SIMPLE}
            />
        );

    const cfg = MARKET_CONFIG[m.market_id] || {
        label: m.market_name,
        color: '#6366f1',
        gradient: 'from-indigo-500 to-purple-500',
    };
    const h5 = m.h5_info;
    const qlib = m.qlib_info;

    return (
        <div className="space-y-5">
            {/* 状态行 */}
            <div className="flex items-center justify-between p-5 rounded-2xl bg-slate-50 border border-slate-100">
                <div className="flex items-center gap-4">
                    <div
                        className={`w-12 h-12 rounded-2xl bg-gradient-to-br ${cfg.gradient} flex items-center justify-center text-white text-xl shadow-lg`}
                    >
                        {MARKET_CONFIG[m.market_id]?.icon || <DatabaseOutlined />}
                    </div>
                    <div>
                        <div className="flex items-center gap-2">
                            <span className="font-black text-lg text-slate-800">
                                {m.market_name}
                            </span>
                            {m.data_ready ? (
                                <Tag className="m-0 border-none bg-emerald-50 text-emerald-600 font-bold rounded-lg text-[10px]">
                                    <CheckCircleFilled className="mr-1" /> 已就绪
                                </Tag>
                            ) : (
                                <Tag className="m-0 border-none bg-amber-50 text-amber-600 font-bold rounded-lg text-[10px]">
                                    <WarningFilled className="mr-1" /> 未就绪
                                </Tag>
                            )}
                        </div>
                        <span className="text-xs text-slate-400">{m.description}</span>
                    </div>
                </div>
                <Space>
                    <Button
                        type="primary"
                        icon={<SyncOutlined />}
                        loading={marketSyncing === m.market_id}
                        onClick={() => onSyncMarket(m.market_id, m.data_ready)}
                        className={`rounded-xl h-10 px-6 font-bold border-none shadow-md ${
                            m.data_ready
                                ? 'bg-slate-600 hover:bg-slate-700'
                                : `bg-gradient-to-r ${cfg.gradient} hover:opacity-90`
                        }`}
                    >
                        {m.data_ready ? '重新同步' : '开始同步'}
                    </Button>
                </Space>
            </div>

            {/* 数据详情网格 */}
            {h5 ? (
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                    {[
                        { label: '标的数量', value: `${h5.symbols} 只`, color: 'text-indigo-600' },
                        { label: '数据行数', value: h5.rows.toLocaleString(), color: 'text-slate-800' },
                        { label: '起始日期', value: h5.start_date, color: 'text-slate-600' },
                        { label: '截止日期', value: h5.end_date, color: 'text-emerald-600' },
                        { label: '文件大小', value: `${h5.file_size_mb} MB`, color: 'text-amber-600' },
                    ].map((item, i) => (
                        <div
                            key={i}
                            className="p-4 rounded-2xl bg-white border border-slate-100 shadow-sm"
                        >
                            <Text className="text-[9px] font-bold text-slate-400 uppercase tracking-widest block mb-1">
                                {item.label}
                            </Text>
                            <Text className={`font-black text-base tracking-tight ${item.color}`}>
                                {item.value}
                            </Text>
                        </div>
                    ))}
                </div>
            ) : qlib && m.data_ready ? (
                <div className="p-6 rounded-2xl bg-emerald-50 border border-emerald-100 text-center">
                    <CheckCircleFilled className="text-emerald-500 text-lg mr-2" />
                    <Text className="text-emerald-700 font-bold text-sm">
                        Qlib 数据文件已就绪，可用于 Alpha Agent 因子挖掘
                    </Text>
                </div>
            ) : (
                <div className="p-6 rounded-2xl bg-amber-50 border border-amber-100 text-center">
                    <WarningFilled className="text-amber-500 text-lg mr-2" />
                    <Text className="text-amber-700 font-bold text-sm">
                        数据文件不存在，请点击「开始同步」下载数据
                    </Text>
                </div>
            )}

            {/* Qlib 信息 */}
            {qlib && (
                <div className="flex items-center gap-6 px-5 py-3 rounded-xl bg-indigo-50/50 border border-indigo-100">
                    <div className="flex items-center gap-2">
                        <DatabaseOutlined className="text-indigo-400 text-xs" />
                        <Text className="text-[10px] font-bold text-indigo-500 uppercase">
                            Qlib
                        </Text>
                    </div>
                    <Text className="text-xs text-slate-600">
                        日历:{' '}
                        <span className="font-bold">
                            {qlib.calendar_files?.join(', ') || '—'}
                        </span>
                    </Text>
                    {qlib.calendar_last_date && (
                        <Text className="text-xs text-slate-600">
                            最新日期:{' '}
                            <span className="font-bold text-emerald-600">
                                {qlib.calendar_last_date}
                            </span>
                        </Text>
                    )}
                    {qlib.instruments_count > 0 && (
                        <Text className="text-xs text-slate-600">
                            标的数:{' '}
                            <span className="font-bold text-indigo-600">
                                {qlib.instruments_count}
                            </span>
                        </Text>
                    )}
                    <Text className="text-xs text-slate-600">
                        特征目录:{' '}
                        <span className="font-bold text-indigo-600">
                            {qlib.feature_dirs}
                        </span>{' '}
                        个
                    </Text>
                </div>
            )}
        </div>
    );
};
