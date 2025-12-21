import React, { useMemo, useState } from 'react';
import { CanvasModule, SplitDataOutput, DataPreview } from '../types';
import { XCircleIcon, SparklesIcon, ArrowDownTrayIcon } from './icons';
import { GoogleGenAI } from "@google/genai";
import { MarkdownRenderer } from './MarkdownRenderer';
import { SpreadViewModal } from './SpreadViewModal';

type SortConfig = {
    key: string;
    direction: 'ascending' | 'descending';
} | null;

// A component to display descriptive statistics for a dataset
const StatsTable: React.FC<{ title: string; data: DataPreview }> = ({ title, data }) => {
    const allColumns = useMemo(() => data.columns, [data]);
    const rows = useMemo(() => data.rows || [], [data.rows]);

    // fix: Moved statDisplay before useMemo hook to be accessible and defined the correct type for 'results'.
    const statDisplay = [
        { key: 'count', label: 'Count' },
        { key: 'mean', label: 'Mean' },
        { key: 'std', label: 'Std Dev' },
        { key: '50%', label: 'Median' },
        { key: 'min', label: 'Min' },
        { key: 'max', label: 'Max' },
        { key: '25%', label: '25%' },
        { key: '75%', label: '75%' },
        { key: 'mode', label: 'Mode' },
        { key: 'nulls', label: 'Null' },
        { key: 'skewness', label: 'Skew' },
        { key: 'kurtosis', label: 'Kurt' },
    ] as const;

    const stats = useMemo(() => {
        type StatKey = typeof statDisplay[number]['key'];
        const results: Record<string, Partial<Record<StatKey, number | string>>> = {};
        if (rows.length === 0) return results;

        allColumns.forEach(col => {
            const allValues = rows.map(r => r[col.name]);
            const isNull = (v: any) => v === null || v === undefined || v === '';
            const nonNullValues = allValues.filter(v => !isNull(v));

            const nulls = allValues.length - nonNullValues.length;
            const count = allValues.length;
            
            let mode: number | string = 'N/A';
            if (nonNullValues.length > 0) {
                const counts: Record<string, number> = {};
                for(const val of nonNullValues) {
                    const key = String(val);
                    counts[key] = (counts[key] || 0) + 1;
                }
                mode = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
            }

            results[col.name] = { count, nulls, mode };

            if (col.type === 'number') {
                const numericValues = nonNullValues.map(v => Number(v)).filter(v => !isNaN(v));
                if (numericValues.length > 0) {
                    numericValues.sort((a,b) => a - b);
                    const sum = numericValues.reduce((a, b) => a + b, 0);
                    const mean = sum / numericValues.length;
                    const n = numericValues.length;
                    const std = Math.sqrt(numericValues.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b) / n);
                    const skewness = std > 0 ? numericValues.reduce((s, val) => s + Math.pow(val - mean, 3), 0) / (n * Math.pow(std, 3)) : 0;
                    const kurtosis = std > 0 ? numericValues.reduce((s, val) => s + Math.pow(val - mean, 4), 0) / (n * Math.pow(std, 4)) - 3 : 0;
                    const getQuantile = (q: number) => {
                        const pos = (numericValues.length - 1) * q;
                        const base = Math.floor(pos);
                        const rest = pos - base;
                        return numericValues[base + 1] !== undefined ? numericValues[base] + rest * (numericValues[base + 1] - numericValues[base]) : numericValues[base];
                    };

                    const numericMode = Number(mode);

                    results[col.name] = {
                        ...results[col.name],
                        mean,
                        std,
                        min: numericValues[0],
                        '25%': getQuantile(0.25),
                        '50%': getQuantile(0.5),
                        '75%': getQuantile(0.75),
                        max: numericValues[numericValues.length - 1],
                        mode: isNaN(numericMode) ? mode : numericMode,
                        skewness,
                        kurtosis,
                    };
                }
            }
        });
        return results;
    }, [allColumns, rows]);

    return (
        <div>
            <h3 className="text-lg font-semibold mb-2 text-gray-700">{title} ({data.totalRowCount} rows)</h3>
            {allColumns.length === 0 ? (
                <p className="text-sm text-gray-500">No columns to display statistics for.</p>
            ) : (
                <div className="overflow-x-auto border border-gray-200 rounded-lg">
                    <table className="w-full text-sm text-left table-auto">
                        <thead className="bg-gray-50">
                            <tr>
                                <th className="py-1.5 px-3 font-semibold text-gray-600">Metric</th>
                                {allColumns.map(col => (
                                    <th key={col.name} className="py-1.5 px-3 font-semibold text-gray-600 text-right truncate" title={col.name}>{col.name}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {statDisplay.map(({ key, label }) => (
                                <tr key={key} className="border-b border-gray-200 last:border-b-0">
                                    <td className="py-1.5 px-3 font-medium text-gray-500">{label}</td>
                                    {allColumns.map(col => {
                                        const value = stats[col.name]?.[key];
                                        let displayValue = 'N/A';
                                        if (value !== undefined && value !== null && !Number.isNaN(value)) {
                                            if (typeof value === 'number' && !Number.isInteger(value)) {
                                                displayValue = value.toFixed(2);
                                            } else {
                                                displayValue = String(value);
                                            }
                                        }
                                        return (
                                            <td key={`${key}-${col.name}`} className="py-1.5 px-3 font-mono text-right text-gray-700">
                                                {displayValue}
                                            </td>
                                        );
                                    })}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
};

// A component to display data rows in a table
const DataTable: React.FC<{ title: string; data: DataPreview }> = ({ title, data }) => {
    const [sortConfig, setSortConfig] = useState<SortConfig>(null);
    const allColumns = useMemo(() => data.columns, [data]);
    const rows = useMemo(() => data.rows || [], [data.rows]);
    
    // 행과 열 개수
    const rowCount = data.totalRowCount || rows.length;
    const columnCount = allColumns.length;

    const sortedRows = useMemo(() => {
        let sortableItems = [...rows];
        if (sortConfig !== null) {
            sortableItems.sort((a, b) => {
                const valA = a[sortConfig.key];
                const valB = b[sortConfig.key];
                if (valA === null || valA === undefined) return 1;
                if (valB === null || valB === undefined) return -1;
                if (valA < valB) {
                    return sortConfig.direction === 'ascending' ? -1 : 1;
                }
                if (valA > valB) {
                    return sortConfig.direction === 'ascending' ? 1 : -1;
                }
                return 0;
            });
        }
        return sortableItems;
    }, [rows, sortConfig]);

    const requestSort = (key: string) => {
        let direction: 'ascending' | 'descending' = 'ascending';
        if (sortConfig && sortConfig.key === key && sortConfig.direction === 'ascending') {
            direction = 'descending';
        }
        setSortConfig({ key, direction });
    };

    return (
        <div className="flex flex-col h-full">
            {/* 행과 열 개수 표시 */}
            <div className="mb-3 text-sm text-gray-600 flex-shrink-0">
                <span className="font-semibold">Rows: {rowCount.toLocaleString()}</span>
                <span className="mx-2">|</span>
                <span className="font-semibold">Columns: {columnCount}</span>
            </div>
            {allColumns.length === 0 ? (
                <p className="text-sm text-gray-500">No columns to display.</p>
            ) : (
                <div className="flex-grow overflow-x-auto overflow-y-auto border border-gray-200 rounded-lg" style={{ maxHeight: '400px' }}>
                    <table className="w-full text-sm text-left table-auto">
                        <thead className="bg-gray-50 sticky top-0">
                            <tr>
                                {allColumns.map(col => (
                                    <th
                                        key={col.name}
                                        onClick={() => requestSort(col.name)}
                                        className="py-2 px-3 font-semibold text-gray-600 cursor-pointer hover:bg-gray-100 select-none"
                                        title={`Click to sort by ${col.name}`}
                                    >
                                        <div className="flex items-center gap-1">
                                            {col.name}
                                            {sortConfig?.key === col.name && (
                                                <span className="text-xs">
                                                    {sortConfig.direction === 'ascending' ? '↑' : '↓'}
                                                </span>
                                            )}
                                        </div>
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {sortedRows.length === 0 ? (
                                <tr>
                                    <td colSpan={allColumns.length} className="py-4 text-center text-gray-500">
                                        No data rows available.
                                    </td>
                                </tr>
                            ) : (
                                sortedRows.map((row, rowIndex) => (
                                    <tr
                                        key={rowIndex}
                                        className="border-b border-gray-200 hover:bg-gray-50 last:border-b-0"
                                    >
                                        {allColumns.map(col => {
                                            const value = row[col.name];
                                            let displayValue: string;
                                            if (value === null || value === undefined || value === '') {
                                                displayValue = '';
                                            } else if (col.type === 'number') {
                                                const numValue = Number(value);
                                                displayValue = isNaN(numValue) ? String(value) : numValue.toFixed(4);
                                            } else {
                                                displayValue = String(value);
                                            }
                                            return (
                                                <td
                                                    key={`${rowIndex}-${col.name}`}
                                                    className={`py-2 px-3 ${
                                                        col.type === 'number' ? 'font-mono text-right' : 'text-left'
                                                    } text-gray-700`}
                                                >
                                                    {displayValue}
                                                </td>
                                            );
                                        })}
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
};


export const SplitDataPreviewModal: React.FC<{ module: CanvasModule; onClose: () => void; }> = ({ module, onClose }) => {
    const [isInterpreting, setIsInterpreting] = useState(false);
    const [aiInterpretation, setAiInterpretation] = useState<string | null>(null);
    const [showSpreadView, setShowSpreadView] = useState(false);
    const [spreadViewTab, setSpreadViewTab] = useState<'train' | 'test'>('train');
    const [activeTab, setActiveTab] = useState<'train' | 'test'>('train');

    const output = module.outputData as SplitDataOutput;
    if (!output || output.type !== 'SplitDataOutput') return null;

    // Spread View용 데이터 변환
    const spreadViewData = useMemo(() => {
        const currentData = spreadViewTab === 'train' ? output.train : output.test;
        return currentData.rows || [];
    }, [output, spreadViewTab]);

    const spreadViewColumns = useMemo(() => {
        const currentData = spreadViewTab === 'train' ? output.train : output.test;
        return currentData.columns || [];
    }, [output, spreadViewTab]);

    const handleInterpret = async () => {
        setIsInterpreting(true);
        setAiInterpretation(null);
        try {
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });
            const prompt = `
You are an ML educator. Please explain the following concepts in Korean, each in a single, simple sentence. Use Markdown for formatting.

### 데이터 분할 핵심 요약

*   **분할 목적:** 데이터를 학습용과 테스트용으로 나누는 가장 중요한 이유는 무엇입니까?
*   **세트 비교:** 두 데이터 세트의 통계가 유사해 보이는 것이 왜 중요합니까?
*   **다음 단계:** 이 분할된 데이터로 다음에는 무엇을 하게 됩니까?
`;
            const response = await ai.models.generateContent({ model: 'gemini-2.5-flash', contents: prompt });
            setAiInterpretation(response.text);
        } catch (error) {
            console.error("AI interpretation failed:", error);
            setAiInterpretation("결과를 해석하는 동안 오류가 발생했습니다.");
        } finally {
            setIsInterpreting(false);
        }
    };

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={onClose}>
            <div className="bg-white text-gray-900 rounded-lg shadow-xl w-full max-w-7xl max-h-[90vh] flex flex-col" onClick={e => e.stopPropagation()}>
                <header className="flex items-center justify-between p-4 border-b border-gray-200 flex-shrink-0">
                    <h2 className="text-xl font-bold text-gray-800">Data Split Preview: {module.name}</h2>
                    <div className="flex items-center gap-2">
                        <button onClick={onClose} className="text-gray-500 hover:text-gray-800">
                            <XCircleIcon className="w-6 h-6" />
                        </button>
                    </div>
                </header>
                <main className="flex-grow p-4 overflow-hidden flex flex-col">
                     <div className="flex justify-end font-sans mb-4 flex-shrink-0">
                        <button
                            onClick={handleInterpret}
                            disabled={isInterpreting}
                            className="flex items-center gap-2 px-4 py-2 text-sm font-semibold text-white bg-purple-600 rounded-lg hover:bg-purple-700 disabled:bg-purple-400 disabled:cursor-wait transition-colors"
                        >
                            <SparklesIcon className="w-5 h-5" />
                            {isInterpreting ? '분석 중...' : 'AI로 결과 해석하기'}
                        </button>
                    </div>

                    {isInterpreting && <div className="text-center p-4 text-gray-600 flex-shrink-0">AI가 데이터 분할의 의미를 분석하고 있습니다...</div>}
                    {aiInterpretation && (
                         <div className="bg-purple-50 p-4 rounded-lg border border-purple-200 mb-4 flex-shrink-0">
                            <h3 className="text-lg font-bold text-purple-800 mb-2 font-sans flex items-center gap-2">
                                <SparklesIcon className="w-5 h-5"/>
                                AI 분석 요약
                            </h3>
                            <MarkdownRenderer text={aiInterpretation} />
                        </div>
                    )}
                    
                    {/* 탭 구조 */}
                    <div className="flex-shrink-0 border-b border-gray-200 mb-4">
                        <nav className="-mb-px flex space-x-8" aria-label="Tabs">
                            <button
                                onClick={() => setActiveTab('train')}
                                className={`${
                                    activeTab === 'train'
                                        ? 'border-indigo-500 text-indigo-600'
                                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
                            >
                                Train Data
                            </button>
                            <button
                                onClick={() => setActiveTab('test')}
                                className={`${
                                    activeTab === 'test'
                                        ? 'border-indigo-500 text-indigo-600'
                                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
                            >
                                Test Data
                            </button>
                        </nav>
                    </div>
                    
                    {/* 탭 내용 */}
                    <div className="flex-grow overflow-hidden flex flex-col">
                        {activeTab === 'train' && (
                            <>
                                <div className="flex items-center justify-end gap-2 mb-3 flex-shrink-0">
                                    <button
                                        onClick={() => {
                                            setSpreadViewTab('train');
                                            setShowSpreadView(true);
                                        }}
                                        className="px-3 py-1.5 text-sm bg-green-600 text-white rounded-md hover:bg-green-700 flex items-center gap-1"
                                    >
                                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
                                        </svg>
                                        Spread View
                                    </button>
                                    <button
                                        onClick={() => {
                                            const currentData = output.train;
                                            if (!currentData || !currentData.rows || currentData.rows.length === 0) return;
                                            const csvContent = [
                                                currentData.columns.map(c => c.name).join(','),
                                                ...currentData.rows.map(row => 
                                                    currentData.columns.map(col => {
                                                        const val = row[col.name];
                                                        if (val === null || val === undefined) return '';
                                                        const str = String(val);
                                                        return str.includes(',') || str.includes('"') || str.includes('\n') 
                                                            ? `"${str.replace(/"/g, '""')}"` 
                                                            : str;
                                                    }).join(',')
                                                )
                                            ].join('\n');
                                            const bom = '\uFEFF';
                                            const blob = new Blob([bom + csvContent], { type: 'text/csv;charset=utf-8;' });
                                            const link = document.createElement('a');
                                            link.href = URL.createObjectURL(blob);
                                            link.download = `${module.name}_train.csv`;
                                            link.click();
                                        }}
                                        className="text-gray-500 hover:text-gray-800 p-1 rounded hover:bg-gray-100"
                                        title="Download CSV"
                                    >
                                        <ArrowDownTrayIcon className="w-6 h-6" />
                                    </button>
                                </div>
                                <div className="flex-grow overflow-hidden">
                                    <DataTable title="Train Data" data={output.train} />
                                </div>
                            </>
                        )}
                        {activeTab === 'test' && (
                            <>
                                <div className="flex items-center justify-end gap-2 mb-3 flex-shrink-0">
                                    <button
                                        onClick={() => {
                                            setSpreadViewTab('test');
                                            setShowSpreadView(true);
                                        }}
                                        className="px-3 py-1.5 text-sm bg-green-600 text-white rounded-md hover:bg-green-700 flex items-center gap-1"
                                    >
                                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
                                        </svg>
                                        Spread View
                                    </button>
                                    <button
                                        onClick={() => {
                                            const currentData = output.test;
                                            if (!currentData || !currentData.rows || currentData.rows.length === 0) return;
                                            const csvContent = [
                                                currentData.columns.map(c => c.name).join(','),
                                                ...currentData.rows.map(row => 
                                                    currentData.columns.map(col => {
                                                        const val = row[col.name];
                                                        if (val === null || val === undefined) return '';
                                                        const str = String(val);
                                                        return str.includes(',') || str.includes('"') || str.includes('\n') 
                                                            ? `"${str.replace(/"/g, '""')}"` 
                                                            : str;
                                                    }).join(',')
                                                )
                                            ].join('\n');
                                            const bom = '\uFEFF';
                                            const blob = new Blob([bom + csvContent], { type: 'text/csv;charset=utf-8;' });
                                            const link = document.createElement('a');
                                            link.href = URL.createObjectURL(blob);
                                            link.download = `${module.name}_test.csv`;
                                            link.click();
                                        }}
                                        className="text-gray-500 hover:text-gray-800 p-1 rounded hover:bg-gray-100"
                                        title="Download CSV"
                                    >
                                        <ArrowDownTrayIcon className="w-6 h-6" />
                                    </button>
                                </div>
                                <div className="flex-grow overflow-hidden">
                                    <DataTable title="Test Data" data={output.test} />
                                </div>
                            </>
                        )}
                    </div>
                </main>
            </div>
            {showSpreadView && spreadViewData.length > 0 && (
                <SpreadViewModal
                    onClose={() => setShowSpreadView(false)}
                    data={spreadViewData}
                    columns={spreadViewColumns}
                    title={`Spread View: ${module.name} - ${spreadViewTab === 'train' ? 'Train' : 'Test'} Data`}
                />
            )}
        </div>
    );
};