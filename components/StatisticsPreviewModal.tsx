import React, { useMemo, useState } from 'react';
import { CanvasModule, StatisticsOutput } from '../types';
import { XCircleIcon, SparklesIcon, ArrowDownTrayIcon } from './icons';
import { GoogleGenAI } from "@google/genai";
import { MarkdownRenderer } from './MarkdownRenderer';
import { SpreadViewModal } from './SpreadViewModal';

interface StatisticsPreviewModalProps {
    module: CanvasModule;
    projectName: string;
    onClose: () => void;
}

const CorrelationHeatmap: React.FC<{ matrix: StatisticsOutput['correlation'] }> = ({ matrix }) => {
    const columns = Object.keys(matrix);
    
    const getColor = (value: number) => {
        const alpha = Math.abs(value);
        if (value > 0) return `rgba(59, 130, 246, ${alpha})`; // Blue for positive
        return `rgba(239, 68, 68, ${alpha})`; // Red for negative
    };

    return (
        <div className="p-2">
            <div className="flex text-xs font-bold">
                <div className="w-20 flex-shrink-0"></div>
                {columns.map(col => <div key={col} className="flex-1 text-center truncate" title={col}>{col}</div>)}
            </div>
            {columns.map(rowCol => (
                <div key={rowCol} className="flex items-center text-xs">
                    <div className="w-20 flex-shrink-0 font-bold truncate" title={rowCol}>{rowCol}</div>
                    {columns.map(colCol => (
                        <div key={`${rowCol}-${colCol}`} className="flex-1 p-0.5">
                            <div
                                className="w-full h-6 rounded-sm flex items-center justify-center text-white font-mono"
                                style={{ backgroundColor: getColor(matrix[rowCol][colCol]) }}
                                title={`${rowCol} vs ${colCol}: ${matrix[rowCol][colCol].toFixed(2)}`}
                            >
                                {matrix[rowCol][colCol].toFixed(1)}
                            </div>
                        </div>
                    ))}
                </div>
            ))}
        </div>
    );
};

const Cell: React.FC<{ row: number, col: number, displayColumns: string[], output: StatisticsOutput }> = ({ row, col, displayColumns, output }) => {
    const colNameX = displayColumns[col];
    const colNameY = displayColumns[row];

    if (row === col) { // Diagonal -> Histogram
        const randomBars = useMemo(() => Array.from({ length: 10 }, () => Math.random()), []);
        return (
            <div className="w-full h-full border border-gray-300 rounded flex items-end justify-around gap-px p-1 bg-gray-100">
                {randomBars.map((height, i) => (
                    <div key={i} className="bg-gray-400 w-full" style={{ height: `${height * 100}%` }}></div>
                ))}
            </div>
        );
    } else { // Off-diagonal -> Scatter plot
        const correlation = output.correlation[colNameY]?.[colNameX] || 0;
        const points = useMemo(() => {
            return Array.from({ length: 30 }, () => {
                const xRand = Math.random();
                const noise = (Math.random() - 0.5) * (1 - Math.abs(correlation));
                let y = xRand * correlation + noise;
                if(correlation < 0) y += Math.abs(correlation);
                
                y = Math.max(0, Math.min(1, y));

                return {
                    x: xRand * 100,
                    y: (1-y) * 100
                };
            });
        }, [correlation]);

        return (
             <div className="w-full h-full border border-gray-300 rounded p-1">
                <svg width="100%" height="100%" viewBox="0 0 100 100">
                    {points.map((p, i) => (
                        <circle key={i} cx={p.x} cy={p.y} r="1.5" fill="#3b82f6" />
                    ))}
                </svg>
            </div>
        );
    }
};

const Pairplot: React.FC<{ output: StatisticsOutput }> = ({ output }) => {
    const numericColumns = Object.keys(output.correlation);
    if (numericColumns.length === 0) {
        return <p className="text-sm text-gray-500">No numeric columns to display in pairplot.</p>;
    }
    const displayColumns = numericColumns.slice(0, 5); 

    const gridStyle: React.CSSProperties = {
        display: 'grid',
        gridTemplateColumns: `repeat(${displayColumns.length}, minmax(0, 1fr))`,
        gridTemplateRows: `repeat(${displayColumns.length}, minmax(0, 1fr))`,
        gap: '8px'
    };
    
    return (
        <div>
            {displayColumns.length < numericColumns.length && (
                <p className="text-sm text-gray-500 mb-2">Displaying first {displayColumns.length} of {numericColumns.length} numeric columns for brevity.</p>
            )}
            <div className="flex">
                <div className="flex flex-col justify-around w-20 text-xs font-bold text-right pr-2">
                    {displayColumns.map(col => <div key={col} className="truncate" title={col}>{col}</div>)}
                </div>
                <div className="flex-1" style={{ aspectRatio: '1 / 1' }}>
                    <div style={gridStyle} className="w-full h-full">
                        {displayColumns.map((_, rowIndex) => 
                            displayColumns.map((_, colIndex) => (
                                <Cell key={`${rowIndex}-${colIndex}`} row={rowIndex} col={colIndex} displayColumns={displayColumns} output={output} />
                            ))
                        )}
                    </div>
                </div>
            </div>
             <div className="flex">
                <div className="w-20"></div>
                <div className="flex-1 flex justify-around text-xs font-bold text-center pt-2">
                    {displayColumns.map(col => <div key={col} className="truncate" title={col}>{col}</div>)}
                </div>
            </div>
        </div>
    );
};


export const StatisticsPreviewModal: React.FC<StatisticsPreviewModalProps> = ({ module, projectName, onClose }) => {
    const [isInterpreting, setIsInterpreting] = useState(false);
    const [aiInterpretation, setAiInterpretation] = useState<string | null>(null);
    const [showSpreadView, setShowSpreadView] = useState(false);

    const output = module.outputData as StatisticsOutput;
    const hasValidOutput = output && output.type === 'StatisticsOutput';
    const stats = hasValidOutput ? output.stats : null;
    const correlation = hasValidOutput ? output.correlation : null;

    // Spread View용 데이터 변환 (Descriptive Statistics 테이블)
    const spreadViewData = useMemo(() => {
        if (!stats) return [];
        const data: Array<Record<string, any>> = [];
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
        ];
        
        statDisplay.forEach(({ key, label }) => {
            const row: Record<string, any> = { Metric: label };
            Object.keys(stats).forEach(col => {
                const value = (stats[col] as any)[key];
                let displayValue = value;
                if (typeof value === 'number' && !Number.isInteger(value)) {
                    displayValue = value.toFixed(2);
                }
                row[col] = displayValue === undefined || displayValue === null || Number.isNaN(displayValue) ? 'N/A' : String(displayValue);
            });
            data.push(row);
        });
        
        return data;
    }, [stats]);

    const spreadViewColumns = useMemo(() => {
        if (!stats) return [{ name: 'Metric', type: 'string' }];
        const cols = [{ name: 'Metric', type: 'string' }];
        Object.keys(stats).forEach(col => {
            cols.push({ name: col, type: 'number' });
        });
        return cols;
    }, [stats]);

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
    ];

    const handleInterpret = async () => {
        setIsInterpreting(true);
        setAiInterpretation(null);
        try {
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });

            const statsText = Object.entries(stats).map(([col, data]) => 
                `- ${col}: Mean=${data.mean?.toFixed(2)}, StdDev=${data.std?.toFixed(2)}, Min=${data.min?.toFixed(2)}, Max=${data.max?.toFixed(2)}`
            ).join('\n');

            const correlationText = Object.entries(correlation)
                .flatMap(([col1, inner]) => 
                    Object.entries(inner)
                        .filter(([col2, val]) => col1 < col2 && Math.abs(val) > 0.5) // Find strong correlations
                        .map(([col2, val]) => `- ${col1} and ${col2}: ${val.toFixed(2)}`)
                ).join('\n') || "No strong correlations found.";

            const prompt = `
You are a data analyst writing a brief summary. Please use Korean and simple Markdown.

### 통계 분석 결과 요약

**프로젝트:** ${projectName}

**통계 데이터:**
${statsText}

**강한 상관관계:**
${correlationText}

---

**1. 핵심 관찰:**
- 통계적으로 가장 눈에 띄는 특징(예: 비정상적으로 변동성이 큰 변수)을 한 가지 요약해 주십시오.

**2. 주요 관계:**
- 가장 강한 상관관계를 보이는 변수 쌍은 무엇이며, 이것이 프로젝트 목표에 대해 무엇을 시사합니까?

**3. 다음 단계 제안:**
- 이 분석을 바탕으로 추천하는 가장 중요한 다음 단계를 한 가지 제안해 주십시오.

**지시:** 각 항목을 한두 문장으로 매우 간결하게 작성하십시오.
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
        <div 
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
            onClick={onClose}
        >
            <div 
                className="bg-white text-gray-900 rounded-lg shadow-xl w-full max-w-6xl max-h-[90vh] flex flex-col"
                onClick={e => e.stopPropagation()}
            >
                <header className="flex items-center justify-between p-4 border-b border-gray-200 flex-shrink-0">
                    <h2 className="text-xl font-bold text-gray-800">Statistics Preview: {module.name}</h2>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => setShowSpreadView(true)}
                            className="px-3 py-1.5 text-sm bg-green-600 text-white rounded-md hover:bg-green-700 flex items-center gap-1"
                        >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
                            </svg>
                            Spread View
                        </button>
                        <button
                            onClick={() => {
                                const csvContent = [
                                    spreadViewColumns.map(c => c.name).join(','),
                                    ...spreadViewData.map(row => 
                                        spreadViewColumns.map(col => {
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
                                link.download = `${module.name}_statistics.csv`;
                                link.click();
                            }}
                            className="text-gray-500 hover:text-gray-800 p-1 rounded hover:bg-gray-100"
                            title="Download CSV"
                        >
                            <ArrowDownTrayIcon className="w-6 h-6" />
                        </button>
                        <button onClick={onClose} className="text-gray-500 hover:text-gray-800">
                            <XCircleIcon className="w-6 h-6" />
                        </button>
                    </div>
                </header>
                <main className="flex-grow p-4 overflow-auto flex flex-col gap-6">
                    <div className="flex justify-end font-sans">
                        <button
                            onClick={handleInterpret}
                            disabled={isInterpreting}
                            className="flex items-center gap-2 px-4 py-2 text-sm font-semibold text-white bg-purple-600 rounded-lg hover:bg-purple-700 disabled:bg-purple-400 disabled:cursor-wait transition-colors"
                        >
                            <SparklesIcon className="w-5 h-5" />
                            {isInterpreting ? '분석 중...' : 'AI로 결과 해석하기'}
                        </button>
                    </div>

                    {isInterpreting && <div className="text-center p-4 text-gray-600">AI가 통계 결과를 분석하고 있습니다...</div>}
                    {aiInterpretation && (
                         <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                            <h3 className="text-lg font-bold text-purple-800 mb-2 font-sans flex items-center gap-2">
                                <SparklesIcon className="w-5 h-5"/>
                                AI 분석 요약
                            </h3>
                            <MarkdownRenderer text={aiInterpretation} />
                        </div>
                    )}

                    {!hasValidOutput ? (
                        <div className="flex items-center justify-center p-8 text-gray-500">
                            <div className="text-center">
                                <p className="text-lg font-semibold mb-2">No Statistics Data Available</p>
                                <p className="text-sm">Please run the Statistics module first to view the results.</p>
                            </div>
                        </div>
                    ) : (
                        <>
                            {/* Descriptive Statistics Section */}
                            <div>
                                <h3 className="text-lg font-semibold mb-2 text-gray-700">Descriptive Statistics</h3>
                                <div className="overflow-x-auto border border-gray-200 rounded-lg">
                                    <table className="w-full text-sm text-left table-auto">
                                        <thead className="bg-gray-50">
                                            <tr>
                                                <th className="py-1.5 px-3 font-semibold text-gray-600">Metric</th>
                                                {stats && Object.keys(stats).map(col => (
                                                    <th key={col} className="py-1.5 px-3 font-semibold text-gray-600 text-right">{col}</th>
                                                ))}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {statDisplay.map(({ key, label }) => (
                                                <tr key={key} className="border-b border-gray-200 last:border-b-0">
                                                    <td className="py-1.5 px-3 font-medium text-gray-500">{label}</td>
                                                    {stats && Object.keys(stats).map(col => {
                                                        const value = (stats[col] as any)[key];
                                                        let displayValue = value;
                                                        if (typeof value === 'number' && !Number.isInteger(value)) {
                                                            displayValue = value.toFixed(2);
                                                        }
                                                        return (
                                                            <td key={`${key}-${col}`} className="py-1.5 px-3 font-mono text-right text-gray-700">
                                                                {displayValue === undefined || displayValue === null || Number.isNaN(displayValue) ? 'N/A' : String(displayValue)}
                                                            </td>
                                                        );
                                                    })}
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>

                            {/* Correlation Analysis Section */}
                            {correlation && (
                                <div>
                                    <h3 className="text-lg font-semibold mb-2 text-gray-700">Correlation Analysis</h3>
                                    <div className="overflow-x-auto border border-gray-200 rounded-lg">
                                        <CorrelationHeatmap matrix={correlation} />
                                    </div>
                                </div>
                            )}

                            {/* Pairplot Visualization Section */}
                            {output && (
                                <div>
                                    <h3 className="text-lg font-semibold mb-2 text-gray-700">Pairplot</h3>
                                    <div className="p-4 border border-gray-200 rounded-lg">
                                        <Pairplot output={output} />
                                    </div>
                                </div>
                            )}
                        </>
                    )}
                </main>
            </div>
            {showSpreadView && spreadViewData.length > 0 && (
                <SpreadViewModal
                    onClose={() => setShowSpreadView(false)}
                    data={spreadViewData}
                    columns={spreadViewColumns}
                    title={`Spread View: ${module.name} - Statistics`}
                />
            )}
        </div>
    );
};