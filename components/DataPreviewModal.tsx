import React, { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import { CanvasModule, ColumnInfo, DataPreview, ModuleType } from '../types';
import { XCircleIcon, ChevronUpIcon, ChevronDownIcon, SparklesIcon } from './icons';
import { GoogleGenAI } from "@google/genai";
import { MarkdownRenderer } from './MarkdownRenderer';

interface DataPreviewModalProps {
    module: CanvasModule;
    projectName: string;
    onClose: () => void;
}

type SortConfig = {
    key: string;
    direction: 'ascending' | 'descending';
} | null;

const HistogramPlot: React.FC<{ rows: Record<string, any>[]; column: string; }> = ({ rows, column }) => {
    const data = useMemo(() => rows.map(r => r[column]), [rows, column]);
    const numericData = useMemo(() => data.map(v => parseFloat(v as string)).filter(v => !isNaN(v)), [data]);

    if (numericData.length === 0) {
        return <div className="flex items-center justify-center h-full text-gray-400 text-sm">No numeric data in this column to plot.</div>;
    }

    const { bins } = useMemo(() => {
        const min = Math.min(...numericData);
        const max = Math.max(...numericData);
        const numBins = 10;
        const binSize = (max - min) / numBins;
        const bins = Array(numBins).fill(0);

        for (const value of numericData) {
            let binIndex = binSize > 0 ? Math.floor((value - min) / binSize) : 0;
            if (binIndex === numBins) binIndex--;
            if (binIndex >= 0 && binIndex < numBins) {
                bins[binIndex]++;
            }
        }
        return { bins };
    }, [numericData]);
    
    const maxBinCount = Math.max(...bins, 0);

    return (
        <div className="w-full h-full p-4 flex flex-col border border-gray-200 rounded-lg">
             <div className="flex-grow flex items-center gap-2 overflow-hidden">
                {/* Y-axis Label */}
                <div className="flex items-center justify-center h-full">
                    <p className="text-sm text-gray-600 font-semibold transform -rotate-90 whitespace-nowrap">
                        Frequency
                    </p>
                </div>
                
                {/* Plot area */}
                <div className="flex-grow h-full flex flex-col">
                    <div className="flex-grow flex items-end justify-around gap-1 pt-4">
                        {bins.map((count, index) => {
                            const heightPercentage = maxBinCount > 0 ? (count / maxBinCount) * 100 : 0;
                            return (
                                <div key={index} className="flex-1 h-full flex flex-col justify-end items-center group relative" title={`Count: ${count}`}>
                                    <span className="absolute -top-5 text-xs bg-gray-800 text-white px-1.5 py-0.5 rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">{count}</span>
                                    <div 
                                        className="w-full bg-blue-400 hover:bg-blue-500 transition-colors"
                                        style={{ height: `${heightPercentage}%` }}
                                    >
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                    {/* X-axis Label */}
                    <div className="w-full text-center text-sm text-gray-600 font-semibold mt-2 border-t pt-1">
                        {column}
                    </div>
                </div>
             </div>
        </div>
    );
};

const ScatterPlot: React.FC<{ rows: Record<string, any>[], xCol: string, yCol: string }> = ({ rows, xCol, yCol }) => {
    const dataPoints = useMemo(() => rows.map(r => ({ x: Number(r[xCol]), y: Number(r[yCol]) })).filter(p => !isNaN(p.x) && !isNaN(p.y)), [rows, xCol, yCol]);

    if (dataPoints.length === 0) {
        return <div className="flex items-center justify-center h-full text-gray-400">No valid data points for scatter plot.</div>;
    }

    const margin = { top: 20, right: 20, bottom: 40, left: 50 };
    const width = 600;
    const height = 400;

    const xMin = Math.min(...dataPoints.map(d => d.x));
    const xMax = Math.max(...dataPoints.map(d => d.x));
    const yMin = Math.min(...dataPoints.map(d => d.y));
    const yMax = Math.max(...dataPoints.map(d => d.y));

    const xScale = (x: number) => margin.left + ((x - xMin) / (xMax - xMin || 1)) * (width - margin.left - margin.right);
    const yScale = (y: number) => height - margin.bottom - ((y - yMin) / (yMax - yMin || 1)) * (height - margin.top - margin.bottom);
    
    const getTicks = (min: number, max: number, count: number) => {
        if (min === max) return [min];
        const ticks = [];
        const step = (max - min) / (count - 1);
        for (let i = 0; i < count; i++) {
            ticks.push(min + i * step);
        }
        return ticks;
    };
    
    const xTicks = getTicks(xMin, xMax, 5);
    const yTicks = getTicks(yMin, yMax, 5);

    return (
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto max-w-full">
            {/* Axes */}
            <line x1={margin.left} y1={height - margin.bottom} x2={width - margin.right} y2={height - margin.bottom} stroke="currentColor" strokeWidth="1" />
            <line x1={margin.left} y1={margin.top} x2={margin.left} y2={height - margin.bottom} stroke="currentColor" strokeWidth="1" />

            {/* X Ticks and Labels */}
            {xTicks.map((tick, i) => (
                <g key={`x-${i}`} transform={`translate(${xScale(tick)}, ${height - margin.bottom})`}>
                    <line y2="5" stroke="currentColor" strokeWidth="1" />
                    <text y="20" textAnchor="middle" fill="currentColor" fontSize="10">{tick.toFixed(1)}</text>
                </g>
            ))}
            <text x={width/2} y={height - 5} textAnchor="middle" fill="currentColor" fontSize="12" fontWeight="bold">{xCol}</text>
            
            {/* Y Ticks and Labels */}
            {yTicks.map((tick, i) => (
                <g key={`y-${i}`} transform={`translate(${margin.left}, ${yScale(tick)})`}>
                    <line x2="-5" stroke="currentColor" strokeWidth="1" />
                    <text x="-10" y="3" textAnchor="end" fill="currentColor" fontSize="10">{tick.toFixed(1)}</text>
                </g>
            ))}
            <text transform={`translate(${15}, ${height/2}) rotate(-90)`} textAnchor="middle" fill="currentColor" fontSize="12" fontWeight="bold">{yCol}</text>

            {/* Points */}
            <g>
                {dataPoints.map((d, i) => (
                    <circle key={i} cx={xScale(d.x)} cy={yScale(d.y)} r="2.5" fill="rgba(59, 130, 246, 0.7)" />
                ))}
            </g>
        </svg>
    );
};

const ColumnStatistics: React.FC<{ data: (string | number | null)[]; columnName: string | null; isNumeric: boolean; }> = ({ data, columnName, isNumeric }) => {
    const stats = useMemo(() => {
        const isNull = (v: any) => v === null || v === undefined || v === '';
        const nonNullValues = data.filter(v => !isNull(v));
        const nulls = data.length - nonNullValues.length;
        const count = data.length;

        let mode: number | string = 'N/A';
        if (nonNullValues.length > 0) {
            const counts: Record<string, number> = {};
            for(const val of nonNullValues) {
                const key = String(val);
                counts[key] = (counts[key] || 0) + 1;
            }
            mode = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
        }

        if (!isNumeric) {
            return {
                Count: count,
                Null: nulls,
                Mode: mode,
            };
        }
        
        const numericValues = nonNullValues.map(v => Number(v)).filter(v => !isNaN(v));

        if (numericValues.length === 0) {
             return {
                Count: count,
                Null: nulls,
                Mode: mode,
            };
        }
        
        numericValues.sort((a,b) => a - b);
        const sum = numericValues.reduce((a, b) => a + b, 0);
        const mean = sum / numericValues.length;
        const n = numericValues.length;
        const stdDev = Math.sqrt(numericValues.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b) / n);
        const skewness = stdDev > 0 ? numericValues.reduce((s, val) => s + Math.pow(val - mean, 3), 0) / (n * Math.pow(stdDev, 3)) : 0;
        const kurtosis = stdDev > 0 ? numericValues.reduce((s, val) => s + Math.pow(val - mean, 4), 0) / (n * Math.pow(stdDev, 4)) - 3 : 0;


        const getQuantile = (q: number) => {
            const pos = (numericValues.length - 1) * q;
            const base = Math.floor(pos);
            const rest = pos - base;
            if (numericValues[base + 1] !== undefined) {
                return numericValues[base] + rest * (numericValues[base + 1] - numericValues[base]);
            } else {
                return numericValues[base];
            }
        };

        const numericMode = Number(mode);

        return {
            Count: data.length,
            Mean: mean.toFixed(2),
            'Std Dev': stdDev.toFixed(2),
            Median: getQuantile(0.5).toFixed(2),
            Min: numericValues[0].toFixed(2),
            Max: numericValues[numericValues.length - 1].toFixed(2),
            '25%': getQuantile(0.25).toFixed(2),
            '75%': getQuantile(0.75).toFixed(2),
            Mode: isNaN(numericMode) ? mode : numericMode,
            Null: nulls,
            Skew: skewness.toFixed(2),
            Kurt: kurtosis.toFixed(2),
        };
    }, [data, isNumeric]);
    
    const statOrder = isNumeric 
        ? ['Count', 'Mean', 'Std Dev', 'Median', 'Min', 'Max', '25%', '75%', 'Mode', 'Null', 'Skew', 'Kurt']
        : ['Count', 'Null', 'Mode'];

    return (
        <div className="w-full p-4 border border-gray-200 rounded-lg">
            <h4 className="font-semibold text-gray-700 mb-3">Statistics for {columnName}</h4>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-x-4 gap-y-1 text-sm">
                {statOrder.map(key => {
                    const value = (stats as Record<string, any>)[key];
                    if (value === undefined || value === null) return null;
                    return (
                        <React.Fragment key={key}>
                           <span className="text-gray-500">{key}:</span> 
                           <span className="font-mono text-gray-800 font-medium">{String(value)}</span>
                        </React.Fragment>
                    );
                })}
            </div>
        </div>
    );
};


export const DataPreviewModal: React.FC<DataPreviewModalProps> = ({ module, projectName, onClose }) => {
    // 안전한 데이터 가져오기
    const getPreviewData = (): DataPreview | null => {
        try {
            if (!module || !module.outputData) return null;
            if (module.outputData.type === 'DataPreview') return module.outputData;
            if (module.outputData.type === 'KMeansOutput' || module.outputData.type === 'HierarchicalClusteringOutput' || module.outputData.type === 'DBSCANOutput') {
                return module.outputData.clusterAssignments || null;
            }
            if (module.outputData.type === 'PCAOutput') {
                return module.outputData.transformedData || null;
            }
            return null;
        } catch (error) {
            console.error('Error in getPreviewData:', error);
            return null;
        }
    };
    
    const data = getPreviewData();
    const columns = Array.isArray(data?.columns) ? data.columns : [];
    const rows = Array.isArray(data?.rows) ? data.rows : [];
    
    const [sortConfig, setSortConfig] = useState<SortConfig>(null);
    const [selectedColumn, setSelectedColumn] = useState<string | null>(columns[0]?.name || null);
    const [activeTab, setActiveTab] = useState<'table' | 'visualization'>('table');
    const [yAxisCol, setYAxisCol] = useState<string | null>(null);
    
    // Score Model용 label column state
    const predictCol = useMemo(() => {
        try {
            if (module.type === ModuleType.ScoreModel && Array.isArray(columns) && columns.length > 0) {
                return columns.find(c => c && c.name && (c.name === 'Predict' || c.name.toLowerCase().includes('predict'))) || null;
            }
            return null;
        } catch (error) {
            console.error('Error finding predict column:', error);
            return null;
        }
    }, [module.type, columns]);
    
    const labelCols = useMemo(() => {
        try {
            if (module.type === ModuleType.ScoreModel && Array.isArray(columns) && columns.length > 0) {
                return columns.filter(c => c && c.type === 'number' && c.name && c.name !== predictCol?.name);
            }
            return [];
        } catch (error) {
            console.error('Error filtering label columns:', error);
            return [];
        }
    }, [module.type, columns, predictCol]);
    
    const defaultLabelCol = useMemo(() => {
        try {
            return labelCols && labelCols.length > 0 ? (labelCols[0]?.name || null) : null;
        } catch (error) {
            console.error('Error getting default label column:', error);
            return null;
        }
    }, [labelCols]);
    
    const moduleIdRef = useRef<string | null>(null); // 모듈 ID 추적
    const [scoreLabelCol, setScoreLabelCol] = useState<string | null>(null);
    
    // Score Model의 경우 모듈이 변경될 때만 초기화
    useEffect(() => {
        try {
            if (module.type === ModuleType.ScoreModel) {
                // 모듈이 변경된 경우에만 초기화
                if (moduleIdRef.current !== module.id) {
                    moduleIdRef.current = module.id;
                    setScoreLabelCol(defaultLabelCol);
                }
            } else {
                // Score Model이 아닌 경우 리셋
                if (moduleIdRef.current !== null) {
                    moduleIdRef.current = null;
                    setScoreLabelCol(null);
                }
            }
        } catch (error) {
            console.error('Error in Score Model label column initialization:', error);
        }
    }, [module.id, module.type, defaultLabelCol]); // defaultLabelCol을 의존성에 추가

    const sortedRows = useMemo(() => {
        try {
            if (!Array.isArray(rows)) return [];
            let sortableItems = [...rows];
            if (sortConfig !== null && sortConfig.key) {
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
        } catch (error) {
            console.error('Error sorting rows:', error);
            return Array.isArray(rows) ? rows : [];
        }
    }, [rows, sortConfig]);

    const requestSort = (key: string) => {
        let direction: 'ascending' | 'descending' = 'ascending';
        if (sortConfig && sortConfig.key === key && sortConfig.direction === 'ascending') {
            direction = 'descending';
        }
        setSortConfig({ key, direction });
    };

    const selectedColumnData = useMemo(() => {
        try {
            if (!selectedColumn || !Array.isArray(rows)) return null;
            return rows.map(row => row[selectedColumn]);
        } catch (error) {
            console.error('Error getting selected column data:', error);
            return null;
        }
    }, [selectedColumn, rows]);
    
    const selectedColInfo = useMemo(() => {
        try {
            if (!Array.isArray(columns) || !selectedColumn) return null;
            return columns.find(c => c && c.name === selectedColumn) || null;
        } catch (error) {
            console.error('Error finding selected column info:', error);
            return null;
        }
    }, [columns, selectedColumn]);
    
    const isSelectedColNumeric = useMemo(() => selectedColInfo?.type === 'number', [selectedColInfo]);
    
    const numericCols = useMemo(() => {
        try {
            if (!Array.isArray(columns)) return [];
            return columns.filter(c => c && c.type === 'number').map(c => c.name).filter(Boolean);
        } catch (error) {
            console.error('Error getting numeric columns:', error);
            return [];
        }
    }, [columns]);

    useEffect(() => {
        if (isSelectedColNumeric && selectedColumn) {
            const defaultY = numericCols.find(c => c !== selectedColumn);
            setYAxisCol(defaultY || null);
        } else {
            setYAxisCol(null);
        }
    }, [isSelectedColNumeric, selectedColumn, numericCols]);

    if (!data) {
        console.warn('DataPreviewModal: No data available for module', module.id, module.type, module.outputData);
        return (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={onClose}>
                <div className="bg-white p-6 rounded-lg shadow-xl" onClick={e => e.stopPropagation()}>
                    <h3 className="text-lg font-bold">No Data Available</h3>
                    <p>The selected module has no previewable data.</p>
                    <p className="text-sm text-gray-500 mt-2">Module Type: {module.type}</p>
                    <p className="text-sm text-gray-500">Output Data Type: {module.outputData?.type || 'null'}</p>
                </div>
            </div>
        );
    }
    
    return (
        <div 
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
            onClick={onClose}
        >
            <div 
                className="bg-white text-gray-900 rounded-lg shadow-xl w-full max-w-7xl max-h-[90vh] flex flex-col"
                onClick={e => e.stopPropagation()}
            >
                <header className="flex items-center justify-between p-4 border-b border-gray-200 flex-shrink-0">
                    <h2 className="text-xl font-bold text-gray-800">Data Preview: {module.name}</h2>
                    <button onClick={onClose} className="text-gray-500 hover:text-gray-800">
                        <XCircleIcon className="w-6 h-6" />
                    </button>
                </header>
                <main className="flex-grow p-4 overflow-auto flex flex-col gap-4">
                    <div className="flex-shrink-0 border-b border-gray-200">
                        <nav className="-mb-px flex space-x-8" aria-label="Tabs">
                            <button
                                onClick={() => setActiveTab('table')}
                                className={`${
                                    activeTab === 'table'
                                        ? 'border-indigo-500 text-indigo-600'
                                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
                            >
                                Data Table
                            </button>
                            <button
                                onClick={() => setActiveTab('visualization')}
                                className={`${
                                    activeTab === 'visualization'
                                        ? 'border-indigo-500 text-indigo-600'
                                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
                            >
                                Visualization
                            </button>
                        </nav>
                    </div>

                    {activeTab === 'table' && (
                        <>
                            <div className="flex justify-between items-center flex-shrink-0">
                                <div className="text-sm text-gray-600">
                                    Showing {Math.min(rows.length, 1000)} of {data.totalRowCount.toLocaleString()} rows and {columns.length} columns. Click a column to see details.
                                </div>
                            </div>
                            <div className="flex-grow flex gap-4 overflow-hidden">
                                {/* Score Model인 경우 테이블만 표시 */}
                                {module.type === ModuleType.ScoreModel ? (
                                    <div className="w-full overflow-auto border border-gray-200 rounded-lg">
                                        <table className="min-w-full text-sm text-left">
                                            <thead className="bg-gray-50 sticky top-0">
                                                <tr>
                                                    {columns.map(col => (
                                                        <th 
                                                            key={col.name} 
                                                            className="py-2 px-3 font-semibold text-gray-600 cursor-pointer hover:bg-gray-100"
                                                            onClick={() => requestSort(col.name)}
                                                        >
                                                            <div className="flex items-center gap-1">
                                                                <span className="truncate" title={col.name}>{col.name}</span>
                                                                {sortConfig?.key === col.name && (sortConfig.direction === 'ascending' ? <ChevronUpIcon className="w-3 h-3" /> : <ChevronDownIcon className="w-3 h-3" />)}
                                                            </div>
                                                        </th>
                                                    ))}
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {sortedRows.map((row, rowIndex) => (
                                                    <tr key={rowIndex} className="border-b border-gray-200 last:border-b-0">
                                                        {columns.map(col => (
                                                            <td 
                                                                key={col.name} 
                                                                className="py-1.5 px-3 font-mono truncate hover:bg-gray-50"
                                                                title={String(row[col.name])}
                                                            >
                                                                {row[col.name] === null ? <i className="text-gray-400">null</i> : String(row[col.name])}
                                                            </td>
                                                        ))}
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                ) : (
                                    <>
                                        <div className={`overflow-auto border border-gray-200 rounded-lg ${selectedColumnData ? 'w-1/2' : 'w-full'}`}>
                                            <table className="min-w-full text-sm text-left">
                                                <thead className="bg-gray-50 sticky top-0">
                                                    <tr>
                                                        {columns.map(col => (
                                                            <th 
                                                                key={col.name} 
                                                                className="py-2 px-3 font-semibold text-gray-600 cursor-pointer hover:bg-gray-100"
                                                                onClick={() => requestSort(col.name)}
                                                            >
                                                                <div className="flex items-center gap-1">
                                                                    <span className="truncate" title={col.name}>{col.name}</span>
                                                                    {sortConfig?.key === col.name && (sortConfig.direction === 'ascending' ? <ChevronUpIcon className="w-3 h-3" /> : <ChevronDownIcon className="w-3 h-3" />)}
                                                                </div>
                                                            </th>
                                                        ))}
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {sortedRows.map((row, rowIndex) => (
                                                        <tr key={rowIndex} className="border-b border-gray-200 last:border-b-0">
                                                            {columns.map(col => (
                                                                <td 
                                                                    key={col.name} 
                                                                    className={`py-1.5 px-3 font-mono truncate ${selectedColumn === col.name ? 'bg-blue-100' : 'hover:bg-gray-50 cursor-pointer'}`}
                                                                    onClick={() => setSelectedColumn(col.name)}
                                                                    title={String(row[col.name])}
                                                                >
                                                                    {row[col.name] === null ? <i className="text-gray-400">null</i> : String(row[col.name])}
                                                                </td>
                                                            ))}
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        </div>
                                        {selectedColumnData && (
                                            <div className="w-1/2 flex flex-col gap-4">
                                                {isSelectedColNumeric ? (
                                                    <HistogramPlot rows={rows} column={selectedColumn!} />
                                                ) : (
                                                    <div className="w-full h-full p-4 flex flex-col border border-gray-200 rounded-lg items-center justify-center">
                                                        <p className="text-gray-500">Plot is not available for non-numeric columns.</p>
                                                    </div>
                                                )}
                                                <ColumnStatistics data={selectedColumnData} columnName={selectedColumn} isNumeric={isSelectedColNumeric} />
                                            </div>
                                        )}
                                    </>
                                )}
                            </div>
                        </>
                    )}

                    {activeTab === 'visualization' && (
                        <div className="flex-grow flex flex-col items-center justify-start p-4 gap-4">
                            {/* Score Model인 경우 y값과 predict 비교 */}
                            {module.type === ModuleType.ScoreModel ? (
                                <>
                                    {!predictCol ? (
                                        <div className="flex items-center justify-center h-full">
                                            <p className="text-gray-500">Predict column not found in the data.</p>
                                        </div>
                                    ) : labelCols.length === 0 ? (
                                        <div className="flex items-center justify-center h-full">
                                            <p className="text-gray-500">No label column found for comparison.</p>
                                        </div>
                                    ) : (
                                        <>
                                            <div className="flex-shrink-0 flex items-center gap-2 self-start">
                                                <label htmlFor="label-select" className="font-semibold text-gray-700">Label Column (Y):</label>
                                                <select
                                                    id="label-select"
                                                    value={scoreLabelCol || ''}
                                                    onChange={e => setScoreLabelCol(e.target.value)}
                                                    className="p-2 border border-gray-300 rounded-md"
                                                >
                                                    <option value="" disabled>Select a column</option>
                                                    {labelCols.map(col => (
                                                        <option key={col.name} value={col.name}>{col.name}</option>
                                                    ))}
                                                </select>
                                            </div>
                                            <div className="w-full flex-grow min-h-0">
                                                {scoreLabelCol ? (
                                                    <ScatterPlot rows={rows} xCol={predictCol.name} yCol={scoreLabelCol} />
                                                ) : (
                                                    <div className="flex items-center justify-center h-full">
                                                        <p className="text-gray-500">Please select a label column to compare with predictions.</p>
                                                    </div>
                                                )}
                                            </div>
                                        </>
                                    )}
                                </>
                            ) : (
                                <>
                                    {!selectedColumn ? (
                                        <div className="flex items-center justify-center h-full">
                                            <p className="text-gray-500">Select a column from the Data Table to use as the X-axis.</p>
                                        </div>
                                    ) : (
                                        <>
                                            {isSelectedColNumeric && numericCols.length > 1 && (
                                                <div className="flex-shrink-0 flex items-center gap-2 self-start">
                                                    <label htmlFor="y-axis-select" className="font-semibold text-gray-700">Y-Axis:</label>
                                                    <select
                                                        id="y-axis-select"
                                                        value={yAxisCol || ''}
                                                        onChange={e => setYAxisCol(e.target.value)}
                                                        className="p-2 border border-gray-300 rounded-md"
                                                    >
                                                        <option value="" disabled>Select a column</option>
                                                        {numericCols.filter(c => c !== selectedColumn).map(col => (
                                                            <option key={col} value={col}>{col}</option>
                                                        ))}
                                                    </select>
                                                </div>
                                            )}

                                            <div className="w-full flex-grow min-h-0">
                                                {isSelectedColNumeric ? (
                                                    yAxisCol ? (
                                                        <ScatterPlot rows={rows} xCol={selectedColumn} yCol={yAxisCol} />
                                                    ) : (
                                                        <HistogramPlot rows={rows} column={selectedColumn} />
                                                    )
                                                ) : (
                                                    <HistogramPlot rows={rows} column={selectedColumn} />
                                                )}
                                            </div>
                                        </>
                                    )}
                                </>
                            )}
                        </div>
                    )}
                </main>
            </div>
        </div>
    );
};