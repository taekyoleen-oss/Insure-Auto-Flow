import React, { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import { CanvasModule, ColumnInfo, DataPreview, ModuleType, Connection } from '../types';
import { XCircleIcon, ChevronUpIcon, ChevronDownIcon, SparklesIcon, ArrowDownTrayIcon } from './icons';
import { GoogleGenAI } from "@google/genai";
import { MarkdownRenderer } from './MarkdownRenderer';
import { SpreadViewModal } from './SpreadViewModal';

interface DataPreviewModalProps {
    module: CanvasModule;
    projectName: string;
    onClose: () => void;
    modules?: CanvasModule[];
    connections?: Connection[];
}

type SortConfig = {
    key: string;
    direction: 'ascending' | 'descending';
} | null;

// Statistics 모듈의 CorrelationHeatmap 컴포넌트 (Load Data 모듈용)
const CorrelationHeatmap: React.FC<{ matrix: Record<string, Record<string, number>> }> = ({ matrix }) => {
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
                                style={{ backgroundColor: getColor(matrix[rowCol]?.[colCol] || 0) }}
                                title={`${rowCol} vs ${colCol}: ${(matrix[rowCol]?.[colCol] || 0).toFixed(2)}`}
                            >
                                {(matrix[rowCol]?.[colCol] || 0).toFixed(1)}
                            </div>
                        </div>
                    ))}
                </div>
            ))}
        </div>
    );
};

// Statistics 모듈의 Pairplot Cell 컴포넌트 (Load Data 모듈용)
const PairplotCell: React.FC<{ 
    row: number; 
    col: number; 
    displayColumns: string[]; 
    correlation: Record<string, Record<string, number>>;
    rows: Record<string, any>[];
}> = ({ row, col, displayColumns, correlation, rows }) => {
    const colNameX = displayColumns[col];
    const colNameY = displayColumns[row];

    if (row === col) { // Diagonal -> Histogram
        const columnData = rows.map(r => parseFloat(r[colNameX])).filter(v => !isNaN(v));
        if (columnData.length === 0) {
            return (
                <div className="w-full h-full border border-gray-300 rounded flex items-center justify-center text-xs text-gray-400">
                    No data
                </div>
            );
        }
        const min = Math.min(...columnData);
        const max = Math.max(...columnData);
        const numBins = 10;
        const binSize = (max - min) / numBins || 1;
        const bins = Array(numBins).fill(0);
        
        for (const value of columnData) {
            let binIndex = binSize > 0 ? Math.floor((value - min) / binSize) : 0;
            if (binIndex === numBins) binIndex--;
            if (binIndex >= 0 && binIndex < numBins) {
                bins[binIndex]++;
            }
        }
        
        const maxBinCount = Math.max(...bins, 1);
        
        return (
            <div className="w-full h-full border border-gray-300 rounded flex items-end justify-around gap-px p-1 bg-gray-100">
                {bins.map((count, i) => (
                    <div 
                        key={i} 
                        className="bg-gray-400 w-full" 
                        style={{ height: `${(count / maxBinCount) * 100}%` }}
                    />
                ))}
            </div>
        );
    } else { // Off-diagonal -> Scatter plot
        const corrValue = correlation[colNameY]?.[colNameX] || 0;
        const xData = rows.map(r => parseFloat(r[colNameX])).filter(v => !isNaN(v));
        const yData = rows.map(r => parseFloat(r[colNameY])).filter(v => !isNaN(v));
        
        if (xData.length === 0 || yData.length === 0) {
            return (
                <div className="w-full h-full border border-gray-300 rounded flex items-center justify-center text-xs text-gray-400">
                    No data
                </div>
            );
        }
        
        const xMin = Math.min(...xData);
        const xMax = Math.max(...xData);
        const yMin = Math.min(...yData);
        const yMax = Math.max(...yData);
        
        const xRange = xMax - xMin || 1;
        const yRange = yMax - yMin || 1;
        
        const points = xData.map((x, i) => {
            const y = yData[i];
            return {
                x: ((x - xMin) / xRange) * 100,
                y: ((y - yMin) / yRange) * 100
            };
        }).slice(0, 100); // 최대 100개 포인트만 표시
        
        return (
            <div className="w-full h-full border border-gray-300 rounded p-1 relative">
                <div className="absolute top-0 left-0 text-xs font-semibold text-gray-700 px-1 bg-white bg-opacity-75 rounded">
                    r = {corrValue.toFixed(2)}
                </div>
                <svg width="100%" height="100%" viewBox="0 0 100 100" className="overflow-visible">
                    {points.map((p, i) => (
                        <circle key={i} cx={p.x} cy={100 - p.y} r="1.5" fill="#3b82f6" opacity="0.6" />
                    ))}
                </svg>
            </div>
        );
    }
};

// Statistics 모듈의 Pairplot 컴포넌트 (Load Data 모듈용)
const Pairplot: React.FC<{ 
    correlation: Record<string, Record<string, number>>;
    numericColumns: string[];
    rows: Record<string, any>[];
}> = ({ correlation, numericColumns, rows }) => {
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
                                <PairplotCell 
                                    key={`${rowIndex}-${colIndex}`} 
                                    row={rowIndex} 
                                    col={colIndex} 
                                    displayColumns={displayColumns} 
                                    correlation={correlation}
                                    rows={rows}
                                />
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


// 상관계수 계산 함수
const calculateCorrelation = (x: number[], y: number[]): number => {
    if (x.length !== y.length || x.length === 0) return 0;
    
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2 = x.reduce((a, b) => a + b * b, 0);
    const sumY2 = y.reduce((a, b) => a + b * b, 0);
    
    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    if (denominator === 0) return 0;
    return numerator / denominator;
};

// 상관계수 행렬 계산
const calculateCorrelationMatrix = (rows: Record<string, any>[], numericColumns: string[]): number[][] => {
    const matrix: number[][] = [];
    
    for (let i = 0; i < numericColumns.length; i++) {
        matrix[i] = [];
        for (let j = 0; j < numericColumns.length; j++) {
            if (i === j) {
                matrix[i][j] = 1;
            } else {
                const col1 = numericColumns[i];
                const col2 = numericColumns[j];
                const values1 = rows.map(r => Number(r[col1])).filter(v => !isNaN(v));
                const values2 = rows.map(r => Number(r[col2])).filter(v => !isNaN(v));
                
                // 길이가 같은 값들만 사용
                const minLength = Math.min(values1.length, values2.length);
                const valid1 = values1.slice(0, minLength);
                const valid2 = values2.slice(0, minLength);
                
                matrix[i][j] = calculateCorrelation(valid1, valid2);
            }
        }
    }
    
    return matrix;
};

// 작은 히스토그램 플롯 (Pairplot 대각선용)
const SmallHistogram: React.FC<{ rows: Record<string, any>[]; column: string }> = ({ rows, column }) => {
    const data = useMemo(() => rows.map(r => r[column]), [rows, column]);
    const numericData = useMemo(() => data.map(v => parseFloat(v as string)).filter(v => !isNaN(v)), [data]);

    if (numericData.length === 0) {
        return <div className="flex items-center justify-center h-full text-gray-400 text-xs">No data</div>;
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
        <div className="w-full h-full p-2">
            <div className="flex-grow flex items-end justify-around gap-0.5 h-full">
                {bins.map((count, index) => {
                    const heightPercentage = maxBinCount > 0 ? (count / maxBinCount) * 100 : 0;
                    return (
                        <div key={index} className="flex-1 h-full flex flex-col justify-end items-center">
                            <div 
                                className="w-full bg-blue-400"
                                style={{ height: `${heightPercentage}%` }}
                            />
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

// 작은 산점도 플롯 (Pairplot용)
const SmallScatterPlot: React.FC<{ rows: Record<string, any>[], xCol: string, yCol: string }> = ({ rows, xCol, yCol }) => {
    const dataPoints = useMemo(() => rows.map(r => ({ x: Number(r[xCol]), y: Number(r[yCol]) })).filter(p => !isNaN(p.x) && !isNaN(p.y)), [rows, xCol, yCol]);

    if (dataPoints.length === 0) {
        return <div className="flex items-center justify-center h-full text-gray-400 text-xs">No data</div>;
    }

    const margin = { top: 5, right: 5, bottom: 20, left: 20 };
    const width = 120;
    const height = 120;

    const xMin = Math.min(...dataPoints.map(d => d.x));
    const xMax = Math.max(...dataPoints.map(d => d.x));
    const yMin = Math.min(...dataPoints.map(d => d.y));
    const yMax = Math.max(...dataPoints.map(d => d.y));

    const xScale = (x: number) => margin.left + ((x - xMin) / (xMax - xMin || 1)) * (width - margin.left - margin.right);
    const yScale = (y: number) => height - margin.bottom - ((y - yMin) / (yMax - yMin || 1)) * (height - margin.top - margin.bottom);

    return (
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full">
            {/* Axes */}
            <line x1={margin.left} y1={height - margin.bottom} x2={width - margin.right} y2={height - margin.bottom} stroke="currentColor" strokeWidth="0.5" />
            <line x1={margin.left} y1={margin.top} x2={margin.left} y2={height - margin.bottom} stroke="currentColor" strokeWidth="0.5" />

            {/* Points */}
            <g>
                {dataPoints.map((d, i) => (
                    <circle key={i} cx={xScale(d.x)} cy={yScale(d.y)} r="1.5" fill="rgba(59, 130, 246, 0.6)" />
                ))}
            </g>
        </svg>
    );
};

// Pairplot 컴포넌트
const CorrelationPlots: React.FC<{ 
    correlationMatrix: number[][]; 
    columnNames: string[];
    rows: Record<string, any>[];
}> = ({ correlationMatrix, columnNames, rows }) => {
    const numCols = columnNames.length;
    
    return (
        <div className="w-full overflow-auto">
            <div 
                className="inline-grid gap-1 border border-gray-300 p-2 bg-white"
                style={{ 
                    gridTemplateColumns: `repeat(${numCols}, minmax(120px, 1fr))`,
                    gridTemplateRows: `repeat(${numCols}, minmax(120px, 1fr))`
                }}
            >
                {columnNames.map((colY, rowIdx) =>
                    columnNames.map((colX, colIdx) => {
                        const isDiagonal = rowIdx === colIdx;
                        const isUpperTriangle = rowIdx < colIdx;
                        const isLowerTriangle = rowIdx > colIdx;
                        
                        return (
                            <div 
                                key={`${rowIdx}-${colIdx}`}
                                className="border border-gray-200 rounded bg-white relative"
                                style={{ minWidth: '120px', minHeight: '120px' }}
                            >
                                {isDiagonal ? (
                                    // 대각선: 히스토그램
                                    <>
                                        <div className="absolute top-1 left-1 text-xs font-semibold text-gray-700 z-10">
                                            {colX.length > 10 ? colX.substring(0, 10) + '...' : colX}
                                        </div>
                                        <SmallHistogram rows={rows} column={colX} />
                                    </>
                                ) : isUpperTriangle ? (
                                    // 위쪽 삼각형: 산점도 (상관계수 표시)
                                    <>
                                        <div className="absolute top-1 left-1 text-xs font-semibold text-gray-700 z-10">
                                            r = {correlationMatrix[rowIdx][colIdx].toFixed(2)}
                                        </div>
                                        <SmallScatterPlot rows={rows} xCol={colX} yCol={colY} />
                                    </>
                                ) : (
                                    // 아래쪽 삼각형: 산점도 (상관계수 표시)
                                    <>
                                        <div className="absolute top-1 left-1 text-xs font-semibold text-gray-700 z-10">
                                            r = {correlationMatrix[rowIdx][colIdx].toFixed(2)}
                                        </div>
                                        <SmallScatterPlot rows={rows} xCol={colX} yCol={colY} />
                                    </>
                                )}
                            </div>
                        );
                    })
                )}
            </div>
        </div>
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


export const DataPreviewModal: React.FC<DataPreviewModalProps> = ({ module, projectName, onClose, modules = [], connections = [] }) => {
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
        // MissingHandlerOutput, EncoderOutput, NormalizerOutput의 경우 inputData 사용
        if (module.outputData.type === 'MissingHandlerOutput' || 
            module.outputData.type === 'EncoderOutput' || 
            module.outputData.type === 'NormalizerOutput') {
            // data_in 포트로 연결된 inputData 찾기
            if (!connections || !modules) {
                console.warn('DataPreviewModal: connections or modules not provided for', module.outputData.type);
                return null;
            }
            // 먼저 data_in 포트를 찾고, 없으면 data 포트도 시도
            let dataConnection = connections.find(
                c => c.to.moduleId === module.id && c.to.portName === 'data_in'
            );
            if (!dataConnection) {
                dataConnection = connections.find(
                    c => c.to.moduleId === module.id && c.to.portName === 'data'
                );
            }
            if (dataConnection) {
                const sourceModule = modules.find(m => m.id === dataConnection.from.moduleId);
                if (sourceModule?.outputData) {
                    if (sourceModule.outputData.type === 'DataPreview') {
                        return sourceModule.outputData;
                    }
                    if (sourceModule.outputData.type === 'SplitDataOutput') {
                        // SplitDataOutput의 경우 train 데이터 사용
                        return sourceModule.outputData.train;
                    }
                }
            } else {
                console.warn('DataPreviewModal: No data connection found for', module.id, module.outputData.type, 'Available connections:', connections.filter(c => c.to.moduleId === module.id).map(c => c.to.portName));
            }
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
    const [yAxisCol, setYAxisCol] = useState<string | null>(null);
    const [showSpreadView, setShowSpreadView] = useState(false);
    
    // Load Data 모듈용 탭 상태
    const [loadDataTab, setLoadDataTab] = useState<'detail' | 'graph'>('detail');
    const [graphXCol, setGraphXCol] = useState<string | null>(null);
    const [graphYCol, setGraphYCol] = useState<string | null>(null);
    
    // Load Data 모듈인지 확인
    const isLoadDataModule = module.type === ModuleType.LoadData;
    // Select Data 모듈도 Load Data와 동일한 형식으로 표시
    const isSelectDataModule = module.type === ModuleType.SelectData;
    // Transition Data, Resample Data, Prep Encode, Prep Normalize, Transform Data, HandleMissingValues도 동일한 형식으로 표시
    const isDataModule = isLoadDataModule || 
                         isSelectDataModule || 
                         module.type === ModuleType.TransitionData ||
                         module.type === ModuleType.ResampleData ||
                         module.type === ModuleType.EncodeCategorical ||
                         module.type === ModuleType.NormalizeData ||
                         module.type === ModuleType.TransformData ||
                         module.type === ModuleType.HandleMissingValues;

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
    
    // Load Data/Select Data 모듈용: Graph 탭에서 사용할 열 초기화 (Detail 탭에서 선택된 열을 기본으로 사용)
    useEffect(() => {
        if (isDataModule && numericCols.length >= 2) {
            // Detail 탭에서 선택된 열이 숫자형이면 기본값으로 사용
            if (selectedColumn && isSelectedColNumeric) {
                if (!graphXCol || graphXCol !== selectedColumn) {
                    setGraphXCol(selectedColumn);
                }
                if (!graphYCol || graphYCol === selectedColumn) {
                    const defaultY = numericCols.find(c => c !== selectedColumn) || numericCols[1] || null;
                    setGraphYCol(defaultY);
                }
            } else if (!graphXCol) {
                // 선택된 열이 없거나 숫자형이 아니면 첫 번째 숫자형 열 사용
                setGraphXCol(numericCols[0] || null);
            }
            if (!graphYCol && graphXCol) {
                const defaultY = numericCols.find(c => c !== graphXCol) || numericCols[1] || null;
                setGraphYCol(defaultY);
            }
        }
    }, [isDataModule, numericCols, selectedColumn, isSelectedColNumeric, graphXCol, graphYCol]);
    
    // 상관계수 행렬 계산 (Load Data/Select Data 모듈용)
    const correlationMatrix = useMemo(() => {
        if (!isDataModule || numericCols.length < 2) return null;
        return calculateCorrelationMatrix(rows, numericCols);
    }, [isLoadDataModule, rows, numericCols]);
    
    // correlationMatrix를 Statistics 형식의 correlation으로 변환
    const correlation = useMemo(() => {
        if (!correlationMatrix || !numericCols.length) return null;
        const result: Record<string, Record<string, number>> = {};
        numericCols.forEach((col, i) => {
            result[col] = {};
            numericCols.forEach((col2, j) => {
                result[col][col2] = correlationMatrix[i][j];
            });
        });
        return result;
    }, [correlationMatrix, numericCols]);

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
                                if (!data || !data.columns || !data.rows) return;
                                const csvContent = [
                                    data.columns.map(c => c.name).join(','),
                                    ...data.rows.map(row => 
                                        data.columns.map(col => {
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
                                link.download = `${module.name}_data.csv`;
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
                <main className="flex-grow p-4 overflow-auto flex flex-col gap-4">
                    {/* Load Data/Select Data 모듈용 탭 */}
                    {isDataModule && (
                        <div className="flex-shrink-0 border-b border-gray-200">
                            <nav className="-mb-px flex space-x-8" aria-label="Tabs">
                                <button
                                    onClick={() => setLoadDataTab('detail')}
                                    className={`${
                                        loadDataTab === 'detail'
                                            ? 'border-indigo-500 text-indigo-600'
                                            : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                    } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
                                >
                                    Detail
                                </button>
                                <button
                                    onClick={() => setLoadDataTab('graph')}
                                    className={`${
                                        loadDataTab === 'graph'
                                            ? 'border-indigo-500 text-indigo-600'
                                            : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                    } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
                                >
                                    Graph
                                </button>
                            </nav>
                        </div>
                    )}
                    
                    {isDataModule && loadDataTab === 'graph' ? (
                        /* Graph 탭 */
                        <div className="flex-grow flex flex-col gap-4">
                            <div className="flex-shrink-0 flex items-center gap-4">
                                <div className="flex items-center gap-2">
                                    <label htmlFor="graph-x-select" className="font-semibold text-gray-700">X-Axis:</label>
                                    <select
                                        id="graph-x-select"
                                        value={graphXCol || ''}
                                        onChange={e => setGraphXCol(e.target.value)}
                                        className="px-3 py-1.5 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                                    >
                                        <option value="" disabled>Select a column</option>
                                        {numericCols.map(col => (
                                            <option key={col} value={col}>{col}</option>
                                        ))}
                                    </select>
                                </div>
                                <div className="flex items-center gap-2">
                                    <label htmlFor="graph-y-select" className="font-semibold text-gray-700">Y-Axis:</label>
                                    <select
                                        id="graph-y-select"
                                        value={graphYCol || ''}
                                        onChange={e => setGraphYCol(e.target.value)}
                                        className="px-3 py-1.5 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                                    >
                                        <option value="" disabled>Select a column</option>
                                        {numericCols.filter(c => c !== graphXCol).map(col => (
                                            <option key={col} value={col}>{col}</option>
                                        ))}
                                    </select>
                                </div>
                            </div>
                            {graphXCol && graphYCol ? (
                                <div className="flex-grow flex items-center justify-center border border-gray-200 rounded-lg p-4">
                                    <ScatterPlot rows={rows} xCol={graphXCol} yCol={graphYCol} />
                                </div>
                            ) : (
                                <div className="flex-grow flex items-center justify-center text-gray-500">
                                    Please select both X and Y axis columns.
                                </div>
                            )}
                        </div>
                    ) : (
                        /* Detail 탭 또는 일반 모듈 */
                        <div className="flex-grow flex flex-col gap-4">
                        {/* Prep Missing 모듈의 경우 입력/출력 데이터 행/열 정보 표시 */}
                        {module.type === ModuleType.HandleMissingValues && module.outputData?.type === 'MissingHandlerOutput' ? (
                            <div className="flex-shrink-0 bg-gray-50 border border-gray-200 rounded-lg p-4 mb-4">
                                <div className="flex items-center gap-6 text-sm">
                                    <div className="flex items-center gap-2">
                                        <span className="font-semibold text-gray-700">입력 데이터:</span>
                                        <span className="text-gray-600">
                                            {data.totalRowCount.toLocaleString()}행 × {columns.length}열
                                        </span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <span className="font-semibold text-gray-700">출력 데이터:</span>
                                        <span className="text-gray-600">
                                            {module.outputData.method === 'remove_row' 
                                                ? 'remove_row'
                                                : module.outputData.method === 'impute'
                                                ? `impute (${module.outputData.strategy || 'mean'})`
                                                : module.outputData.method === 'knn'
                                                ? `knn (n_neighbors=${module.outputData.n_neighbors || 5})`
                                                : module.outputData.method
                                            }
                                            {' - '}
                                            {Math.min(sortedRows.length, 1000).toLocaleString()}행 × {columns.length}열
                                        </span>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div className="flex justify-between items-center flex-shrink-0">
                                <div className="text-sm text-gray-600">
                                    Showing {Math.min(rows.length, 1000)} of {data.totalRowCount.toLocaleString()} rows and {columns.length} columns. Click a column to see details.
                                </div>
                            </div>
                        )}
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
                            <div className={`border border-gray-200 rounded-lg ${selectedColumnData ? 'w-1/2' : 'w-full'} overflow-hidden flex flex-col`}>
                                <div className="overflow-y-auto overflow-x-auto" style={{ maxHeight: '400px' }}>
                                    <table className="min-w-full text-sm text-left">
                                        <thead className="bg-gray-50 sticky top-0 z-10">
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
                        
                        {/* Load Data/Select Data 모듈용: 상관계수 표시 (Statistics 모듈 형식) */}
                        {isDataModule && numericCols.length >= 2 && correlation && (
                            <div className="flex-shrink-0 flex flex-col gap-4">
                                <div className="border-t border-gray-200 pt-4">
                                    {/* Correlation Analysis Section */}
                                    <div>
                                        <h3 className="text-lg font-semibold mb-2 text-gray-700">Correlation Analysis</h3>
                                        <div className="overflow-x-auto border border-gray-200 rounded-lg">
                                            <CorrelationHeatmap matrix={correlation} />
                                        </div>
                                    </div>

                                    {/* Pairplot Visualization Section */}
                                    <div className="mt-4">
                                        <h3 className="text-lg font-semibold mb-2 text-gray-700">Pairplot</h3>
                                        <div className="p-4 border border-gray-200 rounded-lg">
                                            <Pairplot correlation={correlation} numericColumns={numericCols} rows={rows} />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                        </div>
                    )}
                </main>
            </div>
            {showSpreadView && rows.length > 0 && (
                <SpreadViewModal
                    onClose={() => setShowSpreadView(false)}
                    data={rows}
                    columns={columns}
                    title={`Spread View: ${module.name}`}
                />
            )}
        </div>
    );
};