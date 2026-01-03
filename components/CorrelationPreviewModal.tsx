import React, { useState, useEffect } from 'react';
import { CanvasModule, CorrelationOutput } from '../types';
import { XCircleIcon } from './icons';

interface CorrelationPreviewModalProps {
    module: CanvasModule;
    projectName: string;
    onClose: () => void;
}

export const CorrelationPreviewModal: React.FC<CorrelationPreviewModalProps> = ({ 
    module, 
    projectName, 
    onClose
}) => {
    const output = module.outputData as CorrelationOutput;
    if (!output || output.type !== 'CorrelationOutput') return null;

    const { columns, numericColumns, categoricalColumns, correlationMatrices, heatmapImage, pairplotImage, summary } = output;

    // 사용 가능한 메서드 목록
    const availableMethods = correlationMatrices?.map(m => m.method) || [];
    const defaultTab = availableMethods.length > 0 ? availableMethods[0] : "pearson";
    
    const [activeTab, setActiveTab] = useState<string>(defaultTab);
    
    // activeTab이 설정되지 않았거나 사용 불가능한 경우 기본값으로 설정
    useEffect(() => {
        // heatmap이나 pairplot이 활성화되어 있으면 유지
        if (activeTab === "heatmap" && heatmapImage) return;
        if (activeTab === "pairplot" && pairplotImage) return;
        
        // 그 외의 경우 사용 가능한 메서드로 설정
        if (!activeTab || (!availableMethods.includes(activeTab) && activeTab !== "heatmap" && activeTab !== "pairplot")) {
            setActiveTab(defaultTab);
        }
    }, [availableMethods, defaultTab, activeTab, heatmapImage, pairplotImage]);

    const getMethodLabel = (method: string): string => {
        const labels: Record<string, string> = {
            'pearson': 'Pearson Correlation',
            'spearman': 'Spearman Correlation',
            'kendall': 'Kendall Correlation',
            'cramers_v': "Cramér's V",
        };
        return labels[method] || method;
    };

    const formatValue = (value: number): string => {
        if (value === null || value === undefined) return 'N/A';
        if (Math.abs(value) < 0.0001 || Math.abs(value) > 1000000) {
            return value.toExponential(4);
        }
        return value.toFixed(4);
    };

    const getCorrelationColor = (value: number): string => {
        const absValue = Math.abs(value);
        if (absValue >= 0.7) return 'text-red-600 font-bold';
        if (absValue >= 0.5) return 'text-orange-600 font-semibold';
        if (absValue >= 0.3) return 'text-yellow-600';
        return 'text-gray-600';
    };

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
                    <h2 className="text-xl font-bold text-gray-800">Correlation Analysis: {module.name}</h2>
                    <button onClick={onClose} className="text-gray-500 hover:text-gray-800">
                        <XCircleIcon className="w-6 h-6" />
                    </button>
                </header>
                
                {/* 탭 네비게이션 */}
                <div className="flex-shrink-0 border-b border-gray-200">
                    <div className="flex overflow-x-auto">
                        {availableMethods.map((method) => {
                            const isActive = activeTab === method;
                            return (
                                <button
                                    key={method}
                                    onClick={() => setActiveTab(method)}
                                    className={`px-4 py-3 text-sm font-semibold whitespace-nowrap border-b-2 transition-colors ${
                                        isActive
                                            ? 'border-blue-500 text-blue-600 bg-blue-50'
                                            : 'border-transparent text-gray-600 hover:text-gray-800 hover:bg-gray-50'
                                    }`}
                                >
                                    {getMethodLabel(method)}
                                </button>
                            );
                        })}
                        {heatmapImage && (
                            <button
                                onClick={() => setActiveTab("heatmap")}
                                className={`px-4 py-3 text-sm font-semibold whitespace-nowrap border-b-2 transition-colors ${
                                    activeTab === "heatmap"
                                        ? 'border-blue-500 text-blue-600 bg-blue-50'
                                        : 'border-transparent text-gray-600 hover:text-gray-800 hover:bg-gray-50'
                                }`}
                            >
                                Heatmap
                            </button>
                        )}
                        {pairplotImage && (
                            <button
                                onClick={() => setActiveTab("pairplot")}
                                className={`px-4 py-3 text-sm font-semibold whitespace-nowrap border-b-2 transition-colors ${
                                    activeTab === "pairplot"
                                        ? 'border-blue-500 text-blue-600 bg-blue-50'
                                        : 'border-transparent text-gray-600 hover:text-gray-800 hover:bg-gray-50'
                                }`}
                            >
                                Pairplot
                            </button>
                        )}
                    </div>
                </div>

                <main className="flex-grow p-6 overflow-auto space-y-6">
                    {/* 요약 정보 */}
                    <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                        <h3 className="text-lg font-bold text-blue-900 mb-2">Summary</h3>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                            <div>
                                <p className="text-gray-600 mb-1">Total Columns</p>
                                <p className="font-semibold text-gray-900">{columns.length}</p>
                            </div>
                            <div>
                                <p className="text-gray-600 mb-1">Numeric Columns</p>
                                <p className="font-semibold text-gray-900">{numericColumns.length}</p>
                            </div>
                            <div>
                                <p className="text-gray-600 mb-1">Categorical Columns</p>
                                <p className="font-semibold text-gray-900">{categoricalColumns.length}</p>
                            </div>
                            <div>
                                <p className="text-gray-600 mb-1">Methods</p>
                                <p className="font-semibold text-gray-900">{correlationMatrices.length}</p>
                            </div>
                        </div>
                    </div>

                    {/* 상관계수 행렬 표시 */}
                    {activeTab !== "heatmap" && activeTab !== "pairplot" && availableMethods.includes(activeTab) && (() => {
                        const currentMatrix = correlationMatrices?.find(m => m.method === activeTab);
                        if (!currentMatrix) return null;

                        const matrix = currentMatrix.matrix;
                        const matrixColumns = currentMatrix.columns;

                        return (
                            <div>
                                <h3 className="text-lg font-bold text-gray-800 mb-3">
                                    {getMethodLabel(currentMatrix.method)} Matrix
                                </h3>
                                <div className="bg-gray-50 rounded-lg p-4 border border-gray-200 overflow-x-auto">
                                    <table className="min-w-full text-sm">
                                        <thead>
                                            <tr>
                                                <th className="px-3 py-2 text-left font-semibold text-gray-700 bg-gray-100 sticky left-0 z-10"></th>
                                                {matrixColumns.map(col => (
                                                    <th key={col} className="px-3 py-2 text-center font-semibold text-gray-700 bg-gray-100 min-w-[100px]">
                                                        {col}
                                                    </th>
                                                ))}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {matrixColumns.map((rowCol) => (
                                                <tr key={rowCol}>
                                                    <td className="px-3 py-2 font-semibold text-gray-700 bg-gray-100 sticky left-0 z-10">
                                                        {rowCol}
                                                    </td>
                                                    {matrixColumns.map((colCol) => {
                                                        const value = matrix[rowCol]?.[colCol];
                                                        return (
                                                            <td key={colCol} className={`px-3 py-2 text-center ${getCorrelationColor(value || 0)}`}>
                                                                {value !== undefined && value !== null ? formatValue(value) : 'N/A'}
                                                            </td>
                                                        );
                                                    })}
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        );
                    })()}

                    {/* Heatmap 이미지 */}
                    {activeTab === "heatmap" && (
                        <div>
                            <h3 className="text-lg font-bold text-gray-800 mb-3">Correlation Heatmap</h3>
                            {heatmapImage ? (
                                <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                                    <img 
                                        src={`data:image/png;base64,${heatmapImage}`} 
                                        alt="Correlation Heatmap"
                                        className="w-full h-auto"
                                    />
                                </div>
                            ) : (
                                <div className="bg-yellow-50 rounded-lg p-4 border border-yellow-200">
                                    <p className="text-yellow-800">Heatmap 이미지가 생성되지 않았습니다.</p>
                                </div>
                            )}
                        </div>
                    )}

                    {/* Pairplot 이미지 */}
                    {activeTab === "pairplot" && (
                        <div>
                            <h3 className="text-lg font-bold text-gray-800 mb-3">Pairplot</h3>
                            {pairplotImage ? (
                                <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                                    <img 
                                        src={`data:image/png;base64,${pairplotImage}`} 
                                        alt="Pairplot"
                                        className="w-full h-auto"
                                    />
                                </div>
                            ) : (
                                <div className="bg-yellow-50 rounded-lg p-4 border border-yellow-200">
                                    <p className="text-yellow-800">Pairplot 이미지가 생성되지 않았습니다. (5개 이하의 숫자형 열이 필요합니다)</p>
                                </div>
                            )}
                        </div>
                    )}

                    {/* 요약 통계 */}
                    {summary && Object.keys(summary).length > 0 && (
                        <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                            <h3 className="text-lg font-bold text-gray-800 mb-3">Summary Statistics</h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                {Object.entries(summary).map(([key, value]) => (
                                    <div key={key} className="bg-white rounded-lg p-3 border border-gray-200">
                                        <div className="text-sm font-semibold text-gray-700 mb-2">{key.replace(/_/g, ' ').toUpperCase()}</div>
                                        {typeof value === 'object' && value !== null ? (
                                            <div className="space-y-1 text-sm">
                                                {Object.entries(value).map(([k, v]) => (
                                                    <div key={k} className="flex justify-between">
                                                        <span className="text-gray-600">{k.replace(/_/g, ' ')}:</span>
                                                        <span className="font-mono text-gray-800">
                                                            {typeof v === 'number' ? formatValue(v) : String(v)}
                                                        </span>
                                                    </div>
                                                ))}
                                            </div>
                                        ) : (
                                            <div className="text-sm font-mono text-gray-800">
                                                {typeof value === 'number' ? formatValue(value) : String(value)}
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </main>
            </div>
        </div>
    );
};

