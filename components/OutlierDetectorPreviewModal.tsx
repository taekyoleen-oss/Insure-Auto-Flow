import React, { useState, useMemo } from 'react';
import { CanvasModule, OutlierDetectorOutput, DataPreview } from '../types';
import { XCircleIcon, CheckIcon } from './icons';

interface OutlierDetectorPreviewModalProps {
    module: CanvasModule;
    projectName: string;
    onClose: () => void;
    onRemoveOutliers?: (column: string, indices: number[]) => void;
    onUpdateData?: (data: Record<string, any>[]) => void;
}

export const OutlierDetectorPreviewModal: React.FC<OutlierDetectorPreviewModalProps> = ({ 
    module, 
    projectName, 
    onClose,
    onRemoveOutliers,
    onUpdateData
}) => {
    const [activeTab, setActiveTab] = useState<string>("");
    const [selectedMethod, setSelectedMethod] = useState<string | null>(null);
    const [selectedIndices, setSelectedIndices] = useState<Set<number>>(new Set());
    const [showRemoveConfirm, setShowRemoveConfirm] = useState(false);
    const [pendingRemove, setPendingRemove] = useState<{ column: string; indices: number[] } | null>(null);

    // OutlierDetectorOutput이 outputData에 있거나 parameters에 저장되어 있는지 확인
    let output: OutlierDetectorOutput | null = null;
    if (module.outputData && module.outputData.type === 'OutlierDetectorOutput') {
      output = module.outputData as OutlierDetectorOutput;
    } else if (module.parameters._outlierOutput) {
      output = module.parameters._outlierOutput as OutlierDetectorOutput;
    }

    if (!output) return null;

    const { columns, columnResults, totalOutliers, allOutlierIndices, cleanedData, originalData } = output;

    // 첫 번째 열을 기본 탭으로 설정
    React.useEffect(() => {
        if (columns.length > 0 && !activeTab) {
            setActiveTab(columns[0]);
        }
    }, [columns, activeTab]);

    // 현재 활성 탭의 결과 가져오기
    const currentColumnResult = useMemo(() => {
        return columnResults.find(r => r.column === activeTab);
    }, [columnResults, activeTab]);

    // 원본 데이터에서 현재 열의 값 가져오기
    const getColumnValue = (index: number, column: string): string => {
        if (originalData && originalData[index]) {
            const value = originalData[index][column];
            return value !== undefined && value !== null ? String(value) : 'N/A';
        }
        return 'N/A';
    };

    const handleSelectMethod = (method: string) => {
        setSelectedMethod(method);
        if (currentColumnResult) {
            const methodResult = currentColumnResult.results.find(r => r.method === method);
            if (methodResult) {
                setSelectedIndices(new Set(methodResult.outlierIndices));
            }
        }
    };

    const handleToggleIndex = (index: number) => {
        const newSet = new Set(selectedIndices);
        if (newSet.has(index)) {
            newSet.delete(index);
        } else {
            newSet.add(index);
        }
        setSelectedIndices(newSet);
    };

    const handleRemoveSelected = () => {
        if (selectedIndices.size === 0 || !activeTab) return;
        setPendingRemove({ column: activeTab, indices: Array.from(selectedIndices) });
        setShowRemoveConfirm(true);
    };

    const confirmRemove = () => {
        if (pendingRemove && onRemoveOutliers) {
            onRemoveOutliers(pendingRemove.column, pendingRemove.indices);
            setShowRemoveConfirm(false);
            setPendingRemove(null);
            setSelectedIndices(new Set());
        }
    };

    const handleRemoveAll = () => {
        if (currentColumnResult && currentColumnResult.outlierIndices.length > 0 && activeTab) {
            setPendingRemove({ column: activeTab, indices: currentColumnResult.outlierIndices });
            setShowRemoveConfirm(true);
        }
    };

    const confirmRemoveAll = () => {
        if (pendingRemove && onRemoveOutliers) {
            onRemoveOutliers(pendingRemove.column, pendingRemove.indices);
            setShowRemoveConfirm(false);
            setPendingRemove(null);
        }
    };

    // 탭 변경 시 선택 초기화
    const handleTabChange = (column: string) => {
        setActiveTab(column);
        setSelectedMethod(null);
        setSelectedIndices(new Set());
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
                    <h2 className="text-xl font-bold text-gray-800">Outlier Detection: {module.name}</h2>
                    <button onClick={onClose} className="text-gray-500 hover:text-gray-800">
                        <XCircleIcon className="w-6 h-6" />
                    </button>
                </header>
                
                {/* 탭 네비게이션 */}
                <div className="flex-shrink-0 border-b border-gray-200">
                    <div className="flex overflow-x-auto">
                        {columns.map((column) => {
                            const colResult = columnResults.find(r => r.column === column);
                            const isActive = activeTab === column;
                            return (
                                <button
                                    key={column}
                                    onClick={() => handleTabChange(column)}
                                    className={`px-4 py-3 text-sm font-semibold whitespace-nowrap border-b-2 transition-colors ${
                                        isActive
                                            ? 'border-blue-500 text-blue-600 bg-blue-50'
                                            : 'border-transparent text-gray-600 hover:text-gray-800 hover:bg-gray-50'
                                    }`}
                                >
                                    {column}
                                    {colResult && colResult.totalOutliers > 0 && (
                                        <span className={`ml-2 px-2 py-0.5 rounded-full text-xs ${
                                            isActive ? 'bg-blue-100 text-blue-700' : 'bg-gray-200 text-gray-700'
                                        }`}>
                                            {colResult.totalOutliers}
                                        </span>
                                    )}
                                </button>
                            );
                        })}
                    </div>
                </div>

                <main className="flex-grow p-6 overflow-auto space-y-6">
                    {currentColumnResult ? (
                        <>
                            {/* 요약 정보 */}
                            <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                                <h3 className="text-lg font-bold text-blue-900 mb-2">Summary - {activeTab}</h3>
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                                    <div>
                                        <p className="text-gray-600 mb-1">Column</p>
                                        <p className="font-semibold text-gray-900">{activeTab}</p>
                                    </div>
                                    <div>
                                        <p className="text-gray-600 mb-1">Total Outliers</p>
                                        <p className="font-semibold text-gray-900">{currentColumnResult.totalOutliers}</p>
                                    </div>
                                    <div>
                                        <p className="text-gray-600 mb-1">Total Rows</p>
                                        <p className="font-semibold text-gray-900">
                                            {originalData ? originalData.length : 'N/A'}
                                        </p>
                                    </div>
                                    <div>
                                        <p className="text-gray-600 mb-1">Outlier %</p>
                                        <p className="font-semibold text-gray-900">
                                            {originalData ? (currentColumnResult.totalOutliers / originalData.length * 100).toFixed(2) : 'N/A'}%
                                        </p>
                                    </div>
                                </div>
                            </div>

                            {/* 방법별 결과 */}
                            <div>
                                <h3 className="text-lg font-bold text-gray-800 mb-3">Detection Methods</h3>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    {currentColumnResult.results.map((result) => (
                                        <div 
                                            key={result.method}
                                            className={`border rounded-lg p-4 cursor-pointer transition-colors ${
                                                selectedMethod === result.method
                                                    ? 'border-blue-500 bg-blue-50'
                                                    : 'border-gray-200 hover:border-gray-300'
                                            }`}
                                            onClick={() => handleSelectMethod(result.method)}
                                        >
                                            <div className="flex items-center justify-between mb-2">
                                                <h4 className="font-semibold text-gray-800">{result.method}</h4>
                                                {selectedMethod === result.method && (
                                                    <CheckIcon className="w-5 h-5 text-blue-500" />
                                                )}
                                            </div>
                                            <div className="text-sm text-gray-600 space-y-1">
                                                <p>Outliers: {result.outlierCount} ({result.outlierPercentage.toFixed(2)}%)</p>
                                                {result.details && (
                                                    <div className="mt-2 text-xs bg-gray-100 p-2 rounded">
                                                        {Object.entries(result.details).map(([key, value]) => (
                                                            <div key={key} className="flex justify-between">
                                                                <span className="text-gray-600">{key}:</span>
                                                                <span className="font-mono text-gray-800">
                                                                    {typeof value === 'number' ? value.toFixed(4) : String(value)}
                                                                </span>
                                                            </div>
                                                        ))}
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* 선택된 이상치 목록 */}
                            {selectedMethod && selectedIndices.size > 0 && (
                                <div>
                                    <div className="flex items-center justify-between mb-3">
                                        <h3 className="text-lg font-bold text-gray-800">
                                            Selected Outliers ({selectedIndices.size})
                                        </h3>
                                        <div className="flex gap-2">
                                            <button
                                                onClick={handleRemoveSelected}
                                                className="px-4 py-2 text-sm font-semibold text-white bg-red-600 rounded-lg hover:bg-red-700 transition-colors"
                                            >
                                                Remove Selected
                                            </button>
                                        </div>
                                    </div>
                                    <div className="bg-gray-50 rounded-lg p-4 border border-gray-200 max-h-60 overflow-y-auto">
                                        <div className="grid grid-cols-5 gap-2 text-xs font-semibold text-gray-600 mb-2 pb-2 border-b">
                                            <div>Index</div>
                                            <div>Value</div>
                                            <div>Method</div>
                                            <div>Select</div>
                                            <div>Action</div>
                                        </div>
                                        {Array.from(selectedIndices).sort((a, b) => a - b).map((idx) => {
                                            const methodResult = currentColumnResult.results.find(r => r.outlierIndices.includes(idx));
                                            return (
                                                <div key={idx} className="grid grid-cols-5 gap-2 text-xs py-1 border-b border-gray-100">
                                                    <div className="font-mono">{idx}</div>
                                                    <div className="font-mono">
                                                        {getColumnValue(idx, activeTab)}
                                                    </div>
                                                    <div>{methodResult?.method || 'Multiple'}</div>
                                                    <div>
                                                        <input
                                                            type="checkbox"
                                                            checked={selectedIndices.has(idx)}
                                                            onChange={() => handleToggleIndex(idx)}
                                                            className="cursor-pointer"
                                                        />
                                                    </div>
                                                    <div>
                                                        <button
                                                            onClick={() => {
                                                                const newSet = new Set(selectedIndices);
                                                                newSet.delete(idx);
                                                                setSelectedIndices(newSet);
                                                            }}
                                                            className="text-red-600 hover:text-red-800"
                                                        >
                                                            Remove
                                                        </button>
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            )}

                            {/* 전체 이상치 제거 */}
                            {currentColumnResult.outlierIndices.length > 0 && (
                                <div className="bg-yellow-50 rounded-lg p-4 border border-yellow-200">
                                    <div className="flex items-center justify-between">
                                        <div>
                                            <h3 className="text-lg font-bold text-yellow-900 mb-1">Remove All Outliers</h3>
                                            <p className="text-sm text-yellow-800">
                                                Remove all {currentColumnResult.outlierIndices.length} outliers detected by any method for column "{activeTab}"
                                            </p>
                                        </div>
                                        <button
                                            onClick={handleRemoveAll}
                                            className="px-4 py-2 text-sm font-semibold text-white bg-yellow-600 rounded-lg hover:bg-yellow-700 transition-colors"
                                        >
                                            Remove All
                                        </button>
                                    </div>
                                </div>
                            )}
                        </>
                    ) : (
                        <div className="text-center text-gray-500 p-8">
                            <p>No data available for selected column.</p>
                        </div>
                    )}

                    {/* 전체 요약 */}
                    <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                        <h3 className="text-lg font-bold text-gray-800 mb-2">Overall Summary</h3>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                            <div>
                                <p className="text-gray-600 mb-1">Total Columns Analyzed</p>
                                <p className="font-semibold text-gray-900">{columns.length}</p>
                            </div>
                            <div>
                                <p className="text-gray-600 mb-1">Total Outliers (All Columns)</p>
                                <p className="font-semibold text-gray-900">{totalOutliers}</p>
                            </div>
                            <div>
                                <p className="text-gray-600 mb-1">Unique Outlier Rows</p>
                                <p className="font-semibold text-gray-900">{allOutlierIndices.length}</p>
                            </div>
                        </div>
                    </div>

                    {/* 확인 다이얼로그 */}
                    {showRemoveConfirm && pendingRemove && (
                        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                            <div className="bg-white rounded-lg p-6 max-w-md">
                                <h3 className="text-lg font-bold text-gray-800 mb-2">Confirm Removal</h3>
                                <p className="text-sm text-gray-600 mb-4">
                                    Are you sure you want to remove {pendingRemove.indices.length} outlier(s) from column "{pendingRemove.column}"?
                                    <br />
                                    This will remove the entire rows from the output table.
                                    <br />
                                    This action cannot be undone.
                                </p>
                                <div className="flex gap-2 justify-end">
                                    <button
                                        onClick={() => {
                                            setShowRemoveConfirm(false);
                                            setPendingRemove(null);
                                        }}
                                        className="px-4 py-2 text-sm font-semibold text-gray-700 bg-gray-200 rounded-lg hover:bg-gray-300 transition-colors"
                                    >
                                        Cancel
                                    </button>
                                    <button
                                        onClick={confirmRemove}
                                        className="px-4 py-2 text-sm font-semibold text-white bg-red-600 rounded-lg hover:bg-red-700 transition-colors"
                                    >
                                        Remove
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}
                </main>
            </div>
        </div>
    );
};
