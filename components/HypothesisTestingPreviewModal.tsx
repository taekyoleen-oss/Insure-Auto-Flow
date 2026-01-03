import React from 'react';
import { CanvasModule, HypothesisTestingOutput } from '../types';
import { XCircleIcon } from './icons';

interface HypothesisTestingPreviewModalProps {
    module: CanvasModule;
    projectName: string;
    onClose: () => void;
}

export const HypothesisTestingPreviewModal: React.FC<HypothesisTestingPreviewModalProps> = ({ 
    module, 
    projectName, 
    onClose
}) => {
    const output = module.outputData as HypothesisTestingOutput;
    if (!output || output.type !== 'HypothesisTestingOutput') return null;

    const { results } = output;

    const getTestTypeLabel = (testType: string): string => {
        const labels: Record<string, string> = {
            't_test_one_sample': 'One-Sample t-test',
            't_test_independent': 'Independent Samples t-test',
            't_test_paired': 'Paired Samples t-test',
            'chi_square': 'Chi-square Test',
            'anova': 'One-way ANOVA',
            'ks_test': 'Kolmogorov-Smirnov Test',
            'shapiro_wilk': 'Shapiro-Wilk Test',
            'levene': 'Levene Test',
        };
        return labels[testType] || testType;
    };

    const formatValue = (value: any): string => {
        if (value === null || value === undefined) return 'N/A';
        if (typeof value === 'number') {
            if (Math.abs(value) < 0.0001 || Math.abs(value) > 1000000) {
                return value.toExponential(4);
            }
            return value.toFixed(4);
        }
        if (Array.isArray(value)) {
            return `[${value.map(v => formatValue(v)).join(', ')}]`;
        }
        return String(value);
    };

    const getConclusionColor = (conclusion?: string): string => {
        if (!conclusion) return 'text-gray-400';
        if (conclusion.includes('Reject')) return 'text-red-400';
        if (conclusion.includes('Fail to reject')) return 'text-green-400';
        return 'text-gray-400';
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
                    <h2 className="text-xl font-bold text-gray-800">Hypothesis Testing: {module.name}</h2>
                    <button onClick={onClose} className="text-gray-500 hover:text-gray-800">
                        <XCircleIcon className="w-6 h-6" />
                    </button>
                </header>
                
                <main className="flex-grow p-6 overflow-auto space-y-6">
                    {results.length === 0 ? (
                        <div className="text-center text-gray-500 p-8">
                            <p>No test results available.</p>
                        </div>
                    ) : (
                        results.map((result, index) => (
                            <div key={index} className="bg-gray-50 rounded-lg p-6 border border-gray-200">
                                <div className="mb-4">
                                    <h3 className="text-lg font-bold text-gray-800 mb-2">
                                        {result.testName || getTestTypeLabel(result.testType)}
                                    </h3>
                                    <div className="text-sm text-gray-600">
                                        <span className="font-semibold">Columns:</span> {result.columns.join(', ')}
                                    </div>
                                </div>

                                {/* 주요 결과 */}
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                                    {result.statistic !== undefined && (
                                        <div className="bg-white rounded-lg p-3 border border-gray-200">
                                            <div className="text-xs text-gray-500 mb-1">Statistic</div>
                                            <div className="text-lg font-semibold text-gray-900">
                                                {formatValue(result.statistic)}
                                            </div>
                                        </div>
                                    )}
                                    {result.pValue !== undefined && (
                                        <div className="bg-white rounded-lg p-3 border border-gray-200">
                                            <div className="text-xs text-gray-500 mb-1">p-value</div>
                                            <div className={`text-lg font-semibold ${
                                                result.pValue < 0.05 ? 'text-red-600' : 'text-green-600'
                                            }`}>
                                                {formatValue(result.pValue)}
                                            </div>
                                        </div>
                                    )}
                                    {result.degreesOfFreedom !== undefined && (
                                        <div className="bg-white rounded-lg p-3 border border-gray-200">
                                            <div className="text-xs text-gray-500 mb-1">Degrees of Freedom</div>
                                            <div className="text-lg font-semibold text-gray-900">
                                                {formatValue(result.degreesOfFreedom)}
                                            </div>
                                        </div>
                                    )}
                                    {result.criticalValue !== undefined && (
                                        <div className="bg-white rounded-lg p-3 border border-gray-200">
                                            <div className="text-xs text-gray-500 mb-1">Critical Value</div>
                                            <div className="text-lg font-semibold text-gray-900">
                                                {formatValue(result.criticalValue)}
                                            </div>
                                        </div>
                                    )}
                                </div>

                                {/* 결론 */}
                                {result.conclusion && (
                                    <div className={`mb-4 p-3 rounded-lg border ${
                                        result.conclusion.includes('Reject')
                                            ? 'bg-red-50 border-red-200'
                                            : 'bg-green-50 border-green-200'
                                    }`}>
                                        <div className="text-sm font-semibold text-gray-700 mb-1">Conclusion</div>
                                        <div className={getConclusionColor(result.conclusion)}>
                                            {result.conclusion}
                                        </div>
                                    </div>
                                )}

                                {/* 해석 */}
                                {result.interpretation && (
                                    <div className="mb-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
                                        <div className="text-sm font-semibold text-blue-900 mb-1">Interpretation</div>
                                        <div className="text-sm text-blue-800">{result.interpretation}</div>
                                    </div>
                                )}

                                {/* 상세 정보 */}
                                {result.details && Object.keys(result.details).length > 0 && (
                                    <div className="mt-4">
                                        <div className="text-sm font-semibold text-gray-700 mb-2">Details</div>
                                        <div className="bg-white rounded-lg p-3 border border-gray-200">
                                            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm">
                                                {Object.entries(result.details).map(([key, value]) => (
                                                    <div key={key} className="flex justify-between">
                                                        <span className="text-gray-600">{key}:</span>
                                                        <span className="font-mono text-gray-800 text-right">
                                                            {formatValue(value)}
                                                        </span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {/* 에러 표시 */}
                                {result.testName && result.testName.startsWith('Error:') && (
                                    <div className="mt-4 p-3 bg-red-50 rounded-lg border border-red-200">
                                        <div className="text-sm font-semibold text-red-900 mb-1">Error</div>
                                        <div className="text-sm text-red-800">{result.testName}</div>
                                    </div>
                                )}
                            </div>
                        ))
                    )}
                </main>
            </div>
        </div>
    );
};


