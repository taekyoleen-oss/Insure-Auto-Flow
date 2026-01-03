import React, { useState, useEffect } from 'react';
import { CanvasModule, TrainedModelOutput, ModuleType } from '../types';
import { XCircleIcon, SparklesIcon } from './icons';
import { GoogleGenAI } from "@google/genai";
import { MarkdownRenderer } from './MarkdownRenderer';

interface TrainedModelPreviewModalProps {
    module: CanvasModule;
    projectName: string;
    onClose: () => void;
}

const ModalModelMetrics: React.FC<{ metrics: TrainedModelOutput['metrics'] }> = ({ metrics }) => (
    <div>
        <h4 className="text-md font-semibold text-gray-700 mb-2">Evaluation Metrics</h4>
        <div className="bg-gray-50 rounded-lg p-3 space-y-2 border border-gray-200">
            {Object.entries(metrics).map(([key, value]) => (
                <div key={key} className="flex justify-between items-center text-sm">
                    <span className="text-gray-600">{key}:</span>
                    <span className="font-mono text-gray-800 font-medium">{typeof value === 'number' ? value.toFixed(4) : value}</span>
                </div>
            ))}
        </div>
    </div>
);

interface TuningChartProps {
    candidates: Array<{ params: Record<string, number>; score: number }>;
    scoringMetric?: string;
}

const TuningChart: React.FC<TuningChartProps> = ({ candidates, scoringMetric }) => {
    if (!candidates || candidates.length === 0) return null;

    // alpha 또는 C 파라미터와 score 추출
    const dataPoints = candidates
        .map(candidate => {
            const paramValue = candidate.params.alpha ?? candidate.params.C;
            // score가 음수일 수 있으므로 절댓값 처리 (neg_mean_squared_error 등)
            const score = Math.abs(candidate.score);
            return { paramValue, score };
        })
        .filter(point => typeof point.paramValue === 'number' && !isNaN(point.paramValue) && typeof point.score === 'number' && !isNaN(point.score))
        .sort((a, b) => a.paramValue - b.paramValue);

    if (dataPoints.length === 0) return null;
    
    // 파라미터 이름 결정 (alpha 또는 C)
    const paramName = candidates[0]?.params.alpha !== undefined ? 'alpha' : 'C';

    // 차트 크기
    const width = 600;
    const height = 300;
    const padding = { top: 40, right: 60, bottom: 50, left: 60 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // 데이터 범위 계산
    const paramMin = Math.min(...dataPoints.map(p => p.paramValue));
    const paramMax = Math.max(...dataPoints.map(p => p.paramValue));
    const scoreMin = Math.min(...dataPoints.map(p => p.score));
    const scoreMax = Math.max(...dataPoints.map(p => p.score));

    // 스케일 계산 (로그 스케일을 위한 변환)
    const paramRange = paramMax - paramMin || 1;
    const scoreRange = scoreMax - scoreMin || 1;

    // 좌표 변환 함수
    const scaleX = (paramValue: number) => {
        // 로그 스케일 적용
        const logMin = Math.log10(Math.max(paramMin, 0.001));
        const logMax = Math.log10(Math.max(paramMax, 0.001));
        const logParam = Math.log10(Math.max(paramValue, 0.001));
        return padding.left + ((logParam - logMin) / (logMax - logMin)) * chartWidth;
    };

    const scaleY = (score: number) => {
        return padding.top + chartHeight - ((score - scoreMin) / scoreRange) * chartHeight;
    };

    // 경로 생성
    const pathData = dataPoints
        .map((point, idx) => {
            const x = scaleX(point.paramValue);
            const y = scaleY(point.score);
            return idx === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
        })
        .join(' ');

    // X축 눈금 (로그 스케일)
    const xTicks: number[] = [];
    const logMin = Math.log10(Math.max(paramMin, 0.001));
    const logMax = Math.log10(Math.max(paramMax, 0.001));
    const logRange = logMax - logMin;
    const numTicks = 5;
    for (let i = 0; i <= numTicks; i++) {
        const logValue = logMin + (logRange * i) / numTicks;
        xTicks.push(Math.pow(10, logValue));
    }

    // Y축 눈금
    const numYTicks = 5;
    const yTicks: number[] = [];
    for (let i = 0; i <= numYTicks; i++) {
        yTicks.push(scoreMin + (scoreRange * i) / numYTicks);
    }

    // Y축 라벨 (MAE 또는 scoring metric)
    const yLabel = scoringMetric?.includes('MAE') || scoringMetric?.includes('mean_absolute') 
        ? 'MAE' 
        : scoringMetric?.includes('MSE') || scoringMetric?.includes('mean_squared')
        ? 'MSE'
        : scoringMetric?.includes('neg_')
        ? scoringMetric.replace('neg_', '').toUpperCase()
        : 'Score';

    return (
        <div className="mt-4">
            <p className="text-gray-600 mb-2">Hyperparameter Performance</p>
            <div className="bg-white rounded-lg border border-blue-200 p-4">
                <svg width={width} height={height} className="w-full" viewBox={`0 0 ${width} ${height}`}>
                    {/* 그리드 라인 */}
                    {xTicks.map((tick, idx) => {
                        const x = scaleX(tick);
                        return (
                            <g key={`x-grid-${idx}`}>
                                <line
                                    x1={x}
                                    y1={padding.top}
                                    x2={x}
                                    y2={height - padding.bottom}
                                    stroke="#e5e7eb"
                                    strokeWidth="1"
                                    strokeDasharray="2,2"
                                />
                            </g>
                        );
                    })}
                    {yTicks.map((tick, idx) => {
                        const y = scaleY(tick);
                        return (
                            <g key={`y-grid-${idx}`}>
                                <line
                                    x1={padding.left}
                                    y1={y}
                                    x2={width - padding.right}
                                    y2={y}
                                    stroke="#e5e7eb"
                                    strokeWidth="1"
                                    strokeDasharray="2,2"
                                />
                            </g>
                        );
                    })}

                    {/* 데이터 라인 */}
                    <path
                        d={pathData}
                        fill="none"
                        stroke="#ef4444"
                        strokeWidth="2"
                        strokeDasharray="5,5"
                    />

                    {/* 데이터 포인트 */}
                    {dataPoints.map((point, idx) => {
                        const x = scaleX(point.paramValue);
                        const y = scaleY(point.score);
                        return (
                            <circle
                                key={idx}
                                cx={x}
                                cy={y}
                                r="5"
                                fill="#3b82f6"
                                stroke="white"
                                strokeWidth="2"
                            />
                        );
                    })}

                    {/* X축 */}
                    <line
                        x1={padding.left}
                        y1={height - padding.bottom}
                        x2={width - padding.right}
                        y2={height - padding.bottom}
                        stroke="#374151"
                        strokeWidth="2"
                    />
                    {/* X축 라벨 */}
                    {xTicks.map((tick, idx) => {
                        const x = scaleX(tick);
                        return (
                            <g key={`x-tick-${idx}`}>
                                <line
                                    x1={x}
                                    y1={height - padding.bottom}
                                    x2={x}
                                    y2={height - padding.bottom + 5}
                                    stroke="#374151"
                                    strokeWidth="2"
                                />
                                <text
                                    x={x}
                                    y={height - padding.bottom + 20}
                                    textAnchor="middle"
                                    fontSize="12"
                                    fill="#374151"
                                    className="font-mono"
                                >
                                    {tick < 0.1 ? tick.toFixed(2) : tick < 1 ? tick.toFixed(1) : tick.toFixed(0)}
                                </text>
                            </g>
                        );
                    })}
                    <text
                        x={width / 2}
                        y={height - 10}
                        textAnchor="middle"
                        fontSize="14"
                        fill="#374151"
                        fontWeight="bold"
                    >
                        {paramName === 'alpha' ? 'α' : 'C'}
                    </text>

                    {/* Y축 */}
                    <line
                        x1={padding.left}
                        y1={padding.top}
                        x2={padding.left}
                        y2={height - padding.bottom}
                        stroke="#374151"
                        strokeWidth="2"
                    />
                    {/* Y축 라벨 */}
                    {yTicks.map((tick, idx) => {
                        const y = scaleY(tick);
                        return (
                            <g key={`y-tick-${idx}`}>
                                <line
                                    x1={padding.left}
                                    y1={y}
                                    x2={padding.left - 5}
                                    y2={y}
                                    stroke="#374151"
                                    strokeWidth="2"
                                />
                                <text
                                    x={padding.left - 10}
                                    y={y + 4}
                                    textAnchor="end"
                                    fontSize="12"
                                    fill="#374151"
                                    className="font-mono"
                                >
                                    {tick.toFixed(3)}
                                </text>
                            </g>
                        );
                    })}
                    <text
                        x={15}
                        y={height / 2}
                        textAnchor="middle"
                        fontSize="14"
                        fill="#374151"
                        fontWeight="bold"
                        transform={`rotate(-90, 15, ${height / 2})`}
                    >
                        {yLabel}
                    </text>

                    {/* "예상 성능" 텍스트 */}
                    <text
                        x={width - padding.right - 10}
                        y={padding.top + 20}
                        textAnchor="end"
                        fontSize="12"
                        fill="#6b7280"
                        className="font-sans"
                    >
                        예상 성능
                    </text>
                </svg>
            </div>
        </div>
    );
};

// fix: Corrected ModuleType enums to be more specific (e.g., DecisionTreeClassifier) to match the type definitions.
const complexModels = [
    ModuleType.DecisionTree,
    ModuleType.RandomForest,
    ModuleType.NeuralNetwork,
    ModuleType.SVM,
    ModuleType.KNN,
    ModuleType.NaiveBayes,
    ModuleType.LinearDiscriminantAnalysis
];

export const TrainedModelPreviewModal: React.FC<TrainedModelPreviewModalProps> = ({ module, projectName, onClose }) => {
    const [isInterpreting, setIsInterpreting] = useState(false);
    const [aiInterpretation, setAiInterpretation] = useState<string | null>(null);
    const [plotTreeImage, setPlotTreeImage] = useState<string | null>(null);
    const [isLoadingPlot, setIsLoadingPlot] = useState(false);
    const [treeText, setTreeText] = useState<string | null>(null);
    const [isLoadingText, setIsLoadingText] = useState(false);
    const [showTreeImage, setShowTreeImage] = useState(false);

    const output = module.outputData as TrainedModelOutput;
    if (!output || output.type !== 'TrainedModelOutput') return null;

    const { modelType, coefficients, intercept, metrics, featureColumns, labelColumn, tuningSummary, trainingData, modelParameters } = output;
    
    // Decision Tree인 경우 export_text 생성 (기본 표시)
    useEffect(() => {
        if (modelType === ModuleType.DecisionTree && trainingData && modelParameters && featureColumns && labelColumn) {
            setIsLoadingText(true);
            const generateText = async () => {
                try {
                    const pyodideModule = await import('../utils/pyodideRunner');
                    const { generateDecisionTreeText } = pyodideModule;
                    
                    const text = await generateDecisionTreeText(
                        trainingData,
                        featureColumns,
                        labelColumn,
                        output.modelPurpose || "classification",
                        modelParameters.criterion || "gini",
                        modelParameters.maxDepth || null,
                        modelParameters.minSamplesSplit || 2,
                        modelParameters.minSamplesLeaf || 1,
                        modelParameters.classWeight || null
                    );
                    
                    setTreeText(text);
                } catch (error: any) {
                    console.error("Failed to generate Decision Tree text:", error);
                    console.error("Error details:", error?.message, error?.stack);
                    setTreeText(null);
                } finally {
                    setIsLoadingText(false);
                }
            };
            
            generateText();
        }
    }, [modelType, trainingData, modelParameters, featureColumns, labelColumn, output.modelPurpose]);

    // Decision Tree 이미지 생성 (버튼 클릭 시)
    const handleShowTreeImage = async () => {
        if (plotTreeImage) {
            setShowTreeImage(true);
            return;
        }

        if (modelType === ModuleType.DecisionTree && trainingData && modelParameters && featureColumns && labelColumn) {
            setIsLoadingPlot(true);
            try {
                const pyodideModule = await import('../utils/pyodideRunner');
                const { generateDecisionTreePlot } = pyodideModule;
                
                const imageBase64 = await generateDecisionTreePlot(
                    trainingData,
                    featureColumns,
                    labelColumn,
                    output.modelPurpose || "classification",
                    modelParameters.criterion || "gini",
                    modelParameters.maxDepth || null,
                    modelParameters.minSamplesSplit || 2,
                    modelParameters.minSamplesLeaf || 1,
                    modelParameters.classWeight || null
                );
                
                setPlotTreeImage(`data:image/png;base64,${imageBase64}`);
                setShowTreeImage(true);
            } catch (error) {
                console.error("Failed to generate Decision Tree plot:", error);
                setPlotTreeImage(null);
            } finally {
                setIsLoadingPlot(false);
            }
        }
    };

    const handleInterpret = async () => {
        setIsInterpreting(true);
        setAiInterpretation(null);
        try {
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });

            const metricsText = Object.entries(metrics).map(([key, value]) => `- ${key}: ${typeof value === 'number' ? value.toFixed(4) : value}`).join('\n');
            const topFeatures = Object.entries(coefficients)
                .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
                .slice(0, 3)
                .map(([key, value]) => `- ${key}: ${value.toFixed(4)}`)
                .join('\n');

            const prompt = `
You are a data scientist writing a brief report for a non-technical audience. Please use Korean and simple Markdown.

### 머신러닝 모델 분석 보고서

**프로젝트:** ${projectName}
**모델:** ${modelType}
**분석 대상:** ${labelColumn}

**성능 (학습 데이터 기준):**
${metricsText}

**주요 영향 변수 (상위 3개):**
${topFeatures}

---

**1. 모델 성능 요약:**
- 이 모델의 성능을 주요 지표(예: Accuracy, R-squared)를 사용하여 한 문장으로 요약해 주십시오.

**2. 핵심 발견:**
- 분석 대상에 가장 큰 영향을 미치는 변수는 무엇이며, 이 변수가 결과에 긍정적인 영향을 미칩니까, 부정적인 영향을 미치습니까?

**3. 모델 활용 방안:**
- 이 모델을 비즈니스에 어떻게 활용할 수 있을지 간단한 아이디어 한 가지를 제안해 주십시오.

**지시:** 각 항목을 한두 문장으로 매우 간결하게 작성하십시오.
`;
            const response = await ai.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: prompt,
            });

            setAiInterpretation(response.text);
        } catch (error) {
            console.error("AI interpretation failed:", error);
            setAiInterpretation("결과를 해석하는 동안 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.");
        } finally {
            setIsInterpreting(false);
        }
    };
    
    let formulaParts: string[] = [];
    if (!complexModels.includes(modelType)) {
        if (modelType === ModuleType.LogisticRegression) {
            formulaParts = [`ln(p / (1 - p)) = ${intercept.toFixed(4)}`];
        } else {
            formulaParts = [`${labelColumn} ≈ ${intercept.toFixed(4)}`];
        }

        featureColumns.forEach(feature => {
            const value = coefficients[feature];
            const coeff = typeof value === 'number' ? value : 0;
            if (coeff >= 0) {
                formulaParts.push(` + ${coeff.toFixed(4)} * [${feature}]`);
            } else {
                formulaParts.push(` - ${Math.abs(coeff).toFixed(4)} * [${feature}]`);
            }
        });
    }


    return (
        <div 
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
            onClick={onClose}
        >
            <div 
                className="bg-white text-gray-900 rounded-lg shadow-xl w-full max-w-3xl max-h-[90vh] flex flex-col"
                onClick={e => e.stopPropagation()}
            >
                <header className="flex items-center justify-between p-4 border-b border-gray-200 flex-shrink-0">
                    <h2 className="text-xl font-bold text-gray-800">Trained Model Details: {module.name}</h2>
                    <button onClick={onClose} className="text-gray-500 hover:text-gray-800">
                        <XCircleIcon className="w-6 h-6" />
                    </button>
                </header>
                <main className="flex-grow p-6 overflow-auto space-y-6">
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

                    {isInterpreting && (
                        <div className="text-center p-8 text-gray-600 font-sans">
                            <p>AI가 모델 결과를 분석하고 있습니다...</p>
                        </div>
                    )}
                    {aiInterpretation && (
                         <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                            <h3 className="text-lg font-bold text-purple-800 mb-2 font-sans flex items-center gap-2">
                                <SparklesIcon className="w-5 h-5"/>
                                AI 분석 요약
                            </h3>
                            <MarkdownRenderer text={aiInterpretation} />
                        </div>
                    )}

                    {tuningSummary?.enabled && (
                        <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                            <h3 className="text-lg font-bold text-blue-900 mb-3">Hyperparameter Tuning Results</h3>
                            <div className="grid md:grid-cols-2 gap-4 text-sm">
                                <div>
                                    <p className="text-gray-600 mb-1">Strategy</p>
                                    <p className="font-semibold text-gray-900">{tuningSummary.strategy ?? 'grid'}</p>
                                </div>
                                {tuningSummary.scoringMetric && (
                                    <div>
                                        <p className="text-gray-600 mb-1">Scoring Metric</p>
                                        <p className="font-semibold text-gray-900">{tuningSummary.scoringMetric}</p>
                                    </div>
                                )}
                                {typeof tuningSummary.bestScore === 'number' && (
                                    <div>
                                        <p className="text-gray-600 mb-1">Best Score</p>
                                        <p className="font-semibold text-gray-900">{tuningSummary.bestScore.toFixed(4)}</p>
                                    </div>
                                )}
                            </div>
                            {tuningSummary.bestParams && (
                                <div className="mt-4">
                                    <p className="text-gray-600 mb-2">Best Parameters</p>
                                    <div className="flex flex-wrap gap-2">
                                        {Object.entries(tuningSummary.bestParams).map(([key, value]) => (
                                            <span key={key} className="px-3 py-1 bg-white rounded-full text-xs font-semibold text-gray-700 border border-blue-200">
                                                {key}: {typeof value === 'number' ? value.toFixed(4) : value}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            )}
                            {tuningSummary.candidates && tuningSummary.candidates.length > 0 && (
                                <>
                                    <div className="mt-4">
                                        <p className="text-gray-600 mb-2">Top Candidates</p>
                                        <div className="bg-white rounded-lg border border-blue-100 max-h-48 overflow-y-auto">
                                            <table className="w-full text-xs">
                                                <thead className="bg-blue-100 text-blue-900">
                                                    <tr>
                                                        <th className="p-2 text-left">Parameters</th>
                                                        <th className="p-2 text-right">Score</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {tuningSummary.candidates.slice(0, 5).map((candidate, idx) => (
                                                        <tr key={idx} className="border-t border-blue-100">
                                                            <td className="p-2">
                                                                <div className="flex flex-wrap gap-1">
                                                                    {Object.entries(candidate.params).map(([paramKey, paramValue]) => (
                                                                        <span key={paramKey} className="px-2 py-0.5 bg-blue-50 rounded text-blue-900">
                                                                            {paramKey}: {typeof paramValue === 'number' ? paramValue.toFixed(3) : paramValue}
                                                                        </span>
                                                                    ))}
                                                                </div>
                                                            </td>
                                                            <td className="p-2 text-right font-mono">{candidate.score.toFixed(4)}</td>
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                    <TuningChart 
                                        candidates={tuningSummary.candidates} 
                                        scoringMetric={tuningSummary.scoringMetric}
                                    />
                                </>
                            )}
                        </div>
                    )}

                    {formulaParts.length > 0 && (
                        <div>
                            <h4 className="text-md font-semibold text-gray-700 mb-2">Model Equation</h4>
                            <div className="bg-gray-50 rounded-lg p-3 font-mono text-xs text-green-700 whitespace-normal break-words border border-gray-200">
                                <span>{formulaParts[0]}</span>
                                {formulaParts.slice(1).map((part, i) => <span key={i}>{part}</span>)}
                            </div>
                        </div>
                    )}
                    {formulaParts.length === 0 && complexModels.includes(modelType) && modelType !== ModuleType.NeuralNetwork && (
                        <div>
                            <h4 className="text-md font-semibold text-gray-700 mb-2">Model Information</h4>
                            {modelType === ModuleType.DecisionTree ? (
                                <div className="space-y-3">
                                    {showTreeImage && plotTreeImage ? (
                                        <div className="bg-gray-50 rounded-lg p-3 border border-gray-200">
                                            {isLoadingPlot ? (
                                                <div className="text-center p-8 text-gray-600">
                                                    <p>Decision Tree 시각화 생성 중...</p>
                                                </div>
                                            ) : (
                                                <div>
                                                    <div className="flex justify-between items-center mb-2">
                                                        <span className="text-sm text-gray-600">Decision Tree 시각화</span>
                                                        <button
                                                            onClick={() => setShowTreeImage(false)}
                                                            className="text-sm text-blue-600 hover:text-blue-800"
                                                        >
                                                            텍스트 보기
                                                        </button>
                                                    </div>
                                                    <img 
                                                        src={plotTreeImage} 
                                                        alt="Decision Tree" 
                                                        className="w-full h-auto"
                                                        style={{ maxHeight: '600px', objectFit: 'contain' }}
                                                    />
                                                </div>
                                            )}
                                        </div>
                                    ) : (
                                        <div className="bg-gray-50 rounded-lg p-3 border border-gray-200">
                                            <div className="flex justify-between items-center mb-2">
                                                <span className="text-sm font-semibold text-gray-700">Decision Tree 구조</span>
                                                <button
                                                    onClick={handleShowTreeImage}
                                                    disabled={isLoadingPlot}
                                                    className="px-3 py-1 text-sm font-semibold text-white bg-blue-600 rounded hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-wait transition-colors"
                                                >
                                                    {isLoadingPlot ? '생성 중...' : '시각화 보기'}
                                                </button>
                                            </div>
                                            {isLoadingText ? (
                                                <div className="text-center p-8 text-gray-600">
                                                    <p>Decision Tree 텍스트 생성 중...</p>
                                                </div>
                                            ) : treeText ? (
                                                <pre className="text-xs font-mono text-gray-800 whitespace-pre-wrap overflow-x-auto bg-white p-3 rounded border border-gray-200 max-h-96 overflow-y-auto">
                                                    {treeText}
                                                </pre>
                                            ) : (
                                                <div className="text-center p-4 text-gray-500">
                                                    <p>Decision Tree 텍스트를 생성할 수 없습니다.</p>
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </div>
                            ) : (
                                <div className="bg-blue-50 rounded-lg p-3 text-sm text-blue-800 border border-blue-200">
                                    <p className="font-sans">
                                        {modelType === ModuleType.RandomForest
                                            ? "Decision Tree 기반 모델은 트리 구조로 예측을 수행하므로 선형 방정식으로 표현할 수 없습니다. 대신 Feature Importance를 통해 각 변수의 중요도를 확인할 수 있습니다."
                                            : "이 모델 타입은 선형 방정식으로 표현할 수 없습니다. Feature Importance를 통해 각 변수의 중요도를 확인할 수 있습니다."}
                                    </p>
                                </div>
                            )}
                        </div>
                    )}
                    
                    <ModalModelMetrics metrics={metrics} />

                    <div>
                        <h4 className="text-md font-semibold text-gray-700 mb-2">
                            {complexModels.includes(modelType) ? "Feature Importances" : "Coefficients"}
                        </h4>
                        <div className="bg-gray-50 rounded-lg border border-gray-200 max-h-60 overflow-y-auto">
                            <table className="w-full text-sm">
                                <thead className="sticky top-0 bg-gray-100">
                                    <tr className="text-left">
                                        <th className="p-2 font-semibold">Feature</th>
                                        <th className="p-2 font-semibold text-right">
                                            {complexModels.includes(modelType) ? "Importance" : "Coefficient"}
                                        </th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {!complexModels.includes(modelType) && (
                                        <tr className="border-t">
                                            <td className="p-2 font-mono text-gray-600">(Intercept)</td>
                                            <td className="p-2 font-mono text-right">{intercept.toFixed(4)}</td>
                                        </tr>
                                    )}
                                    {Object.entries(coefficients)
                                        .sort(([, a], [, b]) => Math.abs(b as number) - Math.abs(a as number))
                                        .map(([feature, value]) => (
                                            <tr key={feature} className="border-t">
                                                <td className="p-2 font-mono">{feature}</td>
                                                <td className="p-2 font-mono text-right">{(value as number).toFixed(4)}</td>
                                            </tr>
                                        ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </main>
            </div>
        </div>
    );
};