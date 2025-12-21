import React, { useState } from 'react';
import { CanvasModule, StatsModelsResultOutput } from '../types';
import { XCircleIcon, SparklesIcon } from './icons';
import { GoogleGenAI } from "@google/genai";
import { MarkdownRenderer } from './MarkdownRenderer';

interface StatsModelsResultPreviewModalProps {
    module: CanvasModule;
    projectName: string;
    onClose: () => void;
}

export const StatsModelsResultPreviewModal: React.FC<StatsModelsResultPreviewModalProps> = ({ module, projectName, onClose }) => {
    const [isInterpreting, setIsInterpreting] = useState(false);
    const [aiInterpretation, setAiInterpretation] = useState<string | null>(null);

    const output = module.outputData as StatsModelsResultOutput;
    if (!output || output.type !== 'StatsModelsResultOutput') return null;

    const { summary, modelType, labelColumn, featureColumns } = output;
    const isOLS = modelType === 'OLS';

    // 수식 생성 (Train Model 참고)
    const generateFormula = () => {
        const coefficients = summary.coefficients;
        const intercept = coefficients['const']?.coef || 0;
        const formulaParts: string[] = [];

        // 모델 타입에 따라 수식 형식 결정
        if (modelType === 'Logistic' || modelType === 'Logit') {
            formulaParts.push(`ln(p / (1 - p)) = ${intercept.toFixed(4)}`);
        } else {
            formulaParts.push(`${labelColumn} ≈ ${intercept.toFixed(4)}`);
        }

        // featureColumns를 사용하여 수식 생성
        if (featureColumns && featureColumns.length > 0) {
            featureColumns.forEach((feature) => {
                const coeffInfo = coefficients[feature];
                if (coeffInfo) {
                    const coeff = coeffInfo.coef;
                    if (coeff >= 0) {
                        formulaParts.push(` + ${coeff.toFixed(4)} * [${feature}]`);
                    } else {
                        formulaParts.push(` - ${Math.abs(coeff).toFixed(4)} * [${feature}]`);
                    }
                }
            });
        } else {
            // featureColumns가 없으면 coefficients에서 const를 제외한 모든 계수 사용
            Object.entries(coefficients).forEach(([param, values]) => {
                if (param !== 'const') {
                    const coeff = values.coef;
                    if (coeff >= 0) {
                        formulaParts.push(` + ${coeff.toFixed(4)} * [${param}]`);
                    } else {
                        formulaParts.push(` - ${Math.abs(coeff).toFixed(4)} * [${param}]`);
                    }
                }
            });
        }

        return formulaParts;
    };

    const formulaParts = generateFormula();

    const handleInterpret = async () => {
        setIsInterpreting(true);
        setAiInterpretation(null);
        try {
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });

            const metricsText = Object.entries(output.summary.metrics).map(([key, value]) => `- ${key}: ${value}`).join('\n');
            const coefficientsText = Object.entries(output.summary.coefficients).map(([param, values]) => {
                const pValue = values['P>|t|'] ?? values['P>|z|'] ?? 1.0;
                return `- ${param}: ${values.coef.toFixed(4)} (p-value: ${pValue.toFixed(3)})`;
            }).join('\n');
            
            const prompt = `
You are a statistician writing a brief report for a non-technical audience. Please use Korean and simple Markdown.

### 통계 모델 분석 보고서

**프로젝트:** ${projectName}
**모델:** ${output.modelType}
**분석 대상:** ${output.labelColumn}

**성능:**
${metricsText}

**계수 (p-value < 0.05):**
${coefficientsText}

---

**1. 모델 적합도:**
- 이 모델이 데이터를 얼마나 잘 설명합니까? 주요 지표(예: R-squared)를 한 문장으로 요약해 주십시오.

**2. 주요 영향 요인:**
- 통계적으로 가장 유의미한(p-value < 0.05) 상위 2-3개 변수는 무엇이며, 이들이 분석 대상에 어떤 영향을 미치니까?

**3. 결론:**
- 이 분석에서 도출할 수 있는 가장 중요한 결론 한 가지는 무엇입니까?

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

    return (
        <div 
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
            onClick={onClose}
        >
            <div 
                className="bg-white text-gray-900 rounded-lg shadow-xl w-full max-w-4xl max-h-[90vh] flex flex-col font-mono"
                onClick={e => e.stopPropagation()}
            >
                <header className="flex items-center justify-between p-4 border-b border-gray-200 flex-shrink-0">
                    <h2 className="text-xl font-bold text-gray-800 font-sans">Analysis Results: {module.name}</h2>
                    <button onClick={onClose} className="text-gray-500 hover:text-gray-800">
                        <XCircleIcon className="w-6 h-6" />
                    </button>
                </header>
                <main className="flex-grow p-4 overflow-auto text-sm">
                    {/* 모델 적합 수식 - 맨 위에 표시 */}
                    {formulaParts.length > 0 && (
                        <div className="bg-green-50 p-4 rounded-lg border border-green-200 mb-6">
                            <h3 className="text-md font-semibold text-gray-700 mb-2 font-sans">Fitted Model Equation</h3>
                            <div className="bg-white rounded-lg p-3 font-mono text-sm text-green-700 whitespace-normal break-words border border-green-300">
                                <span>{formulaParts[0]}</span>
                                {formulaParts.slice(1).map((part, i) => (
                                    <span key={i}>{part}</span>
                                ))}
                            </div>
                        </div>
                    )}

                    <div className="flex justify-end mb-4 font-sans">
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
                            <div role="status" className="flex flex-col items-center">
                                <svg aria-hidden="true" className="w-8 h-8 text-gray-200 animate-spin fill-purple-600" viewBox="0 0 100 101" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z" fill="currentColor"/>
                                    <path d="M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0492C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z" fill="currentFill"/>
                                </svg>
                                <span className="sr-only">Loading...</span>
                                <p className="mt-4">AI가 결과를 분석하고 있습니다...</p>
                            </div>
                        </div>
                    )}
                    
                    {aiInterpretation && (
                         <div className="bg-purple-50 p-4 rounded-lg border border-purple-200 mb-6">
                            <h3 className="text-lg font-bold text-purple-800 mb-2 font-sans flex items-center gap-2">
                                <SparklesIcon className="w-5 h-5"/>
                                AI 분석 요약
                            </h3>
                            <MarkdownRenderer text={aiInterpretation} />
                        </div>
                    )}

                    <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                        <h3 className="text-center font-bold mb-4 text-base">{modelType} Regression Results</h3>
                        
                        <div className="grid grid-cols-2 gap-x-8 mb-4">
                            <div className="space-y-1">
                                <div className="flex justify-between"><span>Dep. Variable:</span> <span className="font-semibold">{`[${labelColumn}]`}</span></div>
                                <div className="flex justify-between"><span>Model:</span> <span className="font-semibold">{modelType}</span></div>
                                <div className="flex justify-between"><span>Date:</span> <span className="font-semibold">{new Date().toLocaleDateString()}</span></div>
                                <div className="flex justify-between"><span>Time:</span> <span className="font-semibold">{new Date().toLocaleTimeString()}</span></div>
                            </div>
                             <div className="space-y-1">
                                {Object.entries(summary.metrics).slice(0, 4).map(([key, value]) => (
                                     <div key={key} className="flex justify-between"><span>{key}:</span> <span className="font-semibold">{typeof value === 'number' ? Number(value).toFixed(4) : value}</span></div>
                                ))}
                            </div>
                        </div>

                        <hr className="my-4 border-gray-300"/>

                        <div>
                            <table className="w-full text-left">
                                <thead>
                                    <tr className="border-b-2 border-gray-300">
                                        <th className="pb-1"></th>
                                        <th className="pb-1 text-right">coef</th>
                                        <th className="pb-1 text-right">std err</th>
                                        <th className="pb-1 text-right">{isOLS ? 't' : 'z'}</th>
                                        <th className="pb-1 text-right">{isOLS ? 'P>|t|' : 'P>|z|'}</th>
                                        <th className="pb-1 text-right">[0.025</th>
                                        <th className="pb-1 text-right">0.975]</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {Object.entries(summary.coefficients).map(([param, values]) => (
                                        <tr key={param}>
                                            <td className="py-1 font-semibold">{param}</td>
                                            <td className="py-1 text-right">{values.coef.toFixed(4)}</td>
                                            <td className="py-1 text-right">{values['std err'].toFixed(4)}</td>
                                            <td className="py-1 text-right">{((isOLS ? values.t : values.z) ?? 0).toFixed(3)}</td>
                                            <td className="py-1 text-right">{((isOLS ? values['P>|t|'] : values['P>|z|']) ?? 0).toFixed(3)}</td>
                                            <td className="py-1 text-right">{values['[0.025'].toFixed(3)}</td>
                                            <td className="py-1 text-right">{values['0.975]'].toFixed(3)}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                        
                        {Object.keys(summary.metrics).length > 4 && (
                            <>
                                <hr className="my-4 border-gray-300"/>
                                <div className="grid grid-cols-2 gap-x-8">
                                     <div className="space-y-1">
                                        {Object.entries(summary.metrics).slice(4).map(([key, value]) => (
                                            <div key={key} className="flex justify-between"><span>{key}:</span> <span className="font-semibold">{typeof value === 'number' ? Number(value).toFixed(4) : value}</span></div>
                                        ))}
                                    </div>
                                </div>
                            </>
                        )}
                    </div>
                </main>
            </div>
        </div>
    );
};