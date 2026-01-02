import React, { useState, useEffect } from 'react';
import { CanvasModule, ColumnPlotOutput, ModuleType, DataPreview } from '../types';
import { XCircleIcon } from './icons';

interface ColumnPlotPreviewModalProps {
    module: CanvasModule;
    projectName: string;
    onClose: () => void;
    modules: CanvasModule[];
    connections: any[];
}

// 차트 타입 옵션을 반환하는 함수
const getAvailableCharts = (
    plot_type: "single" | "double",
    column1Type: "number" | "string",
    column2Type?: "number" | "string"
): string[] => {
    if (plot_type === "single") {
        if (column1Type === "number") {
            return [
                "Histogram",
                "KDE Plot",
                "Boxplot",
                "Violin Plot",
                "ECDF Plot",
                "QQ-Plot",
                "Line Plot",
                "Area Plot",
            ];
        } else {
            return ["Bar Plot", "Count Plot", "Pie Chart", "Frequency Table"];
        }
    } else {
        // 2개열 선택
        if (column1Type === "number" && column2Type === "number") {
            return [
                "Scatter Plot",
                "Hexbin Plot",
                "Joint Plot",
                "Line Plot",
                "Regression Plot",
                "Heatmap",
            ];
        } else if (
            (column1Type === "number" && column2Type === "string") ||
            (column1Type === "string" && column2Type === "number")
        ) {
            return [
                "Box Plot",
                "Violin Plot",
                "Bar Plot",
                "Strip Plot",
                "Swarm Plot",
            ];
        } else {
            // string + string
            return ["Grouped Bar Plot", "Heatmap", "Mosaic Plot"];
        }
    }
};

export const ColumnPlotPreviewModal: React.FC<ColumnPlotPreviewModalProps> = ({
    module,
    projectName,
    onClose,
    modules,
    connections,
}) => {
    const [selectedChart, setSelectedChart] = useState<string>("");
    const [chartImage, setChartImage] = useState<string | null>(null);
    const [isGenerating, setIsGenerating] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // 입력 데이터 가져오기
    const getInputData = (): DataPreview | null => {
        const connection = connections.find(
            (c) => c.to.moduleId === module.id && c.to.portName === "data_in"
        );
        if (!connection) return null;

        const sourceModule = modules.find((m) => m.id === connection.from.moduleId);
        if (!sourceModule || !sourceModule.outputData) return null;

        if (sourceModule.outputData.type === "DataPreview") {
            return sourceModule.outputData as DataPreview;
        }
        return null;
    };

    const inputData = getInputData();
    const outputData = module.outputData as ColumnPlotOutput | undefined;

    const { plot_type = "single", column1 = "", column2 = "" } = module.parameters;

    // 컬럼 타입 확인
    const col1 = inputData?.columns.find((c) => c.name === column1);
    const col2 = column2
        ? inputData?.columns.find((c) => c.name === column2)
        : undefined;
    const column1Type: "number" | "string" =
        col1?.type === "number" ? "number" : "string";
    const column2Type: "number" | "string" | undefined = col2
        ? col2.type === "number"
            ? "number"
            : "string"
        : undefined;

    const availableCharts = getAvailableCharts(
        plot_type as "single" | "double",
        column1Type,
        column2Type
    );

    // 초기 차트 선택 및 availableCharts 변경 시 업데이트
    useEffect(() => {
        if (availableCharts.length > 0) {
            // selectedChart가 비어있거나 availableCharts에 없으면 첫 번째 항목 선택
            if (!selectedChart || !availableCharts.includes(selectedChart)) {
                setSelectedChart(availableCharts[0]);
            }
        } else {
            setSelectedChart("");
        }
    }, [availableCharts.join(","), plot_type, column1, column2]);

    // 차트 생성
    const generateChart = async () => {
        if (!selectedChart || !inputData || !column1) {
            setError("Please select a chart type and ensure columns are selected.");
            return;
        }

        setIsGenerating(true);
        setError(null);
        setChartImage(null);

        try {
            const pyodideModule = await import("../utils/pyodideRunner");
            const { createColumnPlotPython } = pyodideModule;

            const imageBase64 = await createColumnPlotPython(
                inputData.rows || [],
                plot_type,
                column1,
                column2 || null,
                selectedChart,
                120000 // 타임아웃: 120초
            );

            setChartImage(imageBase64);
        } catch (err: any) {
            const errorMessage = err.message || String(err);
            setError(`Failed to generate chart: ${errorMessage}`);
            console.error("Chart generation error:", err);
        } finally {
            setIsGenerating(false);
        }
    };

    // 차트 타입 변경 시 자동 생성
    useEffect(() => {
        if (selectedChart && inputData && column1) {
            generateChart();
        }
    }, [selectedChart, column1, column2, plot_type]);

    if (!inputData) {
        return (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
                    <div className="flex justify-between items-center mb-4">
                        <h2 className="text-xl font-bold text-gray-800">
                            Column Plot - {module.name}
                        </h2>
                        <button
                            onClick={onClose}
                            className="text-gray-500 hover:text-gray-700"
                        >
                            <XCircleIcon className="w-6 h-6" />
                        </button>
                    </div>
                    <p className="text-gray-600">
                        No input data available. Please connect a data source module.
                    </p>
                </div>
            </div>
        );
    }

    if (!column1) {
        return (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
                    <div className="flex justify-between items-center mb-4">
                        <h2 className="text-xl font-bold text-gray-800">
                            Column Plot - {module.name}
                        </h2>
                        <button
                            onClick={onClose}
                            className="text-gray-500 hover:text-gray-700"
                        >
                            <XCircleIcon className="w-6 h-6" />
                        </button>
                    </div>
                    <p className="text-gray-600">
                        Please select a column in the module properties.
                    </p>
                </div>
            </div>
        );
    }

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-6 max-w-4xl w-full mx-4 max-h-[90vh] overflow-y-auto">
                <div className="flex justify-between items-center mb-4">
                    <h2 className="text-xl font-bold text-gray-800">
                        Column Plot - {module.name}
                    </h2>
                    <button
                        onClick={onClose}
                        className="text-gray-500 hover:text-gray-700"
                    >
                        <XCircleIcon className="w-6 h-6" />
                    </button>
                </div>

                <div className="space-y-4">
                    {/* 차트 타입 선택 */}
                    <div>
                        <label className="block text-sm font-semibold text-gray-700 mb-2">
                            Chart Type
                        </label>
                        {availableCharts.length > 0 ? (
                            <select
                                value={selectedChart}
                                onChange={(e) => setSelectedChart(e.target.value)}
                                className="w-full bg-white border border-gray-300 rounded px-3 py-2 text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500"
                            >
                                {availableCharts.map((chart) => (
                                    <option key={chart} value={chart}>
                                        {chart}
                                    </option>
                                ))}
                            </select>
                        ) : (
                            <select
                                disabled
                                className="w-full bg-gray-100 border border-gray-300 rounded px-3 py-2 text-sm text-gray-500 cursor-not-allowed"
                            >
                                <option>No charts available</option>
                            </select>
                        )}
                    </div>

                    {/* 컬럼 정보 */}
                    <div className="bg-gray-50 rounded-lg p-3 border border-gray-200">
                        <p className="text-sm text-gray-600">
                            <span className="font-semibold">Column 1:</span> {column1} (
                            {column1Type})
                        </p>
                        {plot_type === "double" && column2 && (
                            <p className="text-sm text-gray-600 mt-1">
                                <span className="font-semibold">Column 2:</span> {column2} (
                                {column2Type})
                            </p>
                        )}
                    </div>

                    {/* 차트 표시 */}
                    {isGenerating && (
                        <div className="flex items-center justify-center py-8">
                            <div className="text-gray-600">Generating chart...</div>
                        </div>
                    )}

                    {error && (
                        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                            <p className="text-sm text-red-600">{error}</p>
                        </div>
                    )}

                    {chartImage && !isGenerating && (
                        <div className="bg-gray-900 rounded-lg p-4 flex items-center justify-center">
                            <img
                                src={`data:image/png;base64,${chartImage}`}
                                alt={`${selectedChart} Plot`}
                                className="max-w-full h-auto"
                            />
                        </div>
                    )}

                    {!chartImage && !isGenerating && !error && (
                        <div className="text-center py-8 text-gray-500">
                            Select a chart type to generate visualization.
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

