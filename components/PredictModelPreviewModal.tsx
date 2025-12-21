import React, { useState, useMemo, useEffect } from "react";
import {
  CanvasModule,
  DataPreview,
  Connection,
  StatsModelsResultOutput,
  ModuleType,
} from "../types";
import { XCircleIcon } from "./icons";

interface PredictModelPreviewModalProps {
  module: CanvasModule;
  projectName: string;
  onClose: () => void;
  modules?: CanvasModule[];
  connections?: Connection[];
}

type TabType = "detail" | "graph";

export const PredictModelPreviewModal: React.FC<
  PredictModelPreviewModalProps
> = ({ module, projectName, onClose, modules = [], connections = [] }) => {
  const [activeTab, setActiveTab] = useState<TabType>("detail");
  const [selectedActualColumn, setSelectedActualColumn] = useState<string>("");
  const [selectedPredictColumn, setSelectedPredictColumn] =
    useState<string>("");

  const output = module.outputData as DataPreview;
  if (!output || output.type !== "DataPreview") return null;

  // 사용 가능한 숫자 컬럼 목록
  const numericColumns = useMemo(() => {
    return output.columns
      .filter((col) => col.type === "number")
      .map((col) => col.name);
  }, [output.columns]);

  // Predict 컬럼 찾기
  const predictColumn = output.columns.find(
    (col) =>
      col.name === "Predict" || col.name.toLowerCase().includes("predict")
  );

  // Result Model에서 labelColumn 찾기
  const labelColumn = useMemo(() => {
    // PredictModel에 연결된 ResultModel 찾기 (model_in 포트)
    const modelConnection = connections.find(
      (c) => c.to.moduleId === module.id && c.to.portName === "model_in"
    );

    if (modelConnection) {
      const resultModel = modules.find(
        (m) => m.id === modelConnection.from.moduleId
      );

      if (resultModel && resultModel.type === ModuleType.ResultModel) {
        const resultOutput = resultModel.outputData as StatsModelsResultOutput;
        if (resultOutput && resultOutput.type === "StatsModelsResultOutput") {
          // Result Model의 labelColumn 사용
          return resultOutput.labelColumn;
        }
      }
    }

    // 대체 방법: 모듈 파라미터에서 찾기
    if (module.parameters?.label_column) {
      return module.parameters.label_column;
    }

    // 최후의 수단: Predict 컬럼이 있으면 다른 숫자 컬럼 중 하나를 시도
    const numericCols = output.columns.filter(
      (col) =>
        col.type === "number" &&
        col.name !== "Predict" &&
        !col.name.toLowerCase().includes("predict")
    );
    return numericCols.length > 0 ? numericCols[0].name : null;
  }, [module, modules, connections, output.columns]);

  // 기본값 설정
  useEffect(() => {
    if (
      !selectedActualColumn &&
      labelColumn &&
      numericColumns.includes(labelColumn)
    ) {
      setSelectedActualColumn(labelColumn);
    }
    if (
      !selectedPredictColumn &&
      predictColumn &&
      numericColumns.includes(predictColumn.name)
    ) {
      setSelectedPredictColumn(predictColumn.name);
    }
  }, [
    labelColumn,
    predictColumn,
    numericColumns,
    selectedActualColumn,
    selectedPredictColumn,
  ]);

  const rows = output.rows || [];

  // 선택된 컬럼 이름 (기본값 또는 사용자 선택)
  const actualColName = selectedActualColumn || labelColumn || "";
  const predictColName =
    selectedPredictColumn || predictColumn?.name || "Predict";

  // 그래프 데이터 준비 (선택된 컬럼들의 값 비교)
  const graphData = useMemo(() => {
    if (!actualColName || !predictColName) {
      return null;
    }

    return rows
      .map((row, idx) => {
        // 선택된 실제값 컬럼
        const actual =
          typeof row[actualColName] === "number"
            ? row[actualColName]
            : parseFloat(row[actualColName]);

        // 선택된 예측값 컬럼
        const predicted =
          typeof row[predictColName] === "number"
            ? row[predictColName]
            : parseFloat(row[predictColName]);

        if (isNaN(actual) || isNaN(predicted)) return null;

        return {
          index: idx,
          actual,
          predicted,
          error: actual - predicted, // 차이
          absError: Math.abs(actual - predicted),
        };
      })
      .filter((d): d is NonNullable<typeof d> => d !== null);
  }, [rows, actualColName, predictColName]);

  // 통계 계산
  const statistics = useMemo(() => {
    if (!graphData || graphData.length === 0) return null;

    const errors = graphData.map((d) => d.error);
    const absErrors = graphData.map((d) => d.absError);

    const meanError = errors.reduce((sum, e) => sum + e, 0) / errors.length;
    const meanAbsError =
      absErrors.reduce((sum, e) => sum + e, 0) / absErrors.length;
    const maxError = Math.max(...absErrors);
    const minError = Math.min(...absErrors);

    // RMSE 계산
    const mse = errors.reduce((sum, e) => sum + e * e, 0) / errors.length;
    const rmse = Math.sqrt(mse);

    return {
      meanError,
      meanAbsError,
      maxError,
      minError,
      rmse,
      count: graphData.length,
    };
  }, [graphData]);

  // 그래프 렌더링 함수
  const renderComparisonPlot = () => {
    if (!graphData || graphData.length === 0) {
      return (
        <div className="text-center p-8 text-gray-500">
          실제값과 예측값을 비교할 데이터가 없습니다.
        </div>
      );
    }

    const width = 800;
    const height = 400;
    const padding = { top: 40, right: 60, bottom: 60, left: 60 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // 데이터 범위 계산
    const allValues = [
      ...graphData.map((d) => d.actual),
      ...graphData.map((d) => d.predicted),
    ];
    const minVal = Math.min(...allValues);
    const maxVal = Math.max(...allValues);
    const range = maxVal - minVal || 1;
    const margin = range * 0.1;

    const scaleX = (index: number) =>
      padding.left + (index / (graphData.length - 1 || 1)) * chartWidth;
    const scaleY = (value: number) =>
      padding.top +
      chartHeight -
      ((value - minVal + margin) / (range + margin * 2)) * chartHeight;

    // 실제값과 예측값을 위한 경로 생성
    const actualPath = graphData
      .map((d, i) => `${i === 0 ? "M" : "L"} ${scaleX(i)} ${scaleY(d.actual)}`)
      .join(" ");

    const predictedPath = graphData
      .map(
        (d, i) => `${i === 0 ? "M" : "L"} ${scaleX(i)} ${scaleY(d.predicted)}`
      )
      .join(" ");

    return (
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <h3 className="text-lg font-bold text-gray-800 mb-4 text-center">
          Result Model Label vs Predict Model 예측값
        </h3>
        <svg
          width={width}
          height={height}
          className="w-full"
          viewBox={`0 0 ${width} ${height}`}
        >
          {/* 그리드 라인 */}
          {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
            const y = padding.top + chartHeight - ratio * chartHeight;
            return (
              <line
                key={`grid-${ratio}`}
                x1={padding.left}
                y1={y}
                x2={width - padding.right}
                y2={y}
                stroke="#e5e7eb"
                strokeWidth="1"
                strokeDasharray="2,2"
              />
            );
          })}

          {/* Y축 라벨 */}
          {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
            const y = padding.top + chartHeight - ratio * chartHeight;
            const value = minVal + margin + (range + margin * 2) * (1 - ratio);
            return (
              <text
                key={`y-label-${ratio}`}
                x={padding.left - 10}
                y={y + 4}
                textAnchor="end"
                fontSize="10"
                fill="#666"
              >
                {value.toFixed(2)}
              </text>
            );
          })}

          {/* X축 라벨 */}
          <text
            x={width / 2}
            y={height - 10}
            textAnchor="middle"
            fontSize="12"
            fill="#666"
          >
            데이터 포인트 인덱스
          </text>

          {/* Y축 라벨 */}
          <text
            x={20}
            y={height / 2}
            textAnchor="middle"
            fontSize="12"
            fill="#666"
            transform={`rotate(-90, 20, ${height / 2})`}
          >
            값
          </text>

          {/* 실제값 라인 */}
          <path d={actualPath} fill="none" stroke="#3b82f6" strokeWidth="2" />

          {/* 예측값 라인 */}
          <path
            d={predictedPath}
            fill="none"
            stroke="#ef4444"
            strokeWidth="2"
            strokeDasharray="5,5"
          />

          {/* 범례 */}
          <g
            transform={`translate(${width - padding.right - 150}, ${
              padding.top + 20
            })`}
          >
            <line
              x1="0"
              y1="0"
              x2="30"
              y2="0"
              stroke="#3b82f6"
              strokeWidth="2"
            />
            <text x="35" y="4" fontSize="11" fill="#333">
              Result Label ({actualColName})
            </text>
            <line
              x1="0"
              y1="15"
              x2="30"
              y2="15"
              stroke="#ef4444"
              strokeWidth="2"
              strokeDasharray="5,5"
            />
            <text x="35" y="19" fontSize="11" fill="#333">
              Predict ({predictColName})
            </text>
          </g>
        </svg>
      </div>
    );
  };

  const renderErrorPlot = () => {
    if (!graphData || graphData.length === 0) {
      return null;
    }

    const width = 800;
    const height = 300;
    const padding = { top: 40, right: 60, bottom: 60, left: 60 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // 오차 범위 계산
    const errors = graphData.map((d) => d.error);
    const minError = Math.min(...errors);
    const maxError = Math.max(...errors);
    const errorRange = maxError - minError || 1;
    const margin = errorRange * 0.1;

    const scaleX = (index: number) =>
      padding.left + (index / (graphData.length - 1 || 1)) * chartWidth;
    const scaleY = (error: number) =>
      padding.top +
      chartHeight -
      ((error - minError + margin) / (errorRange + margin * 2)) * chartHeight;

    // 오차 경로 생성
    const errorPath = graphData
      .map((d, i) => `${i === 0 ? "M" : "L"} ${scaleX(i)} ${scaleY(d.error)}`)
      .join(" ");

    // 0 기준선
    const zeroLineY = scaleY(0);

    return (
      <div className="bg-white rounded-lg border border-gray-200 p-4 mt-4">
        <h3 className="text-lg font-bold text-gray-800 mb-4 text-center">
          차이 (Result Label - Predict)
        </h3>
        <svg
          width={width}
          height={height}
          className="w-full"
          viewBox={`0 0 ${width} ${height}`}
        >
          {/* 그리드 라인 */}
          {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
            const y = padding.top + chartHeight - ratio * chartHeight;
            return (
              <line
                key={`grid-${ratio}`}
                x1={padding.left}
                y1={y}
                x2={width - padding.right}
                y2={y}
                stroke="#e5e7eb"
                strokeWidth="1"
                strokeDasharray="2,2"
              />
            );
          })}

          {/* 0 기준선 */}
          <line
            x1={padding.left}
            y1={zeroLineY}
            x2={width - padding.right}
            y2={zeroLineY}
            stroke="#9ca3af"
            strokeWidth="1.5"
            strokeDasharray="3,3"
          />

          {/* Y축 라벨 */}
          {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
            const y = padding.top + chartHeight - ratio * chartHeight;
            const value =
              minError + margin + (errorRange + margin * 2) * (1 - ratio);
            return (
              <text
                key={`y-label-${ratio}`}
                x={padding.left - 10}
                y={y + 4}
                textAnchor="end"
                fontSize="10"
                fill="#666"
              >
                {value.toFixed(2)}
              </text>
            );
          })}

          {/* X축 라벨 */}
          <text
            x={width / 2}
            y={height - 10}
            textAnchor="middle"
            fontSize="12"
            fill="#666"
          >
            데이터 포인트 인덱스
          </text>

          {/* Y축 라벨 */}
          <text
            x={20}
            y={height / 2}
            textAnchor="middle"
            fontSize="12"
            fill="#666"
            transform={`rotate(-90, 20, ${height / 2})`}
          >
            오차
          </text>

          {/* 오차 라인 */}
          <path d={errorPath} fill="none" stroke="#8b5cf6" strokeWidth="2" />

          {/* 데이터 포인트 */}
          {graphData.map((d, i) => (
            <circle
              key={i}
              cx={scaleX(i)}
              cy={scaleY(d.error)}
              r="3"
              fill="#8b5cf6"
            />
          ))}
        </svg>
      </div>
    );
  };

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
      onClick={onClose}
    >
      <div
        className="bg-white text-gray-900 rounded-lg shadow-xl w-full max-w-6xl max-h-[90vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-center justify-between p-4 border-b border-gray-200 flex-shrink-0">
          <h2 className="text-xl font-bold text-gray-800">
            Predict Model Details: {module.name}
          </h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-800"
          >
            <XCircleIcon className="w-6 h-6" />
          </button>
        </header>

        {/* 탭 버튼 */}
        <div className="flex border-b border-gray-200 flex-shrink-0">
          <button
            onClick={() => setActiveTab("detail")}
            className={`flex-1 px-4 py-3 text-sm font-semibold transition-colors ${
              activeTab === "detail"
                ? "bg-gray-100 text-gray-900 border-b-2 border-blue-600"
                : "text-gray-600 hover:bg-gray-50"
            }`}
          >
            Detail
          </button>
          <button
            onClick={() => setActiveTab("graph")}
            className={`flex-1 px-4 py-3 text-sm font-semibold transition-colors ${
              activeTab === "graph"
                ? "bg-gray-100 text-gray-900 border-b-2 border-blue-600"
                : "text-gray-600 hover:bg-gray-50"
            }`}
          >
            Graph
          </button>
        </div>

        <main className="flex-grow p-6 overflow-auto">
          {activeTab === "detail" && (
            <div className="space-y-4">
              <div className="bg-gray-50 rounded-lg p-4">
                <h3 className="text-lg font-bold mb-2">데이터 미리보기</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm border-collapse">
                    <thead>
                      <tr className="bg-gray-200">
                        {output.columns.slice(0, 10).map((col) => (
                          <th
                            key={col.name}
                            className="border border-gray-300 px-2 py-1 text-left font-semibold"
                          >
                            {col.name}
                          </th>
                        ))}
                        {output.columns.length > 10 && (
                          <th className="border border-gray-300 px-2 py-1 text-left font-semibold">
                            ...
                          </th>
                        )}
                      </tr>
                    </thead>
                    <tbody>
                      {rows.slice(0, 100).map((row, idx) => (
                        <tr key={idx} className="hover:bg-gray-100">
                          {output.columns.slice(0, 10).map((col) => (
                            <td
                              key={col.name}
                              className="border border-gray-300 px-2 py-1"
                            >
                              {typeof row[col.name] === "number"
                                ? row[col.name].toFixed(4)
                                : String(row[col.name] || "")}
                            </td>
                          ))}
                          {output.columns.length > 10 && (
                            <td className="border border-gray-300 px-2 py-1">
                              ...
                            </td>
                          )}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="text-sm text-gray-600 mt-2">
                  총 {output.totalRowCount}행, {output.columns.length}열 (최대
                  100행 표시)
                </p>
              </div>
            </div>
          )}

          {activeTab === "graph" && (
            <div className="space-y-4">
              {/* 열 선택 콤보박스 */}
              <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                <h3 className="text-md font-semibold text-gray-800 mb-3">
                  분석할 열(컬럼) 선택
                </h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      실제값 컬럼 (Label)
                    </label>
                    <select
                      value={selectedActualColumn}
                      onChange={(e) => setSelectedActualColumn(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="">컬럼 선택...</option>
                      {numericColumns.map((colName) => (
                        <option key={colName} value={colName}>
                          {colName}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      예측값 컬럼 (Predict)
                    </label>
                    <select
                      value={selectedPredictColumn}
                      onChange={(e) => setSelectedPredictColumn(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="">컬럼 선택...</option>
                      {numericColumns.map((colName) => (
                        <option key={colName} value={colName}>
                          {colName}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
                {(selectedActualColumn || selectedPredictColumn) && (
                  <p className="text-xs text-gray-600 mt-2">
                    선택된 컬럼: {selectedActualColumn || "(미선택)"} vs{" "}
                    {selectedPredictColumn || "(미선택)"}
                  </p>
                )}
              </div>

              {/* 통계 정보 */}
              {statistics && (
                <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                  <h3 className="text-md font-semibold text-gray-800 mb-2">
                    오차 통계
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
                    <div>
                      <span className="text-gray-600">평균 오차:</span>
                      <span className="ml-2 font-mono font-semibold">
                        {statistics.meanError.toFixed(4)}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600">평균 절대 오차:</span>
                      <span className="ml-2 font-mono font-semibold">
                        {statistics.meanAbsError.toFixed(4)}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600">RMSE:</span>
                      <span className="ml-2 font-mono font-semibold">
                        {statistics.rmse.toFixed(4)}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600">최대 오차:</span>
                      <span className="ml-2 font-mono font-semibold">
                        {statistics.maxError.toFixed(4)}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600">최소 오차:</span>
                      <span className="ml-2 font-mono font-semibold">
                        {statistics.minError.toFixed(4)}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600">분석 행 수:</span>
                      <span className="ml-2 font-mono font-semibold">
                        {statistics.count}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {renderComparisonPlot()}
              {renderErrorPlot()}
            </div>
          )}
        </main>
      </div>
    </div>
  );
};
