import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  useMemo,
} from "react";
import {
  CanvasModule,
  ModuleType,
  ModuleStatus,
  StatisticsOutput,
  Connection,
  DataPreview,
  TrainedModelOutput,
  StatsModelsResultOutput,
  FittedDistributionOutput,
  ExposureCurveOutput,
  XoLPriceOutput,
  FinalXolPriceOutput,
  EvaluationOutput,
  ColumnInfo,
  SplitDataOutput,
  MissingHandlerOutput,
  EncoderOutput,
  NormalizerOutput,
} from "../types";
import {
  PlayIcon,
  TableCellsIcon,
  CommandLineIcon,
  CogIcon,
  CodeBracketIcon,
  InformationCircleIcon,
  SparklesIcon,
  ClipboardIcon,
  CheckIcon,
} from "./icons";
import { getModuleCode } from "../codeSnippets";
import { SAMPLE_DATA } from "../sampleData";
import { GoogleGenAI, Type } from "@google/genai";
import { DEFAULT_MODULES } from "../constants";

type TerminalLog = {
  id: number;
  level: "INFO" | "WARN" | "ERROR" | "SUCCESS";
  message: string;
  timestamp: string;
};

interface PropertiesPanelProps {
  module: CanvasModule | null;
  projectName: string;
  updateModuleParameters: (id: string, newParams: Record<string, any>) => void;
  updateModuleName: (id: string, newName: string) => void;
  logs: TerminalLog[];
  modules: CanvasModule[];
  connections: Connection[];
  activeTab: "properties" | "preview" | "code";
  setActiveTab: (tab: "properties" | "preview" | "code") => void;
  onViewDetails: (moduleId: string) => void;
  folderHandle: FileSystemDirectoryHandle | null;
}

const ExplanationRenderer: React.FC<{ text: string }> = ({ text }) => {
  const renderLine = (line: string) => {
    const boldRegex = /\*\*(.*?)\*\*/g;
    const codeRegex = /`([^`]+)`/g;
    const parts = [];
    let lastIndex = 0;
    let result;

    const combinedRegex = new RegExp(
      `(${boldRegex.source})|(${codeRegex.source})`,
      "g"
    );

    while ((result = combinedRegex.exec(line)) !== null) {
      // Text before the match
      if (result.index > lastIndex) {
        parts.push(line.substring(lastIndex, result.index));
      }
      // Matched part
      if (result[2]) {
        // Bold
        parts.push(<strong key={result.index}>{result[2]}</strong>);
      } else if (result[4]) {
        // Code
        parts.push(
          <code
            key={result.index}
            className="bg-gray-700 text-purple-300 px-1 py-0.5 rounded text-xs"
          >
            {result[4]}
          </code>
        );
      }
      lastIndex = combinedRegex.lastIndex;
    }

    // Text after the last match
    if (lastIndex < line.length) {
      parts.push(line.substring(lastIndex));
    }

    return parts.length > 0 ? <>{parts}</> : <>{line}</>;
  };

  return (
    <div className="text-gray-300 space-y-2 text-sm">
      {text.split("\n").map((line, index) => {
        const trimmedLine = line.trim();
        if (trimmedLine.startsWith("### ")) {
          return (
            <h4
              key={index}
              className="text-md font-semibold mt-3 mb-1 text-gray-200"
            >
              {renderLine(trimmedLine.substring(4))}
            </h4>
          );
        }
        if (trimmedLine.startsWith("## ")) {
          return (
            <h3
              key={index}
              className="text-lg font-semibold mt-4 mb-2 text-gray-100"
            >
              {renderLine(trimmedLine.substring(3))}
            </h3>
          );
        }
        if (trimmedLine.startsWith("* ")) {
          return (
            <div key={index} className="flex items-start pl-2">
              <span className="mr-2 mt-1">•</span>
              <div className="flex-1">
                {renderLine(trimmedLine.substring(2))}
              </div>
            </div>
          );
        }
        if (trimmedLine === "") {
          return null;
        }
        return <p key={index}>{renderLine(line)}</p>;
      })}
    </div>
  );
};

const AIModuleExplanation: React.FC<{ module: CanvasModule }> = ({
  module,
}) => {
  const [explanation, setExplanation] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [show, setShow] = useState(false);

  const handleExplain = async () => {
    if (explanation) {
      setShow(!show);
      return;
    }
    setIsLoading(true);
    setShow(true);
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });

      const defaultModuleData = DEFAULT_MODULES.find(
        (m) => m.type === module.type
      );
      const defaultParams = defaultModuleData
        ? defaultModuleData.parameters
        : {};

      const paramDetails = Object.keys(module.parameters)
        .map((key) => {
          const currentValue = module.parameters[key];
          const defaultValue = defaultParams[key];
          return `- **${key}**: (현재 값: \`${JSON.stringify(
            currentValue
          )}\`, 기본값: \`${JSON.stringify(defaultValue)}\`)`;
        })
        .join("\n");

      let optionsContext = "";
      if (
        module.type === ModuleType.NormalizeData &&
        "method" in module.parameters
      ) {
        optionsContext =
          "\n\n**옵션 컨텍스트:**\n'method' 파라미터는 ['MinMax', 'StandardScaler', 'RobustScaler'] 옵션을 가집니다. 각 옵션의 차이점을 설명해 주세요.";
      } else if (
        module.type === ModuleType.SplitData &&
        "shuffle" in module.parameters
      ) {
        optionsContext =
          "\n\n**옵션 컨텍스트:**\n'shuffle'과 'stratify' 파라미터는 ['True', 'False'] 옵션을 가집니다. 각 옵션이 언제 사용되는지 설명해 주세요.";
      } else if (
        module.type === ModuleType.StatModels &&
        "model" in module.parameters
      ) {
        optionsContext =
          "\n\n**옵션 컨텍스트:**\n'model' 파라미터는 ['OLS', 'Logit', 'Poisson', 'NegativeBinomial', 'Gamma', 'Tweedie'] 옵션을 가집니다. 각 모델의 용도를 간략히 설명해 주세요.";
      }

      const prompt = `
당신은 머신러닝 파이프라인 도구를 위한 전문 AI 어시스턴트입니다. 주어진 모듈과 파라미터에 대해 한국어로 명확하고 유용한 설명을 간단한 마크다운 형식으로 제공해야 합니다.

**모듈:** \`${module.type}\`

### 모듈의 목적
(이 모듈이 무엇을 하는지, 어떤 문제를 해결하는지 한두 문장으로 설명해 주세요.)

### 파라미터 상세 정보
${paramDetails}
${optionsContext}

---
**요청사항:**
위 정보를 바탕으로, 각 파라미터에 대해 아래 형식을 사용하여 상세한 설명을 생성해 주세요.

*   **\`파라미터명\`**
    *   **설명:** 이 파라미터의 역할과 중요성을 설명합니다.
    *   **추천:** 일반적인 사용 사례나 추천하는 값 또는 값의 범위를 제시합니다. (예: "일반적으로 0.7 또는 0.8을 사용합니다.")
    *   **옵션:** (선택 가능한 옵션이 있는 경우) 각 옵션의 의미와 장단점, 그리고 어떤 상황에 사용해야 하는지 설명합니다.

전체적으로 초보자도 이해하기 쉽게, 간결하면서도 정보를 충분히 담아 작성해 주세요.
`;

      const response = await ai.models.generateContent({
        model: "gemini-2.5-flash",
        contents: prompt,
      });
      setExplanation(response.text);
    } catch (error) {
      console.error("AI explanation failed:", error);
      setExplanation("설명을 생성하는 데 실패했습니다.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="mt-4 border-t border-gray-700 pt-3">
      <button
        onClick={handleExplain}
        disabled={isLoading}
        className="flex items-center justify-center gap-2 w-full px-3 py-1.5 text-xs bg-purple-600 hover:bg-purple-700 disabled:bg-purple-400 rounded-md font-semibold text-white transition-colors"
      >
        <SparklesIcon className="h-4 w-4" />
        {isLoading
          ? "생성 중..."
          : show && explanation
          ? "설명 숨기기"
          : "AI로 파라미터 설명 보기"}
      </button>
      {show && (
        <div className="mt-2 p-3 bg-gray-700 rounded-lg">
          {isLoading && (
            <p className="text-sm text-gray-400">
              AI 설명을 생성하고 있습니다...
            </p>
          )}
          {explanation && <ExplanationRenderer text={explanation} />}
        </div>
      )}
    </div>
  );
};

const AIParameterRecommender: React.FC<{
  module: CanvasModule;
  inputColumns: string[];
  projectName: string;
  updateModuleParameters: (id: string, newParams: Record<string, any>) => void;
}> = ({ module, inputColumns, projectName, updateModuleParameters }) => {
  const [isLoading, setIsLoading] = useState(false);

  const handleRecommend = async () => {
    setIsLoading(true);
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });

      const prompt = `
You are an expert data scientist AI assistant. Your task is to recommend the optimal feature columns and a single label/target column for a machine learning model based on a project goal and a list of available data columns.

### Project Goal
"${projectName}"

### Available Columns
- ${inputColumns.join("\n- ")}

### Instructions
1.  Analyze the project goal and column names to infer the prediction target.
2.  Identify the column that is most likely the **label column** (the variable to be predicted).
3.  Select a set of columns that would be good **feature columns** (input variables for the model). Exclude the label column and any columns that seem irrelevant or are identifiers.
4.  Provide your response *only* in a valid JSON format. The JSON object must contain two keys:
    - \`label_column\`: A string with the name of the single recommended label column.
    - \`feature_columns\`: An array of strings with the names of the recommended feature columns.
`;

      const response = await ai.models.generateContent({
        model: "gemini-2.5-flash",
        contents: prompt,
        config: {
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              feature_columns: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
              },
              label_column: { type: Type.STRING },
            },
          },
        },
      });

      const resultJson = JSON.parse(response.text);

      if (resultJson.feature_columns && resultJson.label_column) {
        const validFeatures = resultJson.feature_columns.filter((col: string) =>
          inputColumns.includes(col)
        );
        const validLabel = inputColumns.includes(resultJson.label_column)
          ? resultJson.label_column
          : null;

        updateModuleParameters(module.id, {
          feature_columns: validFeatures,
          label_column: validLabel,
        });
      } else {
        throw new Error("Invalid JSON structure in AI response.");
      }
    } catch (error) {
      console.error("AI recommendation failed:", error);
      // Optionally, add user-facing error feedback here
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="mb-4">
      <button
        onClick={handleRecommend}
        disabled={isLoading}
        className="flex items-center justify-center gap-2 w-full px-3 py-1.5 text-xs bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-400 rounded-md font-semibold text-white transition-colors"
      >
        <SparklesIcon className="h-4 w-4" />
        {isLoading ? "분석 중..." : "AI 추천"}
      </button>
    </div>
  );
};

const PropertyGroup: React.FC<{
  title: string;
  children: React.ReactNode;
  module: CanvasModule;
}> = ({ title, children, module }) => (
  <div className="mb-4">
    <h4 className="text-xs text-gray-500 uppercase font-bold mb-2">{title}</h4>
    <div className="bg-gray-800 p-3 rounded-lg">
      {children}
      <AIModuleExplanation module={module} />
    </div>
  </div>
);

const PropertyInput: React.FC<{
  label: string;
  value: any;
  onChange: (value: any) => void;
  type?: string;
  step?: string;
}> = ({ label, value, onChange, type = "text", step }) => (
  <div className="mb-3 last:mb-0">
    <label className="block text-sm text-gray-400 mb-1">{label}</label>
    <input
      type={type}
      value={value}
      step={step}
      onChange={(e) =>
        onChange(
          type === "number" ? parseFloat(e.target.value) : e.target.value
        )
      }
      className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
    />
  </div>
);

const PropertySelect: React.FC<{
  label: string;
  value: any;
  onChange: (value: string) => void;
  options: (string | { label: string; value: string })[];
}> = ({ label, value, onChange, options }) => (
  <div className="mb-3 last:mb-0">
    <label className="block text-sm text-gray-400 mb-1">{label}</label>
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
    >
      {options.map((opt) => {
        const optionValue = typeof opt === "string" ? opt : opt.value;
        const optionLabel = typeof opt === "string" ? opt : opt.label;
        return (
          <option key={optionValue} value={optionValue}>
            {optionLabel}
          </option>
        );
      })}
    </select>
  </div>
);

const PropertyDisplay: React.FC<{ label: string; value: React.ReactNode }> = ({
  label,
  value,
}) => (
  <div className="mb-3 last:mb-0">
    <label className="block text-sm text-gray-400 mb-1">{label}</label>
    <div className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text-sm text-gray-300">
      {value}
    </div>
  </div>
);

// Helper function to get connected data source, used in renderParameters and PropertiesPanel
const getConnectedDataSourceHelper = (
  moduleId: string,
  allModules: CanvasModule[],
  allConnections: Connection[],
  portNameToFind?: string
): DataPreview | undefined => {
  const portName = portNameToFind || "data_in";
  let inputConnection = allConnections.find(
    (c) => c.to.moduleId === moduleId && c.to.portName === portName
  );

  // Only perform fallback if no specific port was requested and the initial attempt failed
  if (!inputConnection && !portNameToFind) {
    // A safer fallback: find the first connection to any 'data' type port.
    inputConnection = allConnections.find((c) => {
      if (c.to.moduleId === moduleId) {
        const targetModule = allModules.find((m) => m.id === moduleId);
        const targetPort = targetModule?.inputs.find(
          (p) => p.name === c.to.portName
        );
        return targetPort?.type === "data";
      }
      return false;
    });
  }

  if (!inputConnection) return undefined;

  const sourceModule = allModules.find(
    (m) => m.id === inputConnection.from.moduleId
  );
  if (!sourceModule?.outputData) return undefined;

  if (sourceModule.outputData.type === "DataPreview") {
    return sourceModule.outputData;
  } else if (sourceModule.outputData.type === "SplitDataOutput") {
    const fromPortName = inputConnection.from.portName;
    return fromPortName === "train_data_out"
      ? sourceModule.outputData.train
      : sourceModule.outputData.test;
  }
  return undefined;
};

const renderParameters = (
  module: CanvasModule,
  onParamChange: (key: string, value: any) => void,
  fileInputRef: React.RefObject<HTMLInputElement>,
  allModules: CanvasModule[],
  allConnections: Connection[],
  projectName: string,
  updateModuleParameters: (id: string, newParams: Record<string, any>) => void,
  onSampleLoad: (sample: { name: string; content: string }) => void,
  folderHandle: FileSystemDirectoryHandle | null
) => {
  // Use the helper function
  const getConnectedDataSource = (moduleId: string, portNameToFind?: string) =>
    getConnectedDataSourceHelper(
      moduleId,
      allModules,
      allConnections,
      portNameToFind
    );

  switch (module.type) {
    // ... [Previous cases remain unchanged: LoadData, SelectData, HandleMissingValues, TransformData, EncodeCategorical, NormalizeData, TransitionData, ResampleData, SplitData] ...
    case ModuleType.LoadData:
    case ModuleType.XolLoading: {
      const handleBrowseClick = async () => {
        if (folderHandle && (window as any).showOpenFilePicker) {
          try {
            const [fileHandle] = await (window as any).showOpenFilePicker({
              startIn: folderHandle,
              types: [
                {
                  description: "CSV Files",
                  accept: { "text/csv": [".csv"] },
                },
              ],
            });
            const file = await fileHandle.getFile();
            const reader = new FileReader();
            reader.onload = (e) => {
              const content = e.target?.result as string;
              updateModuleParameters(module.id, {
                source: file.name,
                fileContent: content,
              });
            };
            reader.readAsText(file);
          } catch (error: any) {
            if (error.name !== "AbortError") {
              console.warn(
                "Could not use directory picker, falling back to default.",
                error
              );
              fileInputRef.current?.click();
            }
          }
        } else {
          fileInputRef.current?.click();
        }
      };

      return (
        <div>
          <label className="block text-sm text-gray-400 mb-1">Source</label>
          <div className="flex gap-2">
            <input
              type="text"
              value={module.parameters.source}
              onChange={(e) => onParamChange("source", e.target.value)}
              className="flex-grow bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="No file selected"
            />
            <button
              onClick={handleBrowseClick}
              className="px-3 py-1.5 text-sm bg-gray-600 hover:bg-gray-500 rounded-md font-semibold text-white transition-colors"
            >
              Browse...
            </button>
          </div>
          <div className="mt-4">
            <h4 className="text-xs text-gray-500 uppercase font-bold mb-2">
              Examples
            </h4>
            <div className="bg-gray-700 p-2 rounded-md space-y-1">
              {SAMPLE_DATA.map((sample) => (
                <div
                  key={sample.name}
                  onDoubleClick={() => onSampleLoad(sample)}
                  className="px-2 py-1.5 text-sm text-gray-300 rounded-md hover:bg-gray-600 cursor-pointer"
                  title="Double-click to load"
                >
                  {sample.name}
                </div>
              ))}
            </div>
          </div>
        </div>
      );
    }
    case ModuleType.SelectData: {
      const sourceData = getConnectedDataSource(module.id);
      const inputColumns = sourceData?.columns || [];
      const availableDataTypes = [
        "string",
        "number",
        "boolean",
        "category",
        "datetime",
      ];

      const currentSelections = module.parameters.columnSelections || {};

      const handleSelectionChange = (
        colName: string,
        key: "selected" | "type",
        value: boolean | string
      ) => {
        const newSelections = {
          ...currentSelections,
          [colName]: {
            ...(currentSelections[colName] || {
              selected: true,
              type: "string",
            }),
            [key]: value,
          },
        };
        onParamChange("columnSelections", newSelections);
      };

      const handleSelectAll = (selectAll: boolean) => {
        const newSelections = { ...currentSelections };
        inputColumns.forEach((col) => {
          newSelections[col.name] = {
            ...(currentSelections[col.name] || { type: col.type }),
            selected: selectAll,
          };
        });
        onParamChange("columnSelections", newSelections);
      };

      if (inputColumns.length === 0) {
        return (
          <p className="text-sm text-gray-500">
            Connect a data source module to configure columns.
          </p>
        );
      }

      return (
        <div className="flex flex-col">
          <div className="flex justify-end gap-2 mb-2">
            <button
              onClick={() => handleSelectAll(true)}
              className="px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded-md font-semibold"
            >
              Select All
            </button>
            <button
              onClick={() => handleSelectAll(false)}
              className="px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded-md font-semibold"
            >
              Deselect All
            </button>
          </div>
          <div className="space-y-2 pr-2">
            <div className="grid grid-cols-3 gap-2 items-center sticky top-0 bg-gray-800 py-1">
              <span className="text-xs font-bold text-gray-400 col-span-2">
                Column Name
              </span>
              <span className="text-xs font-bold text-gray-400">Data Type</span>
            </div>
            {inputColumns.map((col) => {
              const selection = currentSelections[col.name] || {
                selected: true,
                type: col.type,
              };
              return (
                <div
                  key={col.name}
                  className="grid grid-cols-3 gap-2 items-center"
                >
                  <label
                    className="flex items-center gap-2 text-sm truncate col-span-2"
                    title={col.name}
                  >
                    <input
                      type="checkbox"
                      checked={selection.selected}
                      onChange={(e) =>
                        handleSelectionChange(
                          col.name,
                          "selected",
                          e.target.checked
                        )
                      }
                      className="h-4 w-4 rounded bg-gray-700 border-gray-600 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="truncate">{col.name}</span>
                  </label>
                  <select
                    value={selection.type}
                    onChange={(e) =>
                      handleSelectionChange(col.name, "type", e.target.value)
                    }
                    className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-blue-500"
                  >
                    {availableDataTypes.map((type) => (
                      <option key={type} value={type}>
                        {type}
                      </option>
                    ))}
                  </select>
                </div>
              );
            })}
          </div>
        </div>
      );
    }
    case ModuleType.HandleMissingValues: {
      const { method, strategy, n_neighbors, metric } = module.parameters;

      return (
        <>
          <PropertySelect
            label="Method"
            value={method}
            onChange={(v) => onParamChange("method", v)}
            options={[
              { label: "Remove Entire Row", value: "remove_row" },
              { label: "Impute with Representative Value", value: "impute" },
              { label: "Impute using Neighbors (KNN)", value: "knn" },
            ]}
          />
          {method === "impute" && (
            <div className="mt-3 pt-3 border-t border-gray-700">
              <PropertySelect
                label="Strategy"
                value={strategy}
                onChange={(v) => onParamChange("strategy", v)}
                options={["mean", "median", "mode"]}
              />
            </div>
          )}
          {method === "knn" && (
            <div className="mt-3 pt-3 border-t border-gray-700 space-y-3">
              <PropertyInput
                label="n_neighbors"
                type="number"
                value={n_neighbors}
                onChange={(v) => onParamChange("n_neighbors", v)}
              />
              <PropertySelect
                label="Metric"
                value={metric}
                onChange={(v) => onParamChange("metric", v)}
                options={["nan_euclidean"]}
              />
            </div>
          )}
        </>
      );
    }
    case ModuleType.TransformData: {
      const handlerConnection = allConnections.find(
        (c) => c.to.moduleId === module.id && c.to.portName === "handler_in"
      );
      const handlerSourceModule = handlerConnection
        ? allModules.find((m) => m.id === handlerConnection.from.moduleId)
        : undefined;
      const handler = handlerSourceModule?.outputData as
        | MissingHandlerOutput
        | EncoderOutput
        | NormalizerOutput
        | undefined;

      const sourceData = getConnectedDataSource(module.id, "data_in");
      const allColumns = sourceData?.columns || [];
      const { primary_exclude_column = "", exclude_columns = [] } =
        module.parameters;

      if (!sourceData)
        return (
          <p className="text-sm text-gray-500">
            Connect a data source to 'data_in'.
          </p>
        );
      if (!handler)
        return (
          <p className="text-sm text-gray-500">
            Connect a handler (e.g., Prep Missing) to 'handler_in'.
          </p>
        );

      const handleColumnToggle = (colName: string) => {
        const newExcludes = exclude_columns.includes(colName)
          ? exclude_columns.filter((c: string) => c !== colName)
          : [...exclude_columns, colName];
        onParamChange("exclude_columns", newExcludes);
      };

      const handleSelectAll = (isSelect: boolean) => {
        if (isSelect) {
          onParamChange(
            "exclude_columns",
            [primary_exclude_column].filter(Boolean)
          );
        } else {
          const allCols = allColumns.map((c) => c.name);
          onParamChange("exclude_columns", allCols);
        }
      };

      return (
        <div>
          <PropertySelect
            label="Exclude Column Permanently"
            value={primary_exclude_column || ""}
            onChange={(v) => {
              onParamChange("primary_exclude_column", v);
              if (v && !exclude_columns.includes(v)) {
                onParamChange("exclude_columns", [...exclude_columns, v]);
              }
            }}
            options={["", ...allColumns.map((c) => c.name)]}
          />
          <div className="mt-4">
            <div className="flex justify-between items-center mb-2">
              <h5 className="text-xs text-gray-500 uppercase font-bold">
                Columns to Transform
              </h5>
              <div className="flex gap-2">
                <button
                  onClick={() => handleSelectAll(true)}
                  className="px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded-md font-semibold"
                >
                  Select All
                </button>
                <button
                  onClick={() => handleSelectAll(false)}
                  className="px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded-md font-semibold"
                >
                  Deselect All
                </button>
              </div>
            </div>
            <div className="space-y-2 max-h-60 overflow-y-auto panel-scrollbar pr-2">
              {allColumns.map((col) => {
                let isDisabled = false;
                let disabledTooltip = "";
                if (col.name === primary_exclude_column) {
                  isDisabled = true;
                  disabledTooltip = "This column is permanently excluded.";
                } else if (
                  handler.type === "EncoderOutput" &&
                  col.type !== "string"
                ) {
                  isDisabled = true;
                  disabledTooltip = "Prep Encode only works on string columns.";
                } else if (
                  handler.type === "NormalizerOutput" &&
                  col.type !== "number"
                ) {
                  isDisabled = true;
                  disabledTooltip =
                    "Prep Normalize only works on number columns.";
                }
                return (
                  <label
                    key={col.name}
                    className={`flex items-center gap-2 text-sm truncate ${
                      isDisabled ? "opacity-50 cursor-not-allowed" : ""
                    }`}
                    title={disabledTooltip}
                  >
                    <input
                      type="checkbox"
                      checked={!exclude_columns.includes(col.name)}
                      onChange={() => handleColumnToggle(col.name)}
                      disabled={isDisabled}
                      className="h-4 w-4 rounded bg-gray-700 border-gray-600 text-blue-600 focus:ring-blue-500 disabled:cursor-not-allowed disabled:bg-gray-800"
                    />
                    <span className="truncate">{col.name}</span>
                  </label>
                );
              })}
            </div>
          </div>
        </div>
      );
    }
    case ModuleType.EncodeCategorical: {
      const sourceData = getConnectedDataSource(module.id);
      const categoricalColumns = (sourceData?.columns || []).filter(
        (c) => c.type === "string"
      );
      const {
        method,
        columns = [],
        handle_unknown,
        drop,
        ordinal_mapping,
      } = module.parameters;

      const handleColumnToggle = (colName: string) => {
        const newColumns = columns.includes(colName)
          ? columns.filter((c: string) => c !== colName)
          : [...columns, colName];
        onParamChange("columns", newColumns);
      };

      if (!sourceData) {
        return (
          <p className="text-sm text-gray-500">
            Connect a data source to see available columns.
          </p>
        );
      }
      if (categoricalColumns.length === 0) {
        return (
          <p className="text-sm text-gray-500">
            No categorical (string) columns found in input data.
          </p>
        );
      }

      return (
        <>
          <PropertySelect
            label="Method"
            value={method}
            onChange={(v) => onParamChange("method", v)}
            options={[
              { label: "One-Hot Encoding", value: "one_hot" },
              { label: "Ordinal Encoding", value: "ordinal" },
              { label: "Label Encoding", value: "label" },
            ]}
          />

          {method === "one_hot" && (
            <div className="mt-3 pt-3 border-t border-gray-700">
              <PropertySelect
                label="Drop"
                value={drop}
                onChange={(v) => onParamChange("drop", v === "None" ? null : v)}
                options={["first", "if_binary", "None"]}
              />
              <PropertySelect
                label="Handle Unknown"
                value={handle_unknown}
                onChange={(v) => onParamChange("handle_unknown", v)}
                options={["error", "ignore"]}
              />
            </div>
          )}

          {method === "ordinal" && (
            <div className="mt-3 pt-3 border-t border-gray-700">
              <label className="block text-sm text-gray-400 mb-1">
                Ordinal Mapping (JSON)
              </label>
              <textarea
                value={ordinal_mapping}
                onChange={(e) =>
                  onParamChange("ordinal_mapping", e.target.value)
                }
                placeholder={'{\n  "column_name": ["low", "medium", "high"]\n}'}
                className="w-full h-24 p-2 font-mono text-xs bg-gray-700 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <p className="text-xs text-gray-500 mt-1">
                Define the order of categories for each column. Unmapped columns
                will be ordered alphabetically.
              </p>
            </div>
          )}

          <div className="mt-4">
            <h5 className="text-xs text-gray-500 uppercase font-bold mb-2">
              COLUMNS TO ENCODE
            </h5>
            <p className="text-xs text-gray-500 mb-2">
              If none are selected, all string columns will be encoded.
            </p>
            <div className="space-y-2 max-h-48 overflow-y-auto panel-scrollbar pr-2">
              {categoricalColumns.map((col) => (
                <label
                  key={col.name}
                  className="flex items-center gap-2 text-sm truncate"
                  title={col.name}
                >
                  <input
                    type="checkbox"
                    checked={columns.includes(col.name)}
                    onChange={() => handleColumnToggle(col.name)}
                    className="h-4 w-4 rounded bg-gray-700 border-gray-600 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="truncate">{col.name}</span>
                </label>
              ))}
            </div>
          </div>
        </>
      );
    }
    case ModuleType.NormalizeData: {
      const sourceData = getConnectedDataSource(module.id);
      const inputColumns = sourceData?.columns || [];
      const currentSelections = module.parameters.columnSelections || {};

      const handleSelectionChange = (colName: string, value: boolean) => {
        const newSelections = {
          ...currentSelections,
          [colName]: {
            ...(currentSelections[colName] || { type: "string" }),
            selected: value,
          },
        };
        onParamChange("columnSelections", newSelections);
      };

      const handleSelectAll = (selectAll: boolean) => {
        const newSelections = { ...currentSelections };
        inputColumns.forEach((col) => {
          if (col.type === "number") {
            // Only affect numeric columns
            newSelections[col.name] = {
              ...(currentSelections[col.name] || { type: col.type }),
              selected: selectAll,
            };
          }
        });
        onParamChange("columnSelections", newSelections);
      };

      return (
        <>
          <PropertySelect
            label="Method"
            value={module.parameters.method}
            onChange={(v) => onParamChange("method", v)}
            options={["MinMax", "StandardScaler", "RobustScaler"]}
          />

          {inputColumns.length === 0 ? (
            <p className="text-sm text-gray-500 mt-4">
              Connect a data source module to configure columns.
            </p>
          ) : (
            <div className="mt-4">
              <h5 className="text-xs text-gray-500 uppercase font-bold mb-2">
                COLUMNS TO NORMALIZE
              </h5>
              <div className="flex justify-end gap-2 mb-2">
                <button
                  onClick={() => handleSelectAll(true)}
                  className="px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded-md font-semibold"
                >
                  Select All Numeric
                </button>
                <button
                  onClick={() => handleSelectAll(false)}
                  className="px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded-md font-semibold"
                >
                  Deselect All
                </button>
              </div>
              <div className="space-y-2 pr-2">
                <div className="grid grid-cols-5 gap-2 items-center sticky top-0 bg-gray-800 py-1">
                  <span className="text-xs font-bold text-gray-400 col-span-3">
                    Column Name
                  </span>
                  <span className="text-xs font-bold text-gray-400 col-span-2">
                    Data Type
                  </span>
                </div>
                {inputColumns.map((col) => {
                  const selection = currentSelections[col.name] || {
                    selected: false,
                    type: col.type,
                  };
                  return (
                    <div
                      key={col.name}
                      className="grid grid-cols-5 gap-2 items-center"
                    >
                      <label
                        className="flex items-center gap-2 text-sm truncate col-span-3"
                        title={col.name}
                      >
                        <input
                          type="checkbox"
                          checked={selection.selected}
                          onChange={(e) =>
                            handleSelectionChange(col.name, e.target.checked)
                          }
                          className="h-4 w-4 rounded bg-gray-700 border-gray-600 text-blue-600 focus:ring-blue-500 disabled:cursor-not-allowed"
                          disabled={col.type !== "number"}
                        />
                        <span
                          className={`truncate ${
                            col.type !== "number" ? "text-gray-500" : ""
                          }`}
                        >
                          {col.name}
                        </span>
                      </label>
                      <div className="col-span-2">
                        <span className="text-xs bg-gray-700 text-gray-300 px-2 py-1 rounded-md">
                          {col.type}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </>
      );
    }
    case ModuleType.TransitionData: {
      const sourceData = getConnectedDataSource(module.id);
      const numericColumns = (sourceData?.columns || []).filter(
        (c) => c.type === "number"
      );
      const transformations = module.parameters.transformations || {};

      const handleTransformChange = (colName: string, method: string) => {
        const newTransforms = { ...transformations, [colName]: method };
        onParamChange("transformations", newTransforms);
      };

      if (!sourceData) {
        return (
          <p className="text-sm text-gray-500">
            Connect a data source module to configure columns.
          </p>
        );
      }

      const formulaTooltip = `Formulas:\n- Log: log(x)\n- Square Root: sqrt(x)\n- Min-Log: log((x - min) + 1)\n- Min-Square Root: sqrt((x - min) + 1)`;

      return (
        <div>
          <div className="flex justify-between items-center mb-2">
            <h5 className="text-xs text-gray-500 uppercase font-bold">
              Column Transformations
            </h5>
            <div title={formulaTooltip}>
              <InformationCircleIcon className="w-5 h-5 text-gray-400 cursor-help" />
            </div>
          </div>
          {numericColumns.length === 0 ? (
            <p className="text-sm text-gray-500">
              No numeric columns found in the input data.
            </p>
          ) : (
            <div className="space-y-3">
              {numericColumns.map((col) => (
                <div
                  key={col.name}
                  className="grid grid-cols-2 gap-2 items-center"
                >
                  <label className="text-sm truncate" title={col.name}>
                    {col.name}
                  </label>
                  <select
                    value={transformations[col.name] || "None"}
                    onChange={(e) =>
                      handleTransformChange(col.name, e.target.value)
                    }
                    className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-blue-500"
                  >
                    <option value="None">None</option>
                    <option value="Log">Log</option>
                    <option value="Square Root">Square Root</option>
                    <option value="Min-Log">Min-Log</option>
                    <option value="Min-Square Root">Min-Square Root</option>
                  </select>
                </div>
              ))}
            </div>
          )}
        </div>
      );
    }
    case ModuleType.ResampleData: {
      const sourceData = getConnectedDataSource(module.id);
      const inputColumns = sourceData?.columns || [];

      if (inputColumns.length === 0) {
        return (
          <p className="text-sm text-gray-500">
            Connect a data source to configure resampling.
          </p>
        );
      }

      return (
        <>
          <PropertySelect
            label="Method"
            value={module.parameters.method}
            onChange={(v) => onParamChange("method", v)}
            options={["SMOTE", "NearMiss"]}
          />
          <PropertySelect
            label="Target Column"
            value={module.parameters.target_column || ""}
            onChange={(v) =>
              onParamChange("target_column", v === "" ? null : v)
            }
            options={["", ...inputColumns.map((c) => c.name)]}
          />
        </>
      );
    }
    case ModuleType.SplitData: {
      const sourceData = getConnectedDataSource(module.id);
      const inputColumns = sourceData?.columns || [];

      if (inputColumns.length === 0 && !sourceData) {
        return (
          <p className="text-sm text-gray-500">
            Connect a data source module to configure parameters.
          </p>
        );
      }

      return (
        <>
          <PropertyInput
            label="Train Size"
            type="number"
            value={module.parameters.train_size}
            onChange={(v) => onParamChange("train_size", v)}
          />
          <PropertyInput
            label="Random State"
            type="number"
            value={module.parameters.random_state}
            onChange={(v) => onParamChange("random_state", v)}
          />
          <PropertySelect
            label="Shuffle"
            value={module.parameters.shuffle}
            onChange={(v) => onParamChange("shuffle", v)}
            options={["True", "False"]}
          />
          <PropertySelect
            label="Stratify"
            value={module.parameters.stratify}
            onChange={(v) => onParamChange("stratify", v)}
            options={["False", "True"]}
          />
          {module.parameters.stratify === "True" && (
            <PropertySelect
              label="Stratify by Column"
              value={module.parameters.stratify_column}
              onChange={(v) => onParamChange("stratify_column", v)}
              options={["None", ...inputColumns.map((c) => c.name)]}
            />
          )}
        </>
      );
    }
    case ModuleType.LinearRegression:
      return (
        <>
          <PropertyDisplay label="Model Purpose" value="Regression" />
          <PropertySelect
            label="Model Type"
            value={module.parameters.model_type}
            onChange={(v) => onParamChange("model_type", v)}
            options={["LinearRegression", "Lasso", "Ridge", "ElasticNet"]}
          />
          <PropertySelect
            label="Fit Intercept"
            value={module.parameters.fit_intercept}
            onChange={(v) => onParamChange("fit_intercept", v)}
            options={["True", "False"]}
          />
          {["Lasso", "Ridge", "ElasticNet"].includes(
            module.parameters.model_type
          ) && (
            <PropertyInput
              label="Alpha (Regularization)"
              type="number"
              value={module.parameters.alpha}
              onChange={(v) => onParamChange("alpha", v)}
              step="0.1"
            />
          )}
          {module.parameters.model_type === "ElasticNet" && (
            <PropertyInput
              label="L1 Ratio"
              type="number"
              value={module.parameters.l1_ratio}
              onChange={(v) => onParamChange("l1_ratio", v)}
              step="0.1"
            />
          )}
          <PropertySelect
            label="Hyperparameter Tuning"
            value={module.parameters.tuning_enabled || "False"}
            onChange={(v) => onParamChange("tuning_enabled", v)}
            options={["False", "True"]}
          />
          {module.parameters.tuning_enabled === "True" && (
            <>
              <PropertySelect
                label="Tuning Strategy"
                value={module.parameters.tuning_strategy || "GridSearch"}
                onChange={(v) => onParamChange("tuning_strategy", v)}
                options={["GridSearch"]}
              />
              <PropertyInput
                label="Alpha Candidates (comma-separated)"
                type="text"
                value={module.parameters.alpha_candidates || "0.01,0.1,1,10"}
                onChange={(v) => onParamChange("alpha_candidates", v)}
              />
              {module.parameters.model_type === "ElasticNet" && (
                <PropertyInput
                  label="L1 Ratio Candidates (comma-separated)"
                  type="text"
                  value={module.parameters.l1_ratio_candidates || "0.2,0.5,0.8"}
                  onChange={(v) => onParamChange("l1_ratio_candidates", v)}
                />
              )}
              <PropertyInput
                label="CV Folds"
                type="number"
                min="2"
                value={module.parameters.cv_folds ?? 5}
                onChange={(v) => onParamChange("cv_folds", v)}
              />
              <PropertySelect
                label="Scoring Metric"
                value={
                  module.parameters.scoring_metric || "neg_mean_squared_error"
                }
                onChange={(v) => onParamChange("scoring_metric", v)}
                options={[
                  "neg_mean_squared_error",
                  "neg_mean_absolute_error",
                  "r2",
                ]}
              />
            </>
          )}
        </>
      );
    case ModuleType.TrainModel:
    case ModuleType.ResultModel: {
      const sourceData = getConnectedDataSource(module.id);
      const inputColumns = sourceData?.columns || [];

      if (inputColumns.length === 0) {
        return (
          <p className="text-sm text-gray-500">
            Connect a data source to the 'data_in' port to configure.
          </p>
        );
      }

      const { feature_columns = [], label_column = null } = module.parameters;

      const handleFeatureChange = (colName: string, isChecked: boolean) => {
        const newFeatures = isChecked
          ? [...feature_columns, colName]
          : feature_columns.filter((c: string) => c !== colName);
        onParamChange("feature_columns", newFeatures);
      };

      const handleLabelChange = (colName: string) => {
        const newLabel = colName === "" ? null : colName;
        onParamChange("label_column", newLabel);
        // If the new label was a feature, unselect it as a feature
        if (newLabel && feature_columns.includes(newLabel)) {
          onParamChange(
            "feature_columns",
            feature_columns.filter((c: string) => c !== newLabel)
          );
        }
      };

      const handleSelectAllFeatures = (selectAll: boolean) => {
        if (selectAll) {
          const allFeatureCols = inputColumns
            .map((col) => col.name)
            .filter((name) => name !== label_column);
          onParamChange("feature_columns", allFeatureCols);
        } else {
          onParamChange("feature_columns", []);
        }
      };

      return (
        <div>
          <AIParameterRecommender
            module={module}
            inputColumns={inputColumns.map((c) => c.name)}
            projectName={projectName}
            updateModuleParameters={updateModuleParameters}
          />
          <div className="mb-4">
            <h5 className="text-xs text-gray-500 uppercase font-bold mb-2">
              Label Column
            </h5>
            <select
              value={label_column || ""}
              onChange={(e) => handleLabelChange(e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">공백</option>
              {inputColumns.map((col) => (
                <option key={col.name} value={col.name}>
                  {col.name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <div className="flex justify-between items-center mb-2">
              <h5 className="text-xs text-gray-500 uppercase font-bold">
                Feature Columns
              </h5>
              <div className="flex gap-2">
                <button
                  onClick={() => handleSelectAllFeatures(true)}
                  className="px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded-md font-semibold"
                >
                  Select All
                </button>
                <button
                  onClick={() => handleSelectAllFeatures(false)}
                  className="px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded-md font-semibold"
                >
                  Deselect All
                </button>
              </div>
            </div>
            <div className="space-y-2 pr-2">
              {inputColumns.map((col) => (
                <label
                  key={col.name}
                  className="flex items-center gap-2 text-sm truncate"
                  title={col.name}
                >
                  <input
                    type="checkbox"
                    checked={feature_columns.includes(col.name)}
                    onChange={(e) =>
                      handleFeatureChange(col.name, e.target.checked)
                    }
                    disabled={col.name === label_column}
                    className="h-4 w-4 rounded bg-gray-700 border-gray-600 text-blue-600 focus:ring-blue-500 disabled:opacity-50"
                  />
                  <span className="truncate">{col.name}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
      );
    }
    case ModuleType.DiversionChecker: {
      const sourceData = getConnectedDataSource(module.id);
      const inputColumns = sourceData?.columns || [];

      if (inputColumns.length === 0) {
        return (
          <p className="text-sm text-gray-500">
            Connect a data source to the 'data_in' port to configure.
          </p>
        );
      }

      const { feature_columns = [], label_column = null, max_iter = 100 } = module.parameters;

      const handleFeatureChange = (colName: string, isChecked: boolean) => {
        const newFeatures = isChecked
          ? [...feature_columns, colName]
          : feature_columns.filter((c: string) => c !== colName);
        onParamChange("feature_columns", newFeatures);
      };

      const handleLabelChange = (colName: string) => {
        const newLabel = colName === "" ? null : colName;
        onParamChange("label_column", newLabel);
        // If the new label was a feature, unselect it as a feature
        if (newLabel && feature_columns.includes(newLabel)) {
          onParamChange(
            "feature_columns",
            feature_columns.filter((c: string) => c !== newLabel)
          );
        }
      };

      const handleSelectAllFeatures = (selectAll: boolean) => {
        if (selectAll) {
          const allFeatureCols = inputColumns
            .map((col) => col.name)
            .filter((name) => name !== label_column);
          onParamChange("feature_columns", allFeatureCols);
        } else {
          onParamChange("feature_columns", []);
        }
      };

      return (
        <div>
          <AIParameterRecommender
            module={module}
            inputColumns={inputColumns.map((c) => c.name)}
            projectName={projectName}
            updateModuleParameters={updateModuleParameters}
          />
          <div className="mb-4">
            <h5 className="text-xs text-gray-500 uppercase font-bold mb-2">
              Label Column
            </h5>
            <select
              value={label_column || ""}
              onChange={(e) => handleLabelChange(e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">공백</option>
              {inputColumns.map((col) => (
                <option key={col.name} value={col.name}>
                  {col.name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <div className="flex justify-between items-center mb-2">
              <h5 className="text-xs text-gray-500 uppercase font-bold">
                Feature Columns
              </h5>
              <div className="flex gap-2">
                <button
                  onClick={() => handleSelectAllFeatures(true)}
                  className="px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded-md font-semibold"
                >
                  Select All
                </button>
                <button
                  onClick={() => handleSelectAllFeatures(false)}
                  className="px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded-md font-semibold"
                >
                  Deselect All
                </button>
              </div>
            </div>
            <div className="space-y-2 pr-2">
              {inputColumns.map((col) => (
                <label
                  key={col.name}
                  className="flex items-center gap-2 text-sm truncate"
                  title={col.name}
                >
                  <input
                    type="checkbox"
                    checked={feature_columns.includes(col.name)}
                    onChange={(e) =>
                      handleFeatureChange(col.name, e.target.checked)
                    }
                    disabled={col.name === label_column}
                    className="h-4 w-4 rounded bg-gray-700 border-gray-600 text-blue-600 focus:ring-blue-500 disabled:opacity-50"
                  />
                  <span className="truncate">{col.name}</span>
                </label>
              ))}
            </div>
          </div>
          <div className="mt-4">
            <PropertyInput
              label="Max Iterations"
              type="number"
              value={max_iter}
              onChange={(v) => onParamChange("max_iter", v)}
            />
          </div>
        </div>
      );
    }
    case ModuleType.EvaluateStat: {
      const sourceData = getConnectedDataSource(module.id);
      const inputColumns = sourceData?.columns || [];

      if (inputColumns.length === 0) {
        return (
          <p className="text-sm text-gray-500">
            Connect a data source to the 'data_in' port to configure.
          </p>
        );
      }

      const { label_column = null, prediction_column = null, model_type = null } = module.parameters;

      const handleLabelChange = (colName: string) => {
        const newLabel = colName === "" ? null : colName;
        onParamChange("label_column", newLabel);
      };

      const handlePredictionChange = (colName: string) => {
        const newPrediction = colName === "" ? null : colName;
        onParamChange("prediction_column", newPrediction);
      };

      return (
        <div>
          <div className="mb-4">
            <h5 className="text-xs text-gray-500 uppercase font-bold mb-2">
              Label Column
            </h5>
            <select
              value={label_column || ""}
              onChange={(e) => handleLabelChange(e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">공백</option>
              {inputColumns.map((col) => (
                <option key={col.name} value={col.name}>
                  {col.name}
                </option>
              ))}
            </select>
          </div>
          <div className="mb-4">
            <h5 className="text-xs text-gray-500 uppercase font-bold mb-2">
              Prediction Column
            </h5>
            <select
              value={prediction_column || ""}
              onChange={(e) => handlePredictionChange(e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">공백</option>
              {inputColumns.map((col) => (
                <option key={col.name} value={col.name}>
                  {col.name}
                </option>
              ))}
            </select>
          </div>
          <div className="mb-4">
            <PropertySelect
              label="Model Type (Optional)"
              value={model_type || ""}
              onChange={(v) => onParamChange("model_type", v || null)}
              options={["", "OLS", "Logistic", "Logit", "Poisson", "QuasiPoisson", "NegativeBinomial", "Gamma", "Tweedie"]}
            />
            <p className="text-xs text-gray-500 mt-1">
              모델 타입을 선택하면 해당 모델의 특수 통계량(Deviance, AIC, BIC 등)이 계산됩니다. 선택하지 않으면 기본 통계량만 계산됩니다.
            </p>
          </div>
        </div>
      );
    }
    // ... [Rest of module types: KMeans, DBSCAN, LogisticRegression, DecisionTree, etc. remain unchanged] ...
    case ModuleType.KMeans:
    case ModuleType.HierarchicalClustering:
    case ModuleType.DBSCAN:
    case ModuleType.PrincipalComponentAnalysis: {
      const sourceData = getConnectedDataSource(module.id);
      const inputColumns = (sourceData?.columns || [])
        .filter((c) => c.type === "number")
        .map((c) => c.name);

      if (inputColumns.length === 0) {
        return (
          <p className="text-sm text-gray-500">
            Connect a data source with numeric columns to configure.
          </p>
        );
      }

      const { feature_columns = [] } = module.parameters;
      const otherParams = { ...module.parameters };
      delete otherParams.feature_columns;

      const handleFeatureChange = (colName: string, isChecked: boolean) => {
        const newFeatures = isChecked
          ? [...feature_columns, colName]
          : feature_columns.filter((c: string) => c !== colName);
        onParamChange("feature_columns", newFeatures);
      };

      const handleSelectAllFeatures = (selectAll: boolean) => {
        onParamChange("feature_columns", selectAll ? inputColumns : []);
      };

      return (
        <>
          {Object.entries(otherParams).map(([key, value]) => (
            <PropertyInput
              key={key}
              label={key}
              value={value}
              type="number"
              onChange={(v) => onParamChange(key, v)}
            />
          ))}
          <div>
            <div className="flex justify-between items-center mb-2">
              <h5 className="text-xs text-gray-500 uppercase font-bold">
                Feature Columns
              </h5>
              <div className="flex gap-2">
                <button
                  onClick={() => handleSelectAllFeatures(true)}
                  className="px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded-md font-semibold"
                >
                  Select All
                </button>
                <button
                  onClick={() => handleSelectAllFeatures(false)}
                  className="px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded-md font-semibold"
                >
                  Deselect All
                </button>
              </div>
            </div>
            <p className="text-xs text-gray-500 mb-2">
              If none are selected, all numeric columns will be used.
            </p>
            <div className="space-y-2 pr-2 max-h-40 overflow-y-auto panel-scrollbar">
              {inputColumns.map((col) => (
                <label
                  key={col}
                  className="flex items-center gap-2 text-sm truncate"
                  title={col}
                >
                  <input
                    type="checkbox"
                    checked={feature_columns.includes(col)}
                    onChange={(e) => handleFeatureChange(col, e.target.checked)}
                    className="h-4 w-4 rounded bg-gray-700 border-gray-600 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="truncate">{col}</span>
                </label>
              ))}
            </div>
          </div>
        </>
      );
    }
    case ModuleType.LogisticRegression:
      return (
        <>
          <PropertyDisplay label="Model Purpose" value="Classification" />
          <PropertySelect
            label="Penalty"
            value={module.parameters.penalty || "l2"}
            onChange={(v) => onParamChange("penalty", v)}
            options={["l1", "l2", "elasticnet", "none"]}
          />
          <PropertyInput
            label="C (Regularization)"
            type="number"
            value={module.parameters.C || 1.0}
            onChange={(v) => onParamChange("C", v)}
            step="0.1"
          />
          <PropertySelect
            label="Solver"
            value={module.parameters.solver || "lbfgs"}
            onChange={(v) => onParamChange("solver", v)}
            options={["lbfgs", "newton-cg", "liblinear", "sag", "saga"]}
          />
          <PropertyInput
            label="Max Iterations"
            type="number"
            value={module.parameters.max_iter || 100}
            onChange={(v) => onParamChange("max_iter", v)}
          />
          case ModuleType.PoissonRegression: return{" "}
          <>
            <PropertyDisplay
              label="Model Purpose"
              value="Regression (Count Data)"
            />
            <PropertySelect
              label="Distribution Type"
              value={module.parameters.distribution_type || "Poisson"}
              onChange={(v) => onParamChange("distribution_type", v)}
              options={["Poisson", "QuasiPoisson"]}
            />
            <PropertyInput
              label="Max Iterations"
              type="number"
              value={module.parameters.max_iter || 100}
              onChange={(v) => onParamChange("max_iter", v)}
            />
          </>
          ; case ModuleType.NegativeBinomialRegression: return{" "}
          <>
            <PropertyDisplay
              label="Model Purpose"
              value="Regression (Overdispersed Count Data)"
            />
            <PropertySelect
              label="Distribution Type"
              value={module.parameters.distribution_type || "NegativeBinomial"}
              onChange={(v) => onParamChange("distribution_type", v)}
              options={["NegativeBinomial", "QuasiPoisson"]}
            />
            <PropertyInput
              label="Max Iterations"
              type="number"
              value={module.parameters.max_iter || 100}
              onChange={(v) => onParamChange("max_iter", v)}
            />
            <PropertyInput
              label="Dispersion (alpha)"
              type="number"
              value={module.parameters.disp || 1.0}
              onChange={(v) => onParamChange("disp", v)}
              step="0.1"
            />
          </>
          ;
          <PropertySelect
            label="Hyperparameter Tuning"
            value={module.parameters.tuning_enabled || "False"}
            onChange={(v) => onParamChange("tuning_enabled", v)}
            options={["False", "True"]}
          />
          {module.parameters.tuning_enabled === "True" && (
            <>
              <PropertySelect
                label="Tuning Strategy"
                value={module.parameters.tuning_strategy || "GridSearch"}
                onChange={(v) => onParamChange("tuning_strategy", v)}
                options={["GridSearch"]}
              />
              <PropertyInput
                label="C Candidates (comma-separated)"
                type="text"
                value={module.parameters.c_candidates || "0.01,0.1,1,10,100"}
                onChange={(v) => onParamChange("c_candidates", v)}
              />
              {module.parameters.penalty === "elasticnet" && (
                <PropertyInput
                  label="L1 Ratio Candidates (comma-separated)"
                  type="text"
                  value={module.parameters.l1_ratio_candidates || "0.2,0.5,0.8"}
                  onChange={(v) => onParamChange("l1_ratio_candidates", v)}
                />
              )}
              <PropertyInput
                label="CV Folds"
                type="number"
                min="2"
                value={module.parameters.cv_folds ?? 5}
                onChange={(v) => onParamChange("cv_folds", v)}
              />
              <PropertySelect
                label="Scoring Metric"
                value={module.parameters.scoring_metric || "accuracy"}
                onChange={(v) => onParamChange("scoring_metric", v)}
                options={["accuracy", "precision", "recall", "f1", "roc_auc"]}
              />
            </>
          )}
        </>
      );
    case ModuleType.NaiveBayes:
      return (
        <>
          <PropertyDisplay label="Model Purpose" value="Classification" />
          <PropertyInput
            label="Var Smoothing"
            type="number"
            value={module.parameters.var_smoothing}
            onChange={(v) => onParamChange("var_smoothing", v)}
            step="1e-10"
          />
        </>
      );
    case ModuleType.DecisionTree: {
      const purpose = module.parameters.model_purpose;
      const handlePurposeChange = (newPurpose: string) => {
        onParamChange("model_purpose", newPurpose);
        if (
          newPurpose === "classification" &&
          !["gini", "entropy"].includes(module.parameters.criterion)
        ) {
          onParamChange("criterion", "gini");
        } else if (
          newPurpose === "regression" &&
          ![
            "squared_error",
            "friedman_mse",
            "absolute_error",
            "poisson",
          ].includes(module.parameters.criterion)
        ) {
          onParamChange("criterion", "squared_error");
        }
      };
      return (
        <>
          <PropertySelect
            label="Model Purpose"
            value={purpose}
            onChange={handlePurposeChange}
            options={["classification", "regression"]}
          />
          <PropertyInput
            label="Max Depth"
            type="number"
            value={module.parameters.max_depth}
            onChange={(v) => onParamChange("max_depth", v)}
          />
          <PropertySelect
            label="Criterion"
            value={module.parameters.criterion}
            onChange={(v) => onParamChange("criterion", v)}
            options={
              purpose === "classification"
                ? ["gini", "entropy"]
                : ["squared_error", "friedman_mse", "absolute_error", "poisson"]
            }
          />
        </>
      );
    }
    case ModuleType.RandomForest: {
      const purpose = module.parameters.model_purpose;
      const handlePurposeChange = (newPurpose: string) => {
        onParamChange("model_purpose", newPurpose);
        if (
          newPurpose === "classification" &&
          !["gini", "entropy"].includes(module.parameters.criterion)
        ) {
          onParamChange("criterion", "gini");
        } else if (
          newPurpose === "regression" &&
          !["squared_error", "absolute_error", "poisson"].includes(
            module.parameters.criterion
          )
        ) {
          onParamChange("criterion", "squared_error");
        }
      };
      return (
        <>
          <PropertySelect
            label="Model Purpose"
            value={purpose}
            onChange={handlePurposeChange}
            options={["classification", "regression"]}
          />
          <PropertyInput
            label="N Estimators"
            type="number"
            value={module.parameters.n_estimators}
            onChange={(v) => onParamChange("n_estimators", v)}
          />
          <PropertyInput
            label="Max Depth"
            type="number"
            value={module.parameters.max_depth}
            onChange={(v) => onParamChange("max_depth", v)}
          />
          <PropertySelect
            label="Criterion"
            value={module.parameters.criterion}
            onChange={(v) => onParamChange("criterion", v)}
            options={
              purpose === "classification"
                ? ["gini", "entropy"]
                : ["squared_error", "absolute_error", "poisson"]
            }
          />
        </>
      );
    }
    case ModuleType.SVM:
      return (
        <>
          <PropertySelect
            label="Model Purpose"
            value={module.parameters.model_purpose}
            onChange={(v) => onParamChange("model_purpose", v)}
            options={["classification", "regression"]}
          />
          <PropertyInput
            label="C (Regularization)"
            type="number"
            value={module.parameters.C}
            onChange={(v) => onParamChange("C", v)}
            step="0.1"
          />
          <PropertySelect
            label="Kernel"
            value={module.parameters.kernel}
            onChange={(v) => onParamChange("kernel", v)}
            options={["rbf", "linear", "poly", "sigmoid"]}
          />
          <PropertyInput
            label="Gamma"
            value={module.parameters.gamma}
            onChange={(v) => onParamChange("gamma", v)}
          />
        </>
      );
    case ModuleType.EvaluateModel: {
      const sourceData = getConnectedDataSource(module.id);
      const inputColumns = sourceData?.columns?.map((c) => c.name) || [];

      if (inputColumns.length === 0) {
        return (
          <p className="text-sm text-gray-500">
            Connect a scored data module to configure evaluation.
          </p>
        );
      }

      // module.parameters가 undefined일 수 있으므로 안전하게 처리
      const params = module.parameters || {};

      // 연결된 Train Model을 찾아서 모델 타입 감지 및 기본값 설정
      // allConnections와 allModules를 사용해야 함
      const inputConnection = allConnections.find(
        (c) => c.to.moduleId === module.id
      );
      let detectedModelType: "classification" | "regression" | null = null;
      let trainModelLabelColumn: string | null = null;

      if (inputConnection) {
        const sourceModule = allModules.find(
          (m) => m.id === inputConnection.from.moduleId
        );

        // Score Model인 경우, 그 Score Model이 연결된 Train Model 찾기
        if (sourceModule?.type === ModuleType.ScoreModel) {
          const modelInputConnection = allConnections.find(
            (c) =>
              c.to.moduleId === sourceModule.id && c.to.portName === "model_in"
          );
          if (modelInputConnection) {
            const trainModelModule = allModules.find(
              (m) =>
                m.id === modelInputConnection.from.moduleId &&
                m.outputData?.type === "TrainedModelOutput"
            );
            if (trainModelModule?.outputData?.type === "TrainedModelOutput") {
              const trainedModel = trainModelModule.outputData;
              trainModelLabelColumn = trainedModel.labelColumn;
              if (trainedModel.modelPurpose) {
                detectedModelType = trainedModel.modelPurpose;
              } else {
                // modelType으로 분류 모델인지 확인 (간단한 체크)
                const classificationTypes = [
                  ModuleType.LogisticRegression,
                  ModuleType.LinearDiscriminantAnalysis,
                  ModuleType.NaiveBayes,
                ];
                detectedModelType = classificationTypes.includes(
                  trainedModel.modelType
                )
                  ? "classification"
                  : "regression";
              }
            }
          }
        }
      }

      const isClassification =
        detectedModelType === "classification" ||
        params.model_type === "classification";
      const thresholdValue = params.threshold ?? 0.5;

      return (
        <>
          <PropertySelect
            label="Actual Label Column"
            value={params.label_column || ""}
            onChange={(v) => onParamChange("label_column", v)}
            options={["", ...inputColumns]}
          />
          <PropertySelect
            label="Prediction Column"
            value={params.prediction_column || ""}
            onChange={(v) => onParamChange("prediction_column", v)}
            options={["", ...inputColumns]}
          />
          <PropertySelect
            label="Model Type"
            value={params.model_type || "regression"}
            onChange={(v) => onParamChange("model_type", v)}
            options={["regression", "classification"]}
          />
          {isClassification && (
            <PropertyInput
              label="Threshold"
              type="number"
              value={typeof thresholdValue === "number" ? thresholdValue : 0.5}
              onChange={(v) => {
                const newThreshold =
                  typeof v === "number" ? Math.round(v * 10) / 10 : 0.5; // 0.1 단위로 반올림
                onParamChange("threshold", newThreshold);
              }}
              step="0.1"
              min="0"
              max="1"
            />
          )}
        </>
      );
    }
    case ModuleType.OLSModel:
      return (
        <p className="text-sm text-gray-500">
          OLS (Ordinary Least Squares) 모델입니다. 파라미터 설정이 필요 없습니다.
        </p>
      );
    case ModuleType.LogisticModel:
      return (
        <p className="text-sm text-gray-500">
          Logistic 회귀 모델입니다. 파라미터 설정이 필요 없습니다.
        </p>
      );
    case ModuleType.PoissonModel:
      return (
        <PropertyInput
          label="Max Iterations"
          type="number"
          value={module.parameters.max_iter || 100}
          onChange={(v) => onParamChange("max_iter", v)}
        />
      );
    case ModuleType.QuasiPoissonModel:
      return (
        <PropertyInput
          label="Max Iterations"
          type="number"
          value={module.parameters.max_iter || 100}
          onChange={(v) => onParamChange("max_iter", v)}
        />
      );
    case ModuleType.NegativeBinomialModel:
      return (
        <>
          <PropertyInput
            label="Max Iterations"
            type="number"
            value={module.parameters.max_iter || 100}
            onChange={(v) => onParamChange("max_iter", v)}
          />
          <PropertyInput
            label="Dispersion"
            type="number"
            value={module.parameters.disp || 1.0}
            onChange={(v) => onParamChange("disp", v)}
            step="0.1"
          />
        </>
      );
    case ModuleType.StatModels:
      return (
        <PropertySelect
          label="Model Type"
          value={module.parameters.model}
          onChange={(v) => onParamChange("model", v)}
          options={["Gamma", "Tweedie"]}
        />
      );
    default:
      const hasParams = Object.keys(module.parameters).length > 0;
      if (!hasParams) {
        return (
          <p className="text-sm text-gray-500">
            This module has no configurable parameters.
          </p>
        );
      }
      return (
        <div>
          {Object.entries(module.parameters).map(([key, value]) => {
            if (typeof value === "boolean") {
              return (
                <PropertySelect
                  key={key}
                  label={key}
                  value={value ? "True" : "False"}
                  onChange={(v) => onParamChange(key, v === "True")}
                  options={["True", "False"]}
                />
              );
            }
            if (typeof value === "number") {
              return (
                <PropertyInput
                  key={key}
                  label={key}
                  value={value}
                  type="number"
                  onChange={(v) => onParamChange(key, v)}
                />
              );
            }
            return (
              <PropertyInput
                key={key}
                label={key}
                value={value}
                onChange={(v) => onParamChange(key, v)}
              />
            );
          })}
        </div>
      );
  }
};

const StatRow: React.FC<{ label: string; value: React.ReactNode }> = ({
  label,
  value,
}) => (
  <div className="flex justify-between items-center text-sm py-1.5 px-2 border-b border-gray-700 last:border-b-0">
    <span className="text-gray-400">{label}</span>
    <span
      className="font-mono text-gray-200 font-medium truncate"
      title={String(value)}
    >
      {value}
    </span>
  </div>
);

const ColumnInfoTable: React.FC<{
  columns: ColumnInfo[];
  highlights?: Record<string, { color?: string; strikethrough?: boolean }>;
}> = ({ columns, highlights = {} }) => (
  <div className="text-sm">
    {columns.map((col) => {
      const highlight = highlights[col.name] || {};
      const colorClass = highlight.color
        ? `text-${highlight.color}-400`
        : "text-gray-200";
      const strikethroughClass = highlight.strikethrough ? "line-through" : "";

      return (
        <div
          key={col.name}
          className="flex justify-between items-center py-1 px-2 border-b border-gray-700 last:border-b-0"
        >
          <span
            className={`font-mono truncate ${colorClass} ${strikethroughClass}`}
          >
            {col.name}
          </span>
          <span className="text-gray-500 font-mono">{col.type}</span>
        </div>
      );
    })}
  </div>
);

const DataStatsSummary: React.FC<{ data: DataPreview; title?: string }> = ({
  data,
  title,
}) => {
  const numericCols = useMemo(
    () => data.columns.filter((c) => c.type === "number"),
    [data.columns]
  );
  const rows = useMemo(() => data.rows || [], [data.rows]);

  return (
    <div>
      {title && (
        <h4 className="text-xs text-gray-500 uppercase font-bold mb-2">
          {title}
        </h4>
      )}
      <div className="space-y-2">
        {numericCols.map((col) => {
          const values = rows
            .map((r) => r[col.name] as number)
            .filter((v) => typeof v === "number" && !isNaN(v));
          if (values.length === 0) return null;
          const sum = values.reduce((a, b) => a + b, 0);
          const mean = sum / values.length;
          const stdDev = Math.sqrt(
            values
              .map((x) => Math.pow(x - mean, 2))
              .reduce((a, b) => a + b, 0) / values.length
          );
          return (
            <div key={col.name} className="bg-gray-800 p-2 rounded">
              <p className="font-semibold text-sm truncate">{col.name}</p>
              <div className="grid grid-cols-2 gap-x-2 text-xs">
                <StatRow label="Mean" value={mean.toFixed(2)} />
                <StatRow label="Std Dev" value={stdDev.toFixed(2)} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

const MissingValueSummary: React.FC<{ data: DataPreview; title?: string }> = ({
  data,
  title,
}) => {
  const missingCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    if (!data.columns) return [];
    data.columns.forEach((col) => {
      counts[col.name] = (data.rows || []).filter(
        (row) => row[col.name] == null || row[col.name] === ""
      ).length;
    });
    return Object.entries(counts);
  }, [data]);

  return (
    <div>
      {title && (
        <h4 className="text-xs text-gray-500 uppercase font-bold mb-2">
          {title}
        </h4>
      )}
      <StatRow label="Total Rows" value={data.totalRowCount.toLocaleString()} />
      <StatRow label="Total Columns" value={data.columns.length} />
      <h5 className="text-xs text-gray-500 uppercase font-bold my-2">
        Missing Values per Column
      </h5>
      {missingCounts.length > 0 ? (
        <div className="max-h-96 overflow-y-auto panel-scrollbar pr-2">
          {missingCounts.map(([name, count]) => {
            const percentage =
              data.totalRowCount > 0
                ? ((count / data.totalRowCount) * 100).toFixed(1)
                : "0.0";
            return (
              <StatRow
                key={name}
                label={name}
                value={`${count} (${percentage}%)`}
              />
            );
          })}
        </div>
      ) : (
        <p className="text-sm text-gray-500 text-center p-2">
          No columns in input data.
        </p>
      )}
    </div>
  );
};

const DataTableStats: React.FC<{
  data: DataPreview;
  title?: string;
  highlightedColumns?: string[];
}> = ({ data, title, highlightedColumns = [] }) => {
  const numericCols = useMemo(
    () => data.columns.filter((c) => c.type === "number"),
    [data.columns]
  );
  const rows = useMemo(() => data.rows || [], [data.rows]);

  const colStats = useMemo(() => {
    const stats: Record<
      string,
      { mean: number; std: number; min: number; max: number }
    > = {};
    numericCols.forEach((col) => {
      const values = rows
        .map((r) => r[col.name] as number)
        .filter((v) => typeof v === "number" && !isNaN(v));
      if (values.length === 0) return;
      const sum = values.reduce((a, b) => a + b, 0);
      const mean = sum / values.length;
      const std = Math.sqrt(
        values.map((x) => Math.pow(x - mean, 2)).reduce((a, b) => a + b, 0) /
          values.length
      );
      const min = Math.min(...values);
      const max = Math.max(...values);
      stats[col.name] = { mean, std, min, max };
    });
    return stats;
  }, [numericCols, rows]);

  return (
    <div>
      {title && (
        <h3 className="text-md font-semibold mb-2 text-gray-300">{title}</h3>
      )}
      <div className="bg-gray-900 rounded-lg overflow-hidden">
        <div className="text-gray-200">
          {/* Header */}
          <div className="grid grid-cols-5 gap-4 px-4 py-2 border-b border-gray-600 font-semibold text-sm text-gray-400">
            <div className="col-span-1">Column</div>
            <div className="text-right">Mean</div>
            <div className="text-right">Std Dev</div>
            <div className="text-right">Min</div>
            <div className="text-right">Max</div>
          </div>
          {/* Body */}
          <div className="max-h-96 overflow-y-auto panel-scrollbar">
            {Object.keys(colStats).map((colName) => {
              const stats = colStats[colName];
              const isHighlighted = highlightedColumns.includes(colName);
              return (
                <div
                  key={colName}
                  className="grid grid-cols-5 gap-4 px-4 py-2.5 text-sm border-b border-gray-800 last:border-b-0"
                >
                  <div
                    className={`font-mono truncate ${
                      isHighlighted ? "text-red-400" : ""
                    }`}
                    title={colName}
                  >
                    {colName}
                  </div>
                  <div className="font-mono text-right">
                    {stats.mean.toFixed(2)}
                  </div>
                  <div className="font-mono text-right">
                    {stats.std.toFixed(2)}
                  </div>
                  <div className="font-mono text-right">
                    {stats.min.toFixed(2)}
                  </div>
                  <div className="font-mono text-right">
                    {stats.max.toFixed(2)}
                  </div>
                </div>
              );
            })}
            {Object.keys(colStats).length === 0 && (
              <p className="text-sm text-gray-500 text-center p-4">
                No numeric columns to display stats for.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

const PanelModelMetrics: React.FC<{
  metrics: Record<string, string | number>;
}> = ({ metrics }) => (
  <div>
    <h3 className="text-md font-semibold mb-2 text-gray-300">
      Performance Metrics
    </h3>
    <div className="bg-gray-800 rounded-lg p-3 space-y-2">
      {Object.entries(metrics).map(([key, value]) => (
        <StatRow
          key={key}
          label={key}
          value={typeof value === "number" ? Number(value).toFixed(4) : value}
        />
      ))}
    </div>
  </div>
);

export const PropertiesPanel: React.FC<PropertiesPanelProps> = ({
  module,
  projectName,
  updateModuleParameters,
  updateModuleName,
  logs,
  modules,
  connections,
  activeTab,
  setActiveTab,
  onViewDetails,
  folderHandle,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const logContainerRef = useRef<HTMLDivElement>(null);
  const [activePreviewTab, setActivePreviewTab] = useState<"Input" | "Output">(
    "Input"
  );
  const [localModuleName, setLocalModuleName] = useState("");
  const [isCopied, setIsCopied] = useState(false);
  const [terminalHeight, setTerminalHeight] = useState(200);
  const resizableContainerRef = useRef<HTMLDivElement>(null);

  const handleTerminalResizeMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      const startY = e.clientY;
      const startHeight = terminalHeight;
      const container = resizableContainerRef.current;
      if (!container) return;

      document.body.style.cursor = "row-resize";
      document.body.style.userSelect = "none";

      const handleMouseMove = (moveEvent: globalThis.MouseEvent) => {
        const dy = moveEvent.clientY - startY;
        const newHeight = startHeight - dy;

        const minHeight = 80;
        const maxHeight = container.clientHeight - 150; // Leave 150px for the top panel

        if (newHeight >= minHeight && newHeight <= maxHeight) {
          setTerminalHeight(newHeight);
        }
      };

      const handleMouseUp = () => {
        document.body.style.cursor = "default";
        document.body.style.userSelect = "auto";
        window.removeEventListener("mousemove", handleMouseMove);
        window.removeEventListener("mouseup", handleMouseUp);
      };

      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);
    },
    [terminalHeight]
  );

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  useEffect(() => {
    if (module) {
      setLocalModuleName(module.name);
      setActivePreviewTab(
        module.status === ModuleStatus.Success ? "Output" : "Input"
      );
    }
  }, [module]);

  const handleParamChange = useCallback(
    (key: string, value: any) => {
      if (module) {
        updateModuleParameters(module.id, { [key]: value });
      }
    },
    [module, updateModuleParameters]
  );

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && module) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target?.result as string;
        updateModuleParameters(module.id, {
          source: file.name,
          fileContent: content,
        });
      };
      reader.readAsText(file);
    }
  };

  const handleLoadSample = (sample: { name: string; content: string }) => {
    if (module) {
      updateModuleParameters(module.id, {
        source: sample.name,
        fileContent: sample.content,
      });
    }
  };

  const handleNameInputBlur = () => {
    if (module && localModuleName.trim() && localModuleName !== module.name) {
      updateModuleName(module.id, localModuleName.trim());
    } else if (module) {
      setLocalModuleName(module.name); // revert if empty or unchanged
    }
  };

  const handleNameInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleNameInputBlur();
      e.currentTarget.blur();
    } else if (e.key === "Escape") {
      if (module) setLocalModuleName(module.name);
      e.currentTarget.blur();
    }
  };

  const codeSnippet = useMemo(() => getModuleCode(module, modules, connections), [module, modules, connections]);

  const handleCopyCode = useCallback(() => {
    if (codeSnippet) {
      navigator.clipboard
        .writeText(codeSnippet)
        .then(() => {
          setIsCopied(true);
          setTimeout(() => setIsCopied(false), 2000);
        })
        .catch((err) => {
          console.error("Failed to copy code: ", err);
        });
    }
  }, [codeSnippet]);

  const getConnectedModelSource = useCallback(
    (moduleId: string): CanvasModule | undefined => {
      const modelInputConnection = connections.find(
        (c) => c.to.moduleId === moduleId && c.to.portName === "model_in"
      );
      if (!modelInputConnection) return undefined;
      return modules.find((m) => m.id === modelInputConnection.from.moduleId);
    },
    [modules, connections]
  );

  const getConnectedDataSource = useCallback(
    (moduleId: string, portNameToFind?: string) => {
      return getConnectedDataSourceHelper(
        moduleId,
        modules,
        connections,
        portNameToFind
      );
    },
    [modules, connections]
  );

  const renderInputPreview = () => {
    if (!module) return null;

    const handlerConnection = connections.find(
      (c) => c.to.moduleId === module.id && c.to.portName === "handler_in"
    );
    const handlerSourceModule = handlerConnection
      ? modules.find((m) => m.id === handlerConnection.from.moduleId)
      : undefined;
    const handler = handlerSourceModule?.outputData as
      | MissingHandlerOutput
      | EncoderOutput
      | NormalizerOutput
      | undefined;

    if (module.type === ModuleType.TransformData) {
      if (!handler) {
        return (
          <div className="text-center text-gray-500 p-4">
            Connect a handler module to 'handler_in'.
          </div>
        );
      }
      return (
        <div>
          <h4 className="text-xs text-gray-500 uppercase font-bold mb-2">
            Input Handler
          </h4>
          <StatRow label="Type" value={handlerSourceModule?.name} />
          <StatRow label="Method" value={handler.method} />
          {(handler as MissingHandlerOutput).strategy && (
            <StatRow
              label="Strategy"
              value={(handler as MissingHandlerOutput).strategy}
            />
          )}
        </div>
      );
    }

    if (
      module.type === ModuleType.LoadData ||
      module.type === ModuleType.XolLoading
    ) {
      return (
        <StatRow label="File Name" value={module.parameters.source || "N/A"} />
      );
    }

    if (module.type === ModuleType.TrainModel) {
      const modelSource = getConnectedModelSource(module.id);
      if (!modelSource) {
        return (
          <div className="text-center text-gray-500 p-4">
            Connect a model module to 'model_in'.
          </div>
        );
      }

      // Linear Regression 모듈의 경우 model_type 파라미터에서 실제 모델 타입 가져오기
      let modelTypeDisplay = modelSource.type;
      if (modelSource.type === ModuleType.LinearRegression) {
        const modelTypeParam = modelSource.parameters?.model_type;
        if (modelTypeParam && typeof modelTypeParam === "string") {
          modelTypeDisplay = modelTypeParam;
        }
      } else if (modelSource.type === ModuleType.LogisticRegression) {
        // Logistic Regression의 경우 penalty와 C를 조합하여 표시
        const penalty = modelSource.parameters?.penalty || "l2";
        const C = modelSource.parameters?.C || 1.0;
        modelTypeDisplay = `LogisticRegression (${penalty}, C=${C})`;
      }

      return <StatRow label="Model Type" value={modelTypeDisplay} />;
    }

    const inputData = getConnectedDataSource(module.id);
    if (!inputData) {
      return (
        <div className="text-center text-gray-500 p-4">
          Input data not available. Connect a preceding module.
        </div>
      );
    }

    switch (module.type) {
      case ModuleType.ResampleData: {
        const targetColumn = module.parameters.target_column;
        if (!targetColumn)
          return (
            <p className="text-sm text-gray-500">
              Select a target column to see value counts.
            </p>
          );

        const counts: Record<string, number> = {};
        (inputData.rows || []).forEach((row) => {
          const key = String(row[targetColumn]);
          counts[key] = (counts[key] || 0) + 1;
        });

        return (
          <div>
            <h4 className="text-xs text-gray-500 uppercase font-bold mb-2">
              Value Counts for '{targetColumn}'
            </h4>
            {Object.keys(counts).length > 0 ? (
              Object.entries(counts).map(([key, value]) => (
                <StatRow key={key} label={key} value={value} />
              ))
            ) : (
              <p className="text-sm text-gray-500">No data to count.</p>
            )}
          </div>
        );
      }
      case ModuleType.Statistics:
      case ModuleType.SelectData:
        return <ColumnInfoTable columns={inputData.columns} />;
      case ModuleType.EncodeCategorical: {
        const categoricalColumns = inputData.columns.filter(
          (c) => c.type === "string"
        );
        return (
          <div>
            <h4 className="text-xs text-gray-500 uppercase font-bold mb-2">
              Categorical Columns Found
            </h4>
            {categoricalColumns.length > 0 ? (
              <ColumnInfoTable columns={categoricalColumns} />
            ) : (
              <p className="text-sm text-gray-500">
                No string columns to encode.
              </p>
            )}
          </div>
        );
      }
      case ModuleType.HandleMissingValues:
        return <MissingValueSummary data={inputData} />;
      case ModuleType.NormalizeData:
        return <DataTableStats data={inputData} />;
      case ModuleType.TransitionData:
        return <DataStatsSummary data={inputData} />;
      case ModuleType.SplitData:
      case ModuleType.ScoreModel:
        return (
          <div>
            <StatRow
              label="Total Rows"
              value={inputData.totalRowCount.toLocaleString()}
            />
            <StatRow label="Total Columns" value={inputData.columns.length} />
          </div>
        );
      default:
        return <ColumnInfoTable columns={inputData.columns} />;
    }
  };

  const renderOutputPreview = () => {
    if (
      !module ||
      module.status !== ModuleStatus.Success ||
      !module.outputData
    ) {
      return (
        <div className="text-center text-gray-500 p-4">
          Run the module successfully to see the output.
        </div>
      );
    }
    const outputData = module.outputData;

    const visualizableTypes = [
      "DataPreview",
      "StatisticsOutput",
      "SplitDataOutput",
      "TrainedModelOutput",
      "StatsModelsResultOutput",
      "XoLPriceOutput",
      "FinalXolPriceOutput",
      "EvaluationOutput",
      "KMeansOutput",
      "HierarchicalClusteringOutput",
      "PCAOutput",
      "DBSCANOutput",
    ];

    const canVisualize = () => {
      if (!module || !module.outputData) return false;
      if (visualizableTypes.includes(module.outputData.type)) return true;
      if (
        [
          "KMeansOutput",
          "HierarchicalClusteringOutput",
          "DBSCANOutput",
        ].includes(module.outputData.type)
      )
        return true;
      return false;
    };

    const renderTitle = (title: string) => (
      <h3 className="text-md font-semibold mb-2 text-gray-300">{title}</h3>
    );

    const previewContent = (() => {
      switch (module.type) {
        case ModuleType.LoadData:
          if (outputData.type === "DataPreview") {
            return (
              <>
                <h3 className="text-md font-semibold mb-2 text-gray-300">
                  Column Structure
                </h3>
                <ColumnInfoTable columns={outputData.columns} />
              </>
            );
          }
          break;
        case ModuleType.Statistics:
          if (outputData.type === "StatisticsOutput") {
            return (
              <div>
                {renderTitle("Column Statistics")}
                <div className="bg-gray-900 rounded-lg overflow-hidden">
                  <div className="text-gray-200">
                    {/* Header */}
                    <div className="grid grid-cols-4 gap-4 px-4 py-2 border-b border-gray-600 font-semibold text-sm text-gray-400">
                      <div>Column</div>
                      <div className="text-right">Mean</div>
                      <div className="text-right">Median</div>
                      <div className="text-right">nulls</div>
                    </div>
                    {/* Body */}
                    <div className="max-h-96 overflow-y-auto panel-scrollbar">
                      {Object.keys(outputData.stats).map((col) => {
                        const columnStats = outputData.stats[col];
                        return (
                          <div
                            key={col}
                            className="grid grid-cols-4 gap-4 px-4 py-2.5 text-sm border-b border-gray-800 last:border-b-0"
                          >
                            <div className="font-mono truncate" title={col}>
                              {col}
                            </div>
                            <div className="font-mono text-right">
                              {typeof columnStats.mean === "number" &&
                              !isNaN(columnStats.mean)
                                ? columnStats.mean.toFixed(2)
                                : "N/A"}
                            </div>
                            <div className="font-mono text-right">
                              {typeof columnStats["50%"] === "number" &&
                              !isNaN(columnStats["50%"])
                                ? columnStats["50%"].toFixed(2)
                                : "N/A"}
                            </div>
                            <div className="font-mono text-right">
                              {columnStats.nulls}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </div>
            );
          }
          break;
        case ModuleType.SelectData:
          if (outputData.type === "DataPreview") {
            const originalData = getConnectedDataSource(module.id);
            const originalCols = originalData?.columns.map((c) => c.name) || [];
            const newCols = outputData.columns.map((c) => c.name);
            const removedCols = originalCols.filter(
              (c) => !newCols.includes(c)
            );

            const highlights: Record<string, { strikethrough: boolean }> = {};
            removedCols.forEach((colName) => {
              highlights[colName] = { strikethrough: true };
            });

            return (
              <ColumnInfoTable
                columns={originalData?.columns || []}
                highlights={highlights}
              />
            );
          }
          break;
        case ModuleType.ResampleData: {
          const inputData = getConnectedDataSource(module.id);
          if (!inputData || outputData.type !== "DataPreview") break;

          const targetColumn = module.parameters.target_column;
          if (!targetColumn)
            return (
              <p className="text-sm text-gray-500">
                Select a target column to see value counts.
              </p>
            );

          const counts: Record<string, number> = {};
          (outputData.rows || []).forEach((row) => {
            const key = String(row[targetColumn]);
            counts[key] = (counts[key] || 0) + 1;
          });
          return (
            <div>
              <h4 className="text-xs text-gray-500 uppercase font-bold mb-2">
                Value Counts for '{targetColumn}'
              </h4>
              {Object.keys(counts).length > 0 ? (
                Object.entries(counts).map(([key, value]) => (
                  <StatRow key={key} label={key} value={value} />
                ))
              ) : (
                <p className="text-sm text-gray-500">No data to count.</p>
              )}
            </div>
          );
        }
        case ModuleType.SplitData:
          if (outputData.type === "SplitDataOutput") {
            return (
              <div className="space-y-4">
                <DataStatsSummary
                  data={outputData.train}
                  title="Train Data Summary"
                />
                <DataStatsSummary
                  data={outputData.test}
                  title="Test Data Summary"
                />
              </div>
            );
          }
          break;
        case ModuleType.TrainModel:
          if (outputData.type === "TrainedModelOutput") {
            const {
              modelType,
              coefficients,
              intercept,
              featureColumns,
              labelColumn,
            } = outputData;

            const complexModels = [
              ModuleType.DecisionTree,
              ModuleType.RandomForest,
              ModuleType.SVM,
              ModuleType.KNN,
              ModuleType.NaiveBayes,
              ModuleType.LinearDiscriminantAnalysis,
            ];

            let formulaParts: string[] = [];
            if (!complexModels.includes(modelType)) {
              if (modelType === ModuleType.LogisticRegression) {
                formulaParts = [`ln(p / (1 - p)) = ${intercept.toFixed(4)}`];
              } else {
                formulaParts = [`${labelColumn} ≈ ${intercept.toFixed(4)}`];
              }

              featureColumns.forEach((feature) => {
                const value = coefficients[feature];
                const coeff = typeof value === "number" ? value : 0;
                if (coeff >= 0) {
                  formulaParts.push(` + ${coeff.toFixed(4)} * [${feature}]`);
                } else {
                  formulaParts.push(
                    ` - ${Math.abs(coeff).toFixed(4)} * [${feature}]`
                  );
                }
              });
            }

            return (
              <div className="space-y-4">
                {formulaParts.length > 0 && (
                  <div>
                    <h4 className="text-xs text-gray-500 uppercase font-bold mb-2">
                      Model Equation
                    </h4>
                    <div className="bg-gray-900/50 p-3 rounded-lg font-mono text-xs text-green-700 whitespace-normal break-words">
                      <span>{formulaParts[0]}</span>
                      {formulaParts.slice(1).map((part, i) => (
                        <span key={i}>{part}</span>
                      ))}
                    </div>
                  </div>
                )}
                <PanelModelMetrics metrics={outputData.metrics} />
              </div>
            );
          }
          break;
        case ModuleType.EvaluateModel:
          if (outputData.type === "EvaluationOutput") {
            return <PanelModelMetrics metrics={outputData.metrics} />;
          }
          break;
        default:
          if (outputData.type === "DataPreview") {
            return (
              <MissingValueSummary
                data={outputData}
                title="Output Data Summary"
              />
            );
          }
          break;
      }
      return (
        <div className="text-center text-gray-500 p-4">
          No specific preview for this module's output.
        </div>
      );
    })();

    return (
      <div className="space-y-4">
        {previewContent}
        {canVisualize() && (
          <div className="mt-4 border-t border-gray-700 pt-4">
            <button
              onClick={() => onViewDetails(module.id)}
              className="w-full px-3 py-2 text-sm bg-blue-600 hover:bg-blue-700 rounded-md font-semibold text-white transition-colors"
            >
              View Details
            </button>
          </div>
        )}
      </div>
    );
  };

  return (
    <div
      ref={resizableContainerRef}
      className="bg-gray-800 text-white h-full flex flex-col"
    >
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept=".csv"
        className="hidden"
      />
      <div className="flex-grow flex flex-col min-h-0">
        <div className="p-3 border-b border-gray-700 flex-shrink-0">
          <input
            type="text"
            value={localModuleName}
            onChange={(e) => setLocalModuleName(e.target.value)}
            onBlur={handleNameInputBlur}
            onKeyDown={handleNameInputKeyDown}
            className="w-full bg-transparent text-lg font-bold focus:outline-none focus:bg-gray-700 rounded-md px-2 py-1 -ml-2"
            placeholder="Module Name"
            disabled={!module}
          />
          <p className="text-xs text-gray-500 mt-1">
            {module ? module.type : "No module selected"}
          </p>
        </div>

        {module && (
          <div className="flex-shrink-0 border-b border-gray-700">
            <div className="flex">
              <button
                onClick={() => setActiveTab("properties")}
                className={`flex-1 flex items-center justify-center gap-2 p-3 text-sm font-semibold ${
                  activeTab === "properties"
                    ? "bg-gray-700 text-white"
                    : "text-gray-400 hover:bg-gray-700/50"
                }`}
              >
                <CogIcon className="w-5 h-5" /> Properties
              </button>
              <button
                onClick={() => setActiveTab("preview")}
                className={`flex-1 flex items-center justify-center gap-2 p-3 text-sm font-semibold ${
                  activeTab === "preview"
                    ? "bg-gray-700 text-white"
                    : "text-gray-400 hover:bg-gray-700/50"
                }`}
              >
                <TableCellsIcon className="w-5 h-5" /> Preview
              </button>
              <button
                onClick={() => setActiveTab("code")}
                className={`flex-1 flex items-center justify-center gap-2 p-3 text-sm font-semibold ${
                  activeTab === "code"
                    ? "bg-gray-700 text-white"
                    : "text-gray-400 hover:bg-gray-700/50"
                }`}
              >
                <CodeBracketIcon className="w-5 h-5" /> Code
              </button>
            </div>
          </div>
        )}

        <div className="flex-grow overflow-y-auto panel-scrollbar p-3">
          {!module ? (
            <div className="flex items-center justify-center h-full">
              <p className="text-gray-500">
                Select a module to see its properties.
              </p>
            </div>
          ) : (
            <>
              {activeTab === "properties" && (
                <PropertyGroup title="Parameters" module={module}>
                  {renderParameters(
                    module,
                    handleParamChange,
                    fileInputRef,
                    modules,
                    connections,
                    projectName,
                    updateModuleParameters,
                    handleLoadSample,
                    folderHandle
                  )}
                </PropertyGroup>
              )}
              {activeTab === "preview" && (
                <div>
                  <div className="flex mb-3 rounded-md bg-gray-700 p-1">
                    <button
                      onClick={() => setActivePreviewTab("Input")}
                      className={`flex-1 text-center text-sm py-1 rounded-md transition-colors ${
                        activePreviewTab === "Input"
                          ? "bg-gray-600 font-semibold"
                          : "hover:bg-gray-600/50"
                      }`}
                    >
                      Input
                    </button>
                    <button
                      onClick={() => setActivePreviewTab("Output")}
                      className={`flex-1 text-center text-sm py-1 rounded-md transition-colors ${
                        activePreviewTab === "Output"
                          ? "bg-gray-600 font-semibold"
                          : "hover:bg-gray-600/50"
                      }`}
                    >
                      Output
                    </button>
                  </div>
                  <div className="bg-gray-900/50 p-3 rounded-lg">
                    {activePreviewTab === "Input"
                      ? renderInputPreview()
                      : renderOutputPreview()}
                  </div>
                </div>
              )}
              {activeTab === "code" && (
                <div>
                  <div className="relative bg-gray-900 rounded-lg">
                    <button
                      onClick={handleCopyCode}
                      className="absolute top-2 right-2 p-1.5 bg-gray-700 hover:bg-gray-600 rounded-md text-gray-300 transition-colors"
                      title="Copy to clipboard"
                    >
                      {isCopied ? (
                        <CheckIcon className="w-4 h-4 text-green-400" />
                      ) : (
                        <ClipboardIcon className="w-4 h-4" />
                      )}
                    </button>
                    <pre className="p-4 text-xs text-gray-300 overflow-x-auto">
                      <code>{codeSnippet}</code>
                    </pre>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      <div
        className="flex-shrink-0 flex flex-col"
        style={{ height: `${terminalHeight}px` }}
      >
        <div
          onMouseDown={handleTerminalResizeMouseDown}
          className="w-full h-1.5 cursor-row-resize bg-gray-700 hover:bg-blue-500 transition-colors"
          title="Resize Terminal"
        />
        <div className="p-2 border-b border-gray-700 bg-gray-900/50 flex items-center gap-2">
          <CommandLineIcon className="w-5 h-5 text-gray-400" />
          <h3 className="text-sm font-semibold text-gray-300">Terminal</h3>
        </div>
        <div
          ref={logContainerRef}
          className="flex-grow overflow-y-auto bg-gray-900 text-xs font-mono p-2 space-y-1"
          onContextMenu={(e) => {
            // 텍스트가 선택되어 있으면 컨텍스트 메뉴에서 복사 가능하도록
            const selection = window.getSelection();
            if (selection && selection.toString().trim()) {
              // 브라우저 기본 컨텍스트 메뉴 사용 (복사 옵션 포함)
              return;
            }
            // 텍스트가 선택되지 않았으면 기본 동작 방지
            e.preventDefault();
          }}
        >
          {logs.map((log) => (
            <div
              key={log.id}
              className="flex group hover:bg-gray-800/50 rounded px-1 py-0.5"
            >
              <span className="text-gray-500 mr-2 flex-shrink-0 select-none">
                {log.timestamp}
              </span>
              <span
                className={`mr-2 font-bold flex-shrink-0 select-none ${
                  log.level === "INFO"
                    ? "text-blue-400"
                    : log.level === "WARN"
                    ? "text-yellow-400"
                    : log.level === "ERROR"
                    ? "text-red-400"
                    : "text-green-400"
                }`}
              >
                {log.level}:
              </span>
              <span
                className="flex-1 whitespace-pre-wrap break-words cursor-text select-text"
                onDoubleClick={(e) => {
                  e.preventDefault();
                  const text = log.message;
                  navigator.clipboard.writeText(text).then(() => {
                    setIsCopied(true);
                    setTimeout(() => setIsCopied(false), 2000);
                  });
                }}
                onMouseUp={(e) => {
                  // 텍스트 선택 후 Ctrl+C 또는 우클릭으로 복사 가능
                  const selection = window.getSelection();
                  if (selection && selection.toString().trim()) {
                    // 선택된 텍스트가 있으면 복사 가능
                  }
                }}
                title="텍스트를 선택하여 복사하거나 더블클릭하여 전체 메시지 복사"
              >
                {log.message}
              </span>
              <button
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  const text = `${log.timestamp} ${log.level}: ${log.message}`;
                  navigator.clipboard.writeText(text).then(() => {
                    setIsCopied(true);
                    setTimeout(() => setIsCopied(false), 2000);
                  });
                }}
                className="ml-2 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0"
                title="전체 로그 복사"
              >
                {isCopied ? (
                  <CheckIcon className="w-4 h-4 text-green-400" />
                ) : (
                  <ClipboardIcon className="w-4 h-4 text-gray-400 hover:text-gray-300" />
                )}
              </button>
            </div>
          ))}
          {logs.length === 0 && (
            <div className="text-gray-500 text-center py-4">
              로그가 없습니다
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
