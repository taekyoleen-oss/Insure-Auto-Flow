import React, {
  useState,
  useCallback,
  MouseEvent,
  useEffect,
  useRef,
} from "react";
import { Toolbox } from "./components/Toolbox";
import { Canvas } from "./components/Canvas";
import { PropertiesPanel } from "./components/PropertiesPanel";
import { ErrorBoundary } from "./components/ErrorBoundary";
// fix: Add missing 'Port' type to handle portType argument in getSingleInputData.
import {
  CanvasModule,
  ModuleType,
  Connection,
  ModuleStatus,
  StatisticsOutput,
  DataPreview,
  ColumnInfo,
  SplitDataOutput,
  TrainedModelOutput,
  ModelDefinitionOutput,
  StatsModelsResultOutput,
  FittedDistributionOutput,
  ExposureCurveOutput,
  XoLPriceOutput,
  XolContractOutput,
  FinalXolPriceOutput,
  EvaluationOutput,
  KMeansOutput,
  HierarchicalClusteringOutput,
  PCAOutput,
  DBSCANOutput,
  MissingHandlerOutput,
  Port,
  EncoderOutput,
  NormalizerOutput,
  DiversionCheckerOutput,
  EvaluateStatOutput,
  StatsModelFamily,
} from "./types";
import { DEFAULT_MODULES, TOOLBOX_MODULES, SAMPLE_MODELS } from "./constants";
import { SAVED_SAMPLES } from "./savedSamples";

// SAVED_SAMPLES가 없을 경우를 대비한 기본값
const getSavedSamples = () => {
  try {
    return SAVED_SAMPLES || [];
  } catch (error) {
    console.error("Failed to load SAVED_SAMPLES:", error);
    return [];
  }
};
import {
  LogoIcon,
  PlayIcon,
  CodeBracketIcon,
  FolderOpenIcon,
  PlusIcon,
  MinusIcon,
  Bars3Icon,
  CogIcon,
  ArrowUturnLeftIcon,
  ArrowUturnRightIcon,
  SparklesIcon,
  ArrowsPointingOutIcon,
  Squares2X2Icon,
  CheckIcon,
  ArrowPathIcon,
  StarIcon,
} from "./components/icons";
import useHistoryState from "./hooks/useHistoryState";
import { DataPreviewModal } from "./components/DataPreviewModal";
import { StatisticsPreviewModal } from "./components/StatisticsPreviewModal";
import { SplitDataPreviewModal } from "./components/SplitDataPreviewModal";
import { TrainedModelPreviewModal } from "./components/TrainedModelPreviewModal";
import { StatsModelsResultPreviewModal } from "./components/StatsModelsResultPreviewModal";
import { DiversionCheckerPreviewModal } from "./components/DiversionCheckerPreviewModal";
import { EvaluateStatPreviewModal } from "./components/EvaluateStatPreviewModal";
import { XoLPricePreviewModal } from "./components/XoLPricePreviewModal";
import { FinalXolPricePreviewModal } from "./components/FinalXolPricePreviewModal";
import { EvaluationPreviewModal } from "./components/EvaluationPreviewModal";
import { AIPipelineFromGoalModal } from "./components/AIPipelineFromGoalModal";
import { AIPipelineFromDataModal } from "./components/AIPipelineFromDataModal";
import { AIPlanDisplayModal } from "./components/AIPlanDisplayModal";
import { PipelineCodePanel } from "./components/PipelineCodePanel";
import { GoogleGenAI, Type } from "@google/genai";
import { savePipeline, loadPipeline } from "../shared/utils/fileOperations";
import { loadSampleFromFolder, loadFolderSamples } from "../shared/utils/samples";

type TerminalLog = {
  id: number;
  level: "INFO" | "WARN" | "ERROR" | "SUCCESS";
  message: string;
  timestamp: string;
};

type PropertiesTab = "properties" | "preview" | "code";

// --- Helper Functions ---
// Note: All mathematical/statistical calculations are now performed using Pyodide (Python)
// JavaScript is only used for UI rendering and data structure transformations that don't modify Python results

// Sigmoid function for logistic regression predictions
const sigmoid = (x: number): number => {
  return 1 / (1 + Math.exp(-x));
};

// Helper function to determine model type
const isClassification = (
  modelType: ModuleType,
  modelPurpose?: "classification" | "regression"
): boolean => {
  const classificationTypes = [
    ModuleType.LogisticRegression,
    ModuleType.LinearDiscriminantAnalysis,
    ModuleType.NaiveBayes,
  ];
  const dualPurposeTypes = [
    ModuleType.KNN,
    ModuleType.DecisionTree,
    ModuleType.RandomForest,
    ModuleType.SVM,
  ];

  if (classificationTypes.includes(modelType)) {
    return true;
  }
  if (
    dualPurposeTypes.includes(modelType) &&
    modelPurpose === "classification"
  ) {
    return true;
  }
  return false;
};

// All regression and statistical calculations are now performed using Pyodide (Python)
// These JavaScript implementations have been removed to ensure Python-compatible results

const App: React.FC = () => {
  const [modules, setModules, undo, redo, resetModules, canUndo, canRedo] =
    useHistoryState<CanvasModule[]>([]);
  const [connections, _setConnections] = useState<Connection[]>([]);
  const [selectedModuleIds, setSelectedModuleIds] = useState<string[]>([]);
  const [terminalLogs, setTerminalLogs] = useState<TerminalLog[]>([]);
  const [projectName, setProjectName] = useState("Data Analysis");
  const [isEditingProjectName, setIsEditingProjectName] = useState(false);

  const [scale, setScale] = useState(0.8);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [viewingDataForModule, setViewingDataForModule] =
    useState<CanvasModule | null>(null);
  const [viewingSplitDataForModule, setViewingSplitDataForModule] =
    useState<CanvasModule | null>(null);
  const [viewingTrainedModel, setViewingTrainedModel] =
    useState<CanvasModule | null>(null);
  const [viewingStatsModelsResult, setViewingStatsModelsResult] =
    useState<CanvasModule | null>(null);
  const [viewingDiversionChecker, setViewingDiversionChecker] =
    useState<CanvasModule | null>(null);
  const [viewingEvaluateStat, setViewingEvaluateStat] =
    useState<CanvasModule | null>(null);
  const [viewingXoLPrice, setViewingXoLPrice] = useState<CanvasModule | null>(
    null
  );
  const [viewingFinalXolPrice, setViewingFinalXolPrice] =
    useState<CanvasModule | null>(null);
  const [viewingEvaluation, setViewingEvaluation] =
    useState<CanvasModule | null>(null);

  const [isAiGenerating, setIsAiGenerating] = useState(false);
  const [isGoalModalOpen, setIsGoalModalOpen] = useState(false);
  const [isDataModalOpen, setIsDataModalOpen] = useState(false);
  const [aiPlan, setAiPlan] = useState<string | null>(null);
  const [isSampleMenuOpen, setIsSampleMenuOpen] = useState(false);
  const sampleMenuRef = useRef<HTMLDivElement>(null);
  const [folderSamples, setFolderSamples] = useState<
    Array<{ filename: string; name: string; data: any }>
  >([]);
  const [isLoadingSamples, setIsLoadingSamples] = useState(false);
  const [isMyWorkMenuOpen, setIsMyWorkMenuOpen] = useState(false);
  const myWorkMenuRef = useRef<HTMLDivElement>(null);
  const [myWorkModels, setMyWorkModels] = useState<any[]>([]);

  const [isLeftPanelVisible, setIsLeftPanelVisible] = useState(false);
  const [isRightPanelVisible, setIsRightPanelVisible] = useState(false);
  const [isCodePanelVisible, setIsCodePanelVisible] = useState(false);
  const [activePropertiesTab, setActivePropertiesTab] =
    useState<PropertiesTab>("properties");
  const [rightPanelWidth, setRightPanelWidth] = useState(384); // w-96 in Tailwind is 384px

  const canvasContainerRef = useRef<HTMLDivElement>(null);
  const folderHandleRef = useRef<FileSystemDirectoryHandle | null>(null);
  const [suggestion, setSuggestion] = useState<{
    module: CanvasModule;
    connection: Connection;
  } | null>(null);
  const [clipboard, setClipboard] = useState<{
    modules: CanvasModule[];
    connections: Connection[];
  } | null>(null);
  const pasteOffset = useRef(0);

  const [isDirty, setIsDirty] = useState(false);
  const [saveButtonText, setSaveButtonText] = useState("Save");

  // Draggable control panel state
  const [controlPanelPos, setControlPanelPos] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const isDraggingControlPanel = useRef(false);
  const controlPanelDragOffset = useRef({ x: 0, y: 0 });

  const setConnections = useCallback(
    (value: React.SetStateAction<Connection[]>) => {
      const prevConnections = connections;
      const newConnections =
        typeof value === "function" ? value(prevConnections) : value;

      // If a connection to TrainModel is removed, mark connected model definition module as Pending
      const removedConnections = prevConnections.filter(
        (c) => !newConnections.some((nc) => nc.id === c.id)
      );

      removedConnections.forEach((removedConn) => {
        if (removedConn.to.moduleId) {
          const trainModelModule = modules.find(
            (m) =>
              m.id === removedConn.to.moduleId &&
              m.type === ModuleType.TrainModel
          );
          if (trainModelModule && removedConn.to.portName === "model_in") {
            const modelDefinitionModuleId = removedConn.from.moduleId;
            setModules((prev) =>
              prev.map((m) => {
                if (
                  m.id === modelDefinitionModuleId &&
                  MODEL_DEFINITION_TYPES.includes(m.type)
                ) {
                  return {
                    ...m,
                    status: ModuleStatus.Pending,
                    outputData: undefined,
                  };
                }
                return m;
              })
            );
          }
        }
      });

      _setConnections(newConnections);
      setIsDirty(true);
    },
    [connections, modules, setModules]
  );

  // fix: Moved 'addLog' before 'handleSuggestModule' to fix "used before its declaration" error.
  const addLog = useCallback((level: TerminalLog["level"], message: string) => {
    setTerminalLogs((prev) => [
      ...prev,
      {
        id: Date.now(),
        level,
        message,
        timestamp: new Date().toLocaleTimeString(),
      },
    ]);
    if (level === "ERROR" || level === "WARN") {
      setIsRightPanelVisible(true);
    }
  }, []);

  const handleSuggestModule = useCallback(
    async (fromModuleId: string, fromPortName: string) => {
      clearSuggestion();
      const fromModule = modules.find((m) => m.id === fromModuleId);
      if (!fromModule) return;

      setIsAiGenerating(true);
      addLog(
        "INFO",
        `AI is suggesting a module to connect to '${fromModule.name}'...`
      );
      try {
        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });
        const fromPort = fromModule.outputs.find(
          (p) => p.name === fromPortName
        );
        if (!fromPort) throw new Error("Source port not found.");

        const availableModuleTypes = TOOLBOX_MODULES.map((m) => m.type).join(
          ", "
        );

        const prompt = `Given a module of type '${fromModule.type}' with an output port of type '${fromPort.type}', what is the single most logical module type to connect next?
Available module types: [${availableModuleTypes}].
Respond with ONLY the module type string, for example: 'ScoreModel'`;

        const response = await ai.models.generateContent({
          model: "gemini-2.5-flash",
          contents: prompt,
        });

        const suggestedType = response.text.trim() as ModuleType;
        const defaultModule = DEFAULT_MODULES.find(
          (m) => m.type === suggestedType
        );
        if (!defaultModule) {
          throw new Error(
            `AI suggested an unknown module type: '${suggestedType}'`
          );
        }

        const count =
          modules.filter((m) => m.type === suggestedType).length + 1;
        const newModule: CanvasModule = {
          id: `suggestion-${suggestedType}-${Date.now()}`,
          name: `${suggestedType} ${count}`,
          type: suggestedType,
          position: {
            x: fromModule.position.x,
            y: fromModule.position.y + 180,
          },
          status: ModuleStatus.Pending,
          parameters: { ...defaultModule.parameters },
          inputs: [...defaultModule.inputs],
          outputs: [...defaultModule.outputs],
        };

        const toPort = newModule.inputs.find((p) => p.type === fromPort.type);
        if (!toPort) {
          throw new Error(
            `Suggested module '${suggestedType}' has no compatible input port for type '${fromPort.type}'.`
          );
        }

        const newConnection: Connection = {
          id: `suggestion-conn-${Date.now()}`,
          from: { moduleId: fromModuleId, portName: fromPortName },
          to: { moduleId: newModule.id, portName: toPort.name },
        };

        setSuggestion({ module: newModule, connection: newConnection });
        addLog(
          "SUCCESS",
          `AI suggested connecting a '${suggestedType}' module.`
        );
      } catch (error: any) {
        console.error("AI suggestion failed:", error);
        addLog("ERROR", `AI suggestion failed: ${error.message}`);
      } finally {
        setIsAiGenerating(false);
      }
    },
    [modules, addLog]
  );

  const acceptSuggestion = useCallback(() => {
    if (suggestion) {
      const newModuleId = suggestion.module.id.replace("suggestion-", "");
      const newConnectionId = suggestion.connection.id.replace(
        "suggestion-",
        ""
      );

      const finalModule = { ...suggestion.module, id: newModuleId };
      const finalConnection = {
        ...suggestion.connection,
        id: newConnectionId,
        to: { ...suggestion.connection.to, moduleId: newModuleId },
      };

      setModules((prev) => [...prev, finalModule]);
      setConnections((prev) => [...prev, finalConnection]);
      setSuggestion(null);
      setIsDirty(true);
    }
  }, [suggestion, setModules, setConnections]);

  const clearSuggestion = useCallback(() => {
    setSuggestion(null);
  }, []);

  const handleResizeMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      const startWidth = rightPanelWidth;
      const startX = e.clientX;
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";

      // fix: Changed MouseEvent to globalThis.MouseEvent to match the native event type expected by window.addEventListener.
      const handleMouseMove = (moveEvent: globalThis.MouseEvent) => {
        const dx = moveEvent.clientX - startX;
        const newWidth = startWidth - dx;

        const minWidth = 320; // Corresponds to w-80
        const maxWidth = 800; // An arbitrary upper limit

        if (newWidth >= minWidth && newWidth <= maxWidth) {
          setRightPanelWidth(newWidth);
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
    [rightPanelWidth]
  );

  const handleToggleRightPanel = () => {
    setIsRightPanelVisible((prev) => !prev);
  };

  const handleModuleDoubleClick = useCallback((id: string) => {
    setSelectedModuleIds((prev) => {
      if (prev.length === 1 && prev[0] === id) {
        return prev;
      }
      return [id];
    });
    setIsRightPanelVisible(true);
    setActivePropertiesTab("properties");
  }, []);

  const handleFitToView = useCallback(() => {
    if (!canvasContainerRef.current) return;
    const canvasRect = canvasContainerRef.current.getBoundingClientRect();

    if (modules.length === 0) {
      setPan({ x: 0, y: 0 });
      setScale(1);
      return;
    }

    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;

    const moduleWidth = 256; // w-64 (Updated from 192)
    const moduleHeight = 120; // approximate height

    modules.forEach((module) => {
      minX = Math.min(minX, module.position.x);
      minY = Math.min(minY, module.position.y);
      maxX = Math.max(maxX, module.position.x + moduleWidth);
      maxY = Math.max(maxY, module.position.y + moduleHeight);
    });

    const contentWidth = maxX - minX;
    const contentHeight = maxY - minY;

    const padding = 50;
    const scaleX = (canvasRect.width - padding * 2) / contentWidth;
    const scaleY = (canvasRect.height - padding * 2) / contentHeight;
    const newScale = Math.min(scaleX, scaleY, 1);

    const newPanX =
      (canvasRect.width - contentWidth * newScale) / 2 - minX * newScale;
    const newPanY =
      (canvasRect.height - contentHeight * newScale) / 2 - minY * newScale;

    setScale(newScale);
    setPan({ x: newPanX, y: newPanY });
  }, [modules]);

  const handleRotateModules = useCallback(() => {
    if (modules.length === 0) return;

    const moduleWidth = 256; // w-64
    const moduleHeight = 120; // approximate height

    // Calculate bounding box
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;

    modules.forEach((module) => {
      minX = Math.min(minX, module.position.x);
      minY = Math.min(minY, module.position.y);
      maxX = Math.max(maxX, module.position.x + moduleWidth);
      maxY = Math.max(maxY, module.position.y + moduleHeight);
    });

    const contentWidth = maxX - minX;
    const contentHeight = maxY - minY;

    // Calculate center point
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;

    // Determine rotation direction based on aspect ratio
    // 세로가 긴 경우: 90도 반시계방향 (counter-clockwise) -> 가로 모드로 변환
    // 가로가 긴 경우: 90도 시계방향 (clockwise) -> 세로 모드로 변환
    const isVertical = contentHeight > contentWidth;
    const rotationAngle = isVertical ? -90 : 90; // -90 for counter-clockwise, 90 for clockwise
    const isConvertingToHorizontal = isVertical; // 세로 -> 가로 변환
    const spacingMultiplier = isConvertingToHorizontal ? 2 : 0.5; // 가로 모드: 2배 넓게, 세로 모드: 2배 작게

    // Convert angle to radians
    const angleRad = (rotationAngle * Math.PI) / 180;
    const cos = Math.cos(angleRad);
    const sin = Math.sin(angleRad);

    // Rotate each module around the center point
    const rotatedModules = modules.map((module) => {
      // Translate to origin (center)
      const dx = module.position.x + moduleWidth / 2 - centerX;
      const dy = module.position.y + moduleHeight / 2 - centerY;

      // Rotate
      const rotatedDx = dx * cos - dy * sin;
      const rotatedDy = dx * sin + dy * cos;

      // Translate back
      const newX = centerX + rotatedDx - moduleWidth / 2;
      const newY = centerY + rotatedDy - moduleHeight / 2;

      return {
        ...module,
        position: { x: newX, y: newY },
      };
    });

    // Calculate new bounding box after rotation
    let newMinX = Infinity;
    let newMinY = Infinity;
    let newMaxX = -Infinity;
    let newMaxY = -Infinity;

    rotatedModules.forEach((module) => {
      newMinX = Math.min(newMinX, module.position.x);
      newMinY = Math.min(newMinY, module.position.y);
      newMaxX = Math.max(newMaxX, module.position.x + moduleWidth);
      newMaxY = Math.max(newMaxY, module.position.y + moduleHeight);
    });

    const newCenterX = (newMinX + newMaxX) / 2;
    const newCenterY = (newMinY + newMaxY) / 2;

    // Adjust spacing: 가로 모드로 변환 시 간격을 넓히고, 세로 모드로 변환 시 간격을 좁힘
    const adjustedModules = rotatedModules.map((module) => {
      // 모듈 중심점에서 새로운 중심점까지의 거리
      const moduleCenterX = module.position.x + moduleWidth / 2;
      const moduleCenterY = module.position.y + moduleHeight / 2;

      const offsetX = moduleCenterX - newCenterX;
      const offsetY = moduleCenterY - newCenterY;

      // 간격 조정 (중심점에서 멀어지거나 가까워지도록)
      const adjustedOffsetX = offsetX * spacingMultiplier;
      const adjustedOffsetY = offsetY * spacingMultiplier;

      return {
        ...module,
        position: {
          x: newCenterX + adjustedOffsetX - moduleWidth / 2,
          y: newCenterY + adjustedOffsetY - moduleHeight / 2,
        },
      };
    });

    setModules(adjustedModules);
    setIsDirty(true);

    // Fit to view after rotation
    setTimeout(() => handleFitToView(), 100);
  }, [modules, setModules, handleFitToView]);

  const handleRearrangeModules = useCallback(() => {
    if (modules.length === 0) return;

    // Model definition types (auxiliary modules that should be placed to the left)
    const MODEL_DEFINITION_TYPES: ModuleType[] = [
      ModuleType.LinearRegression,
      ModuleType.LogisticRegression,
      ModuleType.PoissonRegression,
      ModuleType.NegativeBinomialRegression,
      ModuleType.DecisionTree,
      ModuleType.RandomForest,
      ModuleType.SVM,
      ModuleType.LinearDiscriminantAnalysis,
      ModuleType.NaiveBayes,
      ModuleType.KNN,
      ModuleType.KMeans,
      ModuleType.HierarchicalClustering,
      ModuleType.DBSCAN,
      ModuleType.PrincipalComponentAnalysis,
      // Traditional Analysis - Statsmodels Models
      ModuleType.OLSModel,
      ModuleType.LogisticModel,
      ModuleType.PoissonModel,
      ModuleType.QuasiPoissonModel,
      ModuleType.NegativeBinomialModel,
      ModuleType.StatModels,
    ];

    // 1. Build graph representations
    const adj: Record<string, string[]> = {};
    const revAdj: Record<string, string[]> = {};
    const inDegree: Record<string, number> = {};
    modules.forEach((m) => {
      adj[m.id] = [];
      revAdj[m.id] = [];
      inDegree[m.id] = 0;
    });

    connections.forEach((conn) => {
      if (adj[conn.from.moduleId] && revAdj[conn.to.moduleId]) {
        adj[conn.from.moduleId].push(conn.to.moduleId);
        revAdj[conn.to.moduleId].push(conn.from.moduleId);
        inDegree[conn.to.moduleId]++;
      }
    });

    // 2. Topological sort to get execution order (top to bottom)
    const queue = modules.filter((m) => inDegree[m.id] === 0).map((m) => m.id);
    const sortedModuleIds: string[] = [];
    const tempInDegree = { ...inDegree };
    while (queue.length > 0) {
      const u = queue.shift()!;
      sortedModuleIds.push(u);
      (adj[u] || []).forEach((v) => {
        tempInDegree[v]--;
        if (tempInDegree[v] === 0) {
          queue.push(v);
        }
      });
    }

    // Handle cycles/unreachable nodes by appending them
    if (sortedModuleIds.length < modules.length) {
      addLog(
        "WARN",
        "Cycle detected or modules are unreachable. Appending to layout."
      );
      modules.forEach((m) => {
        if (!sortedModuleIds.includes(m.id)) {
          sortedModuleIds.push(m.id);
        }
      });
    }

    // 3. Separate auxiliary modules from regular modules
    const regularModules: string[] = [];
    const auxiliaryModules: Record<string, string> = {}; // moduleId -> parentModuleId

    sortedModuleIds.forEach((moduleId) => {
      const module = modules.find((m) => m.id === moduleId);
      if (!module) return;

      if (MODEL_DEFINITION_TYPES.includes(module.type)) {
        // Find the module it connects to (usually TrainModel)
        const connection = connections.find(
          (c) => c.from.moduleId === moduleId
        );
        if (connection) {
          auxiliaryModules[moduleId] = connection.to.moduleId;
        } else {
          // If no connection found, treat as regular module
          regularModules.push(moduleId);
        }
      } else {
        regularModules.push(moduleId);
      }
    });

    // 4. Calculate positions - 위에서 아래로 배치
    const moduleWidth = 256;
    const vSpacing = 150; // 상하 간격
    const auxiliaryOffset = 10; // 보조 모듈 왼쪽 오프셋
    const initialX = 50;
    const initialY = 50;

    const newModules = [...modules];
    const modulePositions: Record<string, { x: number; y: number }> = {};
    let currentY = initialY;

    // Place regular modules vertically from top to bottom
    regularModules.forEach((moduleId) => {
      modulePositions[moduleId] = {
        x: initialX,
        y: currentY,
      };
      currentY += vSpacing;
    });

    // 5. Place auxiliary modules to the left of their parent modules
    Object.entries(auxiliaryModules).forEach(
      ([auxModuleId, parentModuleId]) => {
        const parentPos = modulePositions[parentModuleId];
        if (parentPos) {
          modulePositions[auxModuleId] = {
            x: parentPos.x - moduleWidth - auxiliaryOffset,
            y: parentPos.y,
          };
        } else {
          // If parent not found, place at initial position
          modulePositions[auxModuleId] = {
            x: initialX - moduleWidth - auxiliaryOffset,
            y: initialY,
          };
        }
      }
    );

    // 6. Update module positions
    newModules.forEach((module, index) => {
      const pos = modulePositions[module.id];
      if (pos) {
        newModules[index] = {
          ...module,
          position: pos,
        };
      }
    });

    // 7. Update state
    setModules(newModules);
    setIsDirty(true);
    setTimeout(() => handleFitToView(), 0);
  }, [modules, connections, setModules, handleFitToView, addLog]);

  const handleGeneratePipeline = async (
    prompt: string,
    type: "goal" | "data",
    file?: { content: string; name: string }
  ) => {
    setIsAiGenerating(true);
    addLog("INFO", "AI pipeline generation started...");
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });

      const moduleDescriptions: Record<string, string> = {
        LoadData: "Loads a dataset from a user-provided CSV file.",
        Statistics:
          "Calculates descriptive statistics and correlation matrix for a dataset.",
        SelectData: "Selects or removes specific columns from a dataset.",
        HandleMissingValues:
          "Handles missing (null) values in a dataset by removing rows or filling values.",
        EncodeCategorical:
          "Converts categorical (string) columns into numerical format for modeling.",
        NormalizeData:
          "Scales numerical features to a standard range (e.g., 0-1).",
        TransitionData:
          "Applies mathematical transformations (e.g., log, sqrt) to numeric columns.",
        SplitData: "Splits a dataset into training and testing sets.",
        LinearRegression: "Defines a scikit-learn Linear Regression model.",
        LogisticRegression:
          "Defines a Logistic Regression model for classification.",
        DecisionTreeClassifier:
          "Defines a Decision Tree model for classification.",
        StatModels:
          "Defines a statistical model from the statsmodels library (e.g., OLS, Logit).",
        TrainModel: "Trains a model algorithm using a training dataset.",
        ResultModel:
          "Fits a statistical model (from StatModels) to a dataset and shows the results summary.",
        ScoreModel:
          "Applies a trained ML model to a dataset to generate predictions.",
        PredictModel:
          "Applies a fitted statistical model to a dataset to generate predictions.",
        EvaluateModel:
          "Evaluates the performance of a trained model on a test dataset.",
        FitLossDistribution:
          "Fits a statistical distribution (e.g., Pareto) to loss data.",
        GenerateExposureCurve:
          "Generates an exposure curve from a fitted distribution.",
        PriceXoLLayer:
          "Calculates the premium for an Excess of Loss (XoL) layer using an exposure curve.",
        XolLoading:
          "Loads claims data specifically for experience-based XoL pricing.",
        ApplyThreshold:
          "Filters out claims that are below a specified monetary threshold.",
        DefineXolContract:
          "Defines the parameters for an XoL reinsurance contract.",
        CalculateCededLoss:
          "Calculates the ceded loss for each claim based on contract terms.",
        PriceXolContract:
          "Prices an XoL contract using the burning cost method based on historical data.",
      };

      const detailedModulesString = DEFAULT_MODULES.map((defaultModule) => {
        const moduleInfo = TOOLBOX_MODULES.find(
          (m) => m.type === defaultModule.type
        );
        const description =
          moduleDescriptions[defaultModule.type] || "A standard module.";
        return `
- type: ${defaultModule.type}
  name: ${moduleInfo?.name}
  description: ${description}
  inputs: ${JSON.stringify(defaultModule.inputs)}
  outputs: ${JSON.stringify(defaultModule.outputs)}
`;
      }).join("");

      const fullPrompt = `
You are an expert ML pipeline architect for a tool called "ML Pipeline Canvas Pro".
Your task is to generate a logical machine learning pipeline based on the user's goal and available data, and provide a clear plan for your design.
The pipeline MUST be a valid JSON object containing a 'plan', a list of 'modules', and the 'connections' between them.

### Available Modules & Their Ports
Here are the only modules you can use. You MUST NOT invent new module types. Each module has specific input and output ports. Use these exact port names and types for connections.
${detailedModulesString}

### User's Goal & Data
${prompt}

### Instructions
1.  **Analyze Goal & Data**: Understand the user's objective. If column names are provided, use them to make informed decisions about module parameters. Most pipelines should involve training a model ('TrainModel' or 'ResultModel').
2.  **Select Modules**: Choose a logical sequence of modules from the list above to accomplish the goal. Start with 'LoadData'. If the data has categorical features (strings), you MUST use the 'EncodeCategorical' module. If there are missing values, use 'HandleMissingValues'.
3.  **Configure Modules**:
    *   Provide a short, descriptive \`name\` for each module instance.
    *   If column names are available, set the \`parameters\` for modules like 'TrainModel' or 'ResultModel' by inferring the 'feature_columns' and 'label_column'.
4.  **Define Connections**:
    *   Create connections between the modules using their 0-based index in the 'modules' array.
    *   **CRITICAL**: Connect output ports to input ports. The \`type\` of the ports must match (e.g., 'data' to 'data', 'model' to 'model'). Use the exact port names from the module list.
5.  **Create an Execution Plan**: In the \`plan\` field, provide a step-by-step explanation of your reasoning in Korean using Markdown. Explain why you chose each module.
    *   **IMPORTANT**: If you cannot fully satisfy the user's request with the available modules (e.g., user asks for "time series analysis" but there's no such module), build the best possible pipeline with the existing tools. Then, in the plan, create a "## 추가 제안" section and clearly state the limitations and what the user should do next to fully achieve their goal. DO NOT invent modules.
6.  **Final Output**: Respond ONLY with a single, valid JSON object that conforms to the schema below. Do not include any explanatory text, markdown formatting, or anything else outside the JSON structure.

### JSON Output Schema
The JSON object must contain 'plan', 'modules', and 'connections'.
- \`plan\`: A string containing the Markdown explanation of your pipeline design.
- \`modules\`: An array of module objects.
- \`connections\`: An array of connection objects.
`;

      const response = await ai.models.generateContent({
        model: "gemini-2.5-pro",
        contents: fullPrompt,
        config: {
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              plan: { type: Type.STRING },
              modules: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    type: { type: Type.STRING },
                    name: { type: Type.STRING },
                  },
                  required: ["type", "name"],
                },
              },
              connections: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    fromModuleIndex: { type: Type.INTEGER },
                    fromPort: { type: Type.STRING },
                    toModuleIndex: { type: Type.INTEGER },
                    toPort: { type: Type.STRING },
                  },
                  required: [
                    "fromModuleIndex",
                    "fromPort",
                    "toModuleIndex",
                    "toPort",
                  ],
                },
              },
            },
            required: ["plan", "modules", "connections"],
          },
        },
      });

      const responseText = response.text.trim();
      const pipeline = JSON.parse(responseText);

      if (!pipeline.modules || !pipeline.connections || !pipeline.plan) {
        throw new Error(
          "AI response is missing 'plan', 'modules', or 'connections'."
        );
      }

      setAiPlan(pipeline.plan);

      // --- Render the generated pipeline ---
      const previousState = {
        modules: [...modules],
        connections: [...connections],
      };

      const newModules: CanvasModule[] = [];
      pipeline.modules.forEach((mod: any, index: number) => {
        const defaultData = DEFAULT_MODULES.find((m) => m.type === mod.type);
        if (!defaultData) {
          addLog(
            "WARN",
            `AI generated an unknown module type: '${mod.type}'. Skipping.`
          );
          return;
        }
        const newModule: CanvasModule = {
          id: `${mod.type}-${Date.now()}-${index}`,
          name: mod.name,
          type: mod.type as ModuleType,
          position: { x: 250, y: 100 + index * 150 }, // Simple vertical layout
          status: ModuleStatus.Pending,
          parameters: { ...defaultData.parameters, ...mod.parameters },
          inputs: [...defaultData.inputs],
          outputs: [...defaultData.outputs],
        };

        if (
          file &&
          (newModule.type === ModuleType.LoadData ||
            newModule.type === ModuleType.XolLoading)
        ) {
          newModule.parameters.fileContent = file.content;
          newModule.parameters.source = file.name;
        }

        newModules.push(newModule);
      });

      const newConnections: Connection[] = [];
      pipeline.connections.forEach((conn: any, index: number) => {
        const fromModule = newModules[conn.fromModuleIndex];
        const toModule = newModules[conn.toModuleIndex];
        if (fromModule && toModule) {
          newConnections.push({
            id: `conn-ai-${Date.now()}-${index}`,
            from: { moduleId: fromModule.id, portName: conn.fromPort },
            to: { moduleId: toModule.id, portName: conn.toPort },
          });
        }
      });

      // Use a single state update for undo/redo
      setModules(newModules);
      setConnections(newConnections);
      setIsDirty(false);

      setTimeout(() => handleFitToView(), 0);

      addLog("SUCCESS", "AI successfully generated a new pipeline.");
    } catch (error: any) {
      console.error("AI pipeline generation failed:", error);
      addLog("ERROR", `AI generation failed: ${error.message}`);
    } finally {
      setIsAiGenerating(false);
    }
  };

  const handleGeneratePipelineFromGoal = (goal: string) => {
    handleGeneratePipeline(`Goal: ${goal}`, "goal");
  };

  const handleGeneratePipelineFromData = (
    goal: string,
    fileContent: string,
    fileName: string
  ) => {
    const lines = fileContent.trim().split("\n");
    if (lines.length === 0) {
      addLog("ERROR", "Uploaded file is empty.");
      return;
    }
    const header = lines[0];
    const dataPrompt = `
Goal: ${goal}
---
Dataset Columns:
${header}
`;
    handleGeneratePipeline(dataPrompt, "data", {
      content: fileContent,
      name: fileName,
    });
  };

  const handleSavePipeline = useCallback(async () => {
    try {
      const pipelineState = { modules, connections, projectName };
      
      await savePipeline(pipelineState, {
        extension: ".mla",
        description: "ML Pipeline File",
        onSuccess: (fileName) => {
          addLog("SUCCESS", `Pipeline saved to '${fileName}'.`);
          setIsDirty(false);
          setSaveButtonText("Saved!");
          setTimeout(() => setSaveButtonText("Save"), 2000);
        },
        onError: (error) => {
          console.error("Failed to save pipeline:", error);
          addLog("ERROR", `Failed to save pipeline: ${error.message}`);
        },
      });
    } catch (error: any) {
      if (error.name !== "AbortError") {
        console.error("Failed to save pipeline:", error);
        addLog("ERROR", `Failed to save pipeline: ${error.message}`);
      }
    }
  }, [modules, connections, projectName, addLog]);

  const handleLoadPipeline = useCallback(async () => {
    const savedState = await loadPipeline({
      extension: ".mla",
      onError: (error) => {
        addLog("ERROR", error.message);
      },
    });

    if (savedState) {
      if (savedState.modules && savedState.connections) {
        resetModules(savedState.modules);
        _setConnections(savedState.connections);
        if (savedState.projectName) {
          setProjectName(savedState.projectName);
        }
        setSelectedModuleIds([]);
        setIsDirty(false);
        addLog("SUCCESS", "Pipeline loaded successfully.");
      } else {
        addLog("WARN", "Invalid pipeline file format.");
      }
    }
  }, [resetModules, addLog]);

  const handleLoadSample = useCallback(
    async (
      sampleName: string,
      source: "samples" | "mywork" | "folder" = "samples",
      filename?: string
    ) => {
      console.log(
        "handleLoadSample called with:",
        sampleName,
        "from:",
        source,
        "filename:",
        filename
      );
      try {
        let sampleModel: any = null;

        if (source === "folder" && filename) {
          // Samples 폴더에서 파일 로드 (공통 유틸리티 사용)
          try {
            sampleModel = await loadSampleFromFolder(
              filename,
              "http://localhost:3002/api/samples"
            );
            if (!sampleModel) {
              addLog("ERROR", `Failed to load sample file: ${filename}`);
              return;
            }
          } catch (error: any) {
            console.error("Error loading folder sample:", error);
            addLog(
              "ERROR",
              `Error loading sample file: ${error.message || error}`
            );
            return;
          }
        } else if (source === "mywork") {
          // My Work에서 찾기
          const myWorkModelsStr = localStorage.getItem("myWorkModels");
          if (myWorkModelsStr) {
            try {
              const myWorkModels = JSON.parse(myWorkModelsStr);
              if (Array.isArray(myWorkModels)) {
                sampleModel = myWorkModels.find(
                  (m: any) => m.name === sampleName
                );
              }
            } catch (error) {
              console.error("Failed to parse my work models:", error);
            }
          }
        } else {
          // Samples에서 찾기
          // 먼저 SAVED_SAMPLES에서 찾기
          const savedSamples = getSavedSamples();
          if (savedSamples && savedSamples.length > 0) {
            sampleModel = savedSamples.find((m: any) => m.name === sampleName);
          }

          // SAVED_SAMPLES에 없으면 SAMPLE_MODELS에서 찾기
          if (!sampleModel) {
            sampleModel = SAMPLE_MODELS.find((m: any) => m.name === sampleName);
          }
        }

        console.log("Found sample model:", sampleModel);
        if (!sampleModel) {
          console.error("Sample model not found:", sampleName);
          addLog("ERROR", `Sample model "${sampleName}" not found.`);
          return;
        }

        // Convert sample model format to app format
        const newModules: CanvasModule[] = sampleModel.modules.map(
          (m: any, index: number) => {
            const moduleId = `module-${Date.now()}-${index}`;
            const defaultModule = DEFAULT_MODULES.find(
              (dm) => dm.type === m.type
            );
            if (!defaultModule) {
              addLog(
                "ERROR",
                `Module type "${m.type}" not found in DEFAULT_MODULES.`
              );
              throw new Error(`Module type "${m.type}" not found`);
            }
            const moduleInfo = TOOLBOX_MODULES.find((tm) => tm.type === m.type);
            const defaultName = moduleInfo ? moduleInfo.name : m.type;
            return {
              ...defaultModule,
              id: moduleId,
              name: m.name || defaultName,
              position: m.position,
              status: ModuleStatus.Pending,
            };
          }
        );

        const newConnections: Connection[] = sampleModel.connections.map(
          (conn: any, index: number) => {
            const fromModule = newModules[conn.fromModuleIndex];
            const toModule = newModules[conn.toModuleIndex];
            if (!fromModule || !toModule) {
              addLog("ERROR", `Invalid connection at index ${index}.`);
              throw new Error(`Invalid connection at index ${index}`);
            }
            return {
              id: `connection-${Date.now()}-${index}`,
              from: { moduleId: fromModule.id, portName: conn.fromPort },
              to: { moduleId: toModule.id, portName: conn.toPort },
            };
          }
        );

        resetModules(newModules);
        _setConnections(newConnections);
        setSelectedModuleIds([]);
        setIsDirty(false);
        setProjectName(sampleName);
        setIsSampleMenuOpen(false);
        addLog("SUCCESS", `Sample model "${sampleName}" loaded successfully.`);
        setTimeout(() => handleFitToView(), 100);
      } catch (error: any) {
        console.error("Error loading sample:", error);
        addLog(
          "ERROR",
          `Failed to load sample: ${error.message || "Unknown error"}`
        );
        setIsSampleMenuOpen(false);
      }
    },
    [resetModules, addLog, handleFitToView]
  );

  // Samples 폴더의 파일 목록 가져오기
  const loadFolderSamplesLocal = useCallback(async () => {
    setIsLoadingSamples(true);
    try {
      const samples = await loadFolderSamples("http://localhost:3002/api/samples/list");
      
      if (Array.isArray(samples) && samples.length > 0) {
        console.log(
          `Loaded ${samples.length} samples from server:`,
          samples.map((s: any) => s.name || s.filename)
        );
        setFolderSamples(samples);
      } else {
        console.log("No samples found or empty array");
        setFolderSamples([]);
      }
    } catch (error: any) {
      console.error("Error loading folder samples:", error);
      setFolderSamples([]);
    } finally {
      setIsLoadingSamples(false);
    }
  }, []);

  // Samples 메뉴가 열릴 때마다 폴더 샘플 목록 새로고침
  useEffect(() => {
    if (isSampleMenuOpen) {
      console.log("Samples menu opened, loading folder samples...");
      // 약간의 지연을 두어 메뉴가 완전히 열린 후 로드
      const timer = setTimeout(() => {
        loadFolderSamplesLocal();
      }, 100);
      return () => clearTimeout(timer);
    }
    // 메뉴가 닫혀도 상태는 유지 (다음에 열 때 빠르게 표시)
  }, [isSampleMenuOpen, loadFolderSamplesLocal]);

  // 디버깅: folderSamples 상태 변경 추적
  useEffect(() => {
    if (folderSamples.length > 0) {
      console.log(
        `folderSamples updated: ${folderSamples.length} samples`,
        folderSamples.map((s) => s.name || s.filename)
      );
    } else if (isSampleMenuOpen && !isLoadingSamples) {
      console.log("folderSamples is empty but menu is open and not loading");
    }
  }, [folderSamples, isSampleMenuOpen, isLoadingSamples]);

  // My Work 모델 목록 로드
  useEffect(() => {
    const myWorkModelsStr = localStorage.getItem("myWorkModels");
    if (myWorkModelsStr) {
      try {
        const models = JSON.parse(myWorkModelsStr);
        setMyWorkModels(Array.isArray(models) ? models : []);
      } catch (error) {
        console.error("Failed to load my work models:", error);
      }
    }
  }, []);

  // 초기 화면 로드
  useEffect(() => {
    const initialModelStr = localStorage.getItem("initialModel");
    if (initialModelStr && modules.length === 0) {
      try {
        const initialModel = JSON.parse(initialModelStr);
        handleLoadSample(initialModel.name);
      } catch (error) {
        console.error("Failed to load initial model:", error);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // 초기 마운트 시에만 실행

  // 모듈 배열에 따라 자동으로 Fit to View 실행
  useEffect(() => {
    if (modules.length > 0) {
      // 약간의 지연을 두어 DOM이 완전히 렌더링된 후 실행
      const timer = setTimeout(() => {
        handleFitToView();
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [modules, handleFitToView]);

  // Close sample menu and my work menu when clicking outside
  useEffect(() => {
    if (!isSampleMenuOpen && !isMyWorkMenuOpen) return;

    const handleClickOutside = (event: globalThis.MouseEvent) => {
      const target = event.target as Node;
      if (sampleMenuRef.current && !sampleMenuRef.current.contains(target)) {
        setIsSampleMenuOpen(false);
      }
      if (myWorkMenuRef.current && !myWorkMenuRef.current.contains(target)) {
        setIsMyWorkMenuOpen(false);
      }
    };

    // Longer delay to ensure button click completes first
    const timeoutId = setTimeout(() => {
      document.addEventListener("click", handleClickOutside);
    }, 200);

    return () => {
      clearTimeout(timeoutId);
      document.removeEventListener("click", handleClickOutside);
    };
  }, [isSampleMenuOpen, isMyWorkMenuOpen]);

  // fix: Added missing handleSetFolder function to resolve "Cannot find name" error.
  const handleSetFolder = useCallback(async () => {
    try {
      if (!("showDirectoryPicker" in window)) {
        addLog(
          "WARN",
          "현재 브라우저에서는 폴더 설정 기능을 지원하지 않습니다."
        );
        return;
      }
      const handle = await (window as any).showDirectoryPicker();
      folderHandleRef.current = handle;
      addLog("SUCCESS", `저장 폴더가 '${handle.name}'(으)로 설정되었습니다.`);
    } catch (error: any) {
      if (error.name !== "AbortError") {
        console.error("Failed to set save folder:", error);
        addLog(
          "ERROR",
          `폴더를 설정하지 못했습니다: ${error.message}. 브라우저 권한 설정을 확인해 주세요.`
        );
      }
    }
  }, [addLog]);

  const createModule = useCallback(
    (type: ModuleType, position: { x: number; y: number }) => {
      clearSuggestion();

      // Handle shape types (TextBox, GroupBox)
      if (type === ModuleType.TextBox || type === ModuleType.GroupBox) {
        const shapeName =
          type === ModuleType.TextBox ? "텍스트 상자" : "그룹 상자";
        const count = modules.filter((m) => m.type === type).length + 1;

        let shapeData: CanvasModule["shapeData"];

        if (type === ModuleType.TextBox) {
          shapeData = { text: "", width: 200, height: 100, fontSize: 14 };
        } else {
          // GroupBox: Calculate bounds for selected modules
          const selectedModules = modules.filter(
            (m) =>
              selectedModuleIds.includes(m.id) &&
              m.type !== ModuleType.TextBox &&
              m.type !== ModuleType.GroupBox
          );

          if (selectedModules.length > 0) {
            const moduleWidth = 256;
            const moduleHeight = 120;

            let minX = Infinity,
              minY = Infinity,
              maxX = -Infinity,
              maxY = -Infinity;

            selectedModules.forEach((module) => {
              const x = module.position.x;
              const y = module.position.y;
              minX = Math.min(minX, x);
              minY = Math.min(minY, y);
              maxX = Math.max(maxX, x + moduleWidth);
              maxY = Math.max(maxY, y + moduleHeight);
            });

            // 모듈만 들어갈 수 있도록 여백 최소화
            const padding = 10;
            const bounds = {
              x: minX - padding,
              y: minY - padding,
              width: maxX - minX + padding * 2,
              height: maxY - minY + padding * 2,
            };

            shapeData = {
              moduleIds: selectedModules.map((m) => m.id),
              bounds,
            };

            // Set group box position to bounds position
            position = { x: bounds.x, y: bounds.y };
          } else {
            // Default bounds if no modules selected
            const defaultWidth = 300;
            const defaultHeight = 200;
            shapeData = {
              moduleIds: [],
              bounds: {
                x: position.x,
                y: position.y,
                width: defaultWidth,
                height: defaultHeight,
              },
            };
          }
        }

        const newModule: CanvasModule = {
          id: `${type}-${Date.now()}`,
          name: `${shapeName} ${count}`,
          type,
          position,
          status: ModuleStatus.Pending,
          parameters: {},
          inputs: [],
          outputs: [],
          shapeData,
        };
        setModules((prev) => [...prev, newModule]);
        setSelectedModuleIds([newModule.id]);
        setIsDirty(true);
        return;
      }

      const defaultData = DEFAULT_MODULES.find((m) => m.type === type);
      if (!defaultData) {
        console.error(`No default data found for module type: ${type}`);
        addLog(
          "ERROR",
          `Module type '${type}' is not supported. Please check if the module is properly defined.`
        );
        return;
      }

      const moduleInfo = TOOLBOX_MODULES.find((m) => m.type === type);
      const baseName = moduleInfo ? moduleInfo.name : type;

      const count = modules.filter((m) => m.type === type).length + 1;
      const newModule: CanvasModule = {
        id: `${type}-${Date.now()}`,
        name: `${baseName} ${count}`,
        type,
        position,
        status: ModuleStatus.Pending,
        parameters: { ...defaultData.parameters },
        inputs: [...defaultData.inputs],
        outputs: [...defaultData.outputs],
      };

      setModules((prev) => [...prev, newModule]);
      setSelectedModuleIds([newModule.id]);
      setIsDirty(true);
    },
    [modules, setModules, setSelectedModuleIds, clearSuggestion, addLog]
  );

  const handleModuleToolboxDoubleClick = useCallback(
    (type: ModuleType) => {
      // GroupBox는 선택된 모듈들을 기준으로 생성
      if (type === ModuleType.GroupBox) {
        const selectedModules = modules.filter(
          (m) =>
            selectedModuleIds.includes(m.id) &&
            m.type !== ModuleType.TextBox &&
            m.type !== ModuleType.GroupBox
        );

        if (selectedModules.length === 0) {
          addLog(
            "WARN",
            "그룹 상자를 만들려면 하나 이상의 모듈을 선택해야 합니다."
          );
          return;
        }

        const moduleWidth = 256;
        const moduleHeight = 120;
        const padding = 10; // 여백을 줄임

        let minX = Infinity,
          minY = Infinity,
          maxX = -Infinity,
          maxY = -Infinity;

        selectedModules.forEach((module) => {
          const x = module.position.x;
          const y = module.position.y;
          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          maxX = Math.max(maxX, x + moduleWidth);
          maxY = Math.max(maxY, y + moduleHeight);
        });

        const bounds = {
          x: minX - padding,
          y: minY - padding,
          width: maxX - minX + padding * 2,
          height: maxY - minY + padding * 2,
        };

        const shapeName = "그룹 상자";
        const count = modules.filter((m) => m.type === type).length + 1;
        const newModule: CanvasModule = {
          id: `${type}-${Date.now()}`,
          name: `${shapeName} ${count}`,
          type,
          position: { x: bounds.x, y: bounds.y }, // position과 bounds.x, bounds.y를 일치시킴
          status: ModuleStatus.Pending,
          parameters: {},
          inputs: [],
          outputs: [],
          shapeData: {
            moduleIds: selectedModules.map((m) => m.id),
            bounds,
          },
        };

        setModules((prev) => [...prev, newModule]);
        setSelectedModuleIds([newModule.id]);
        setIsDirty(true);
        addLog(
          "SUCCESS",
          `${selectedModules.length}개의 모듈을 그룹으로 묶었습니다.`
        );
        return;
      }

      // 다른 모듈들은 기존대로 중앙에 생성
      if (canvasContainerRef.current) {
        const canvasRect = canvasContainerRef.current.getBoundingClientRect();
        // Position in the middle, accounting for current pan and scale
        const position = {
          x: (canvasRect.width / 2 - 128 - pan.x) / scale, // 128 is half module width (256/2)
          y: (canvasRect.height / 2 - 60 - pan.y) / scale, // 60 is half module height
        };
        createModule(type, position);
      }
    },
    [
      createModule,
      scale,
      pan,
      modules,
      selectedModuleIds,
      setModules,
      setSelectedModuleIds,
      addLog,
    ]
  );

  const handleFontSizeChange = useCallback(
    (increase: boolean) => {
      const selectedTextBoxes = modules.filter(
        (m) => selectedModuleIds.includes(m.id) && m.type === ModuleType.TextBox
      );

      if (selectedTextBoxes.length === 0) {
        addLog("WARN", "글자 크기를 조절하려면 텍스트 상자를 선택하세요.");
        return;
      }

      setModules(
        (prev) =>
          prev.map((m) => {
            if (selectedTextBoxes.some((tb) => tb.id === m.id)) {
              const currentFontSize = m.shapeData?.fontSize || 14;
              const newFontSize = increase
                ? Math.min(currentFontSize + 2, 32) // 최대 32px
                : Math.max(currentFontSize - 2, 8); // 최소 8px

              return {
                ...m,
                shapeData: {
                  ...m.shapeData,
                  fontSize: newFontSize,
                },
              };
            }
            return m;
          }),
        true
      );
      setIsDirty(true);
    },
    [modules, selectedModuleIds, setModules, addLog]
  );

  const updateModulePositions = useCallback(
    (updates: { id: string; position: { x: number; y: number } }[]) => {
      const updatesMap = new Map(updates.map((u) => [u.id, u.position]));
      setModules((prev) => {
        // First pass: Calculate group box movements and store dx/dy
        const groupMovements = new Map<string, { dx: number; dy: number }>();

        const updatedModules = prev.map((m) => {
          const newPos = updatesMap.get(m.id);
          if (!newPos) return m;

          // If this is a GroupBox being moved, calculate movement delta
          if (m.type === ModuleType.GroupBox && m.shapeData?.moduleIds) {
            const dx = newPos.x - m.position.x;
            const dy = newPos.y - m.position.y;

            // Store movement delta for modules in this group
            if (dx !== 0 || dy !== 0) {
              groupMovements.set(m.id, { dx, dy });
            }

            // Update bounds to match new position
            const newBounds = m.shapeData.bounds
              ? {
                  ...m.shapeData.bounds,
                  x: newPos.x, // bounds.x를 새 position과 일치시킴
                  y: newPos.y, // bounds.y를 새 position과 일치시킴
                }
              : undefined;

            // Return updated group box
            return {
              ...m,
              position: newPos,
              shapeData: { ...m.shapeData, bounds: newBounds },
            };
          }

          return { ...m, position: newPos };
        });

        // Second pass: Update modules in groups if group box was moved
        return updatedModules.map((m) => {
          // Skip if this module is a group box or shape
          if (m.type === ModuleType.GroupBox || m.type === ModuleType.TextBox) {
            return m;
          }

          // Find the group box that contains this module
          const groupBox = updatedModules.find(
            (g) =>
              g.type === ModuleType.GroupBox &&
              g.shapeData?.moduleIds?.includes(m.id)
          );

          if (groupBox && groupMovements.has(groupBox.id)) {
            const movement = groupMovements.get(groupBox.id)!;
            return {
              ...m,
              position: {
                x: m.position.x + movement.dx,
                y: m.position.y + movement.dy,
              },
            };
          }

          return m;
        });
      }, true);
      setIsDirty(true);
    },
    [setModules]
  );

  // Helper function to find all downstream modules (modules that depend on the given module)
  const getDownstreamModules = useCallback(
    (
      moduleId: string,
      allModules: CanvasModule[],
      allConnections: Connection[]
    ): string[] => {
      const downstream: string[] = [];
      const visited = new Set<string>();

      const traverse = (currentId: string) => {
        if (visited.has(currentId)) return;
        visited.add(currentId);

        // Find all modules that receive output from this module
        const outgoingConnections = allConnections.filter(
          (c) => c.from.moduleId === currentId
        );
        outgoingConnections.forEach((conn) => {
          const targetId = conn.to.moduleId;
          if (!downstream.includes(targetId)) {
            downstream.push(targetId);
            traverse(targetId); // Recursively find downstream modules
          }
        });
      };

      traverse(moduleId);
      return downstream;
    },
    []
  );

  const updateModuleParameters = useCallback(
    (id: string, newParams: Record<string, any>) => {
      setModules((prev) => {
        const updated = prev.map((m) =>
          m.id === id
            ? { ...m, parameters: { ...m.parameters, ...newParams } }
            : m
        );

        // Find all downstream modules
        const downstreamIds = getDownstreamModules(id, updated, connections);

        // Find connected model definition module if this is TrainModel
        let modelDefinitionModuleId: string | null = null;
        const modifiedModule = updated.find((m) => m.id === id);
        if (modifiedModule && modifiedModule.type === ModuleType.TrainModel) {
          const modelInputConnection = connections.find(
            (c) => c.to.moduleId === id && c.to.portName === "model_in"
          );
          if (modelInputConnection) {
            modelDefinitionModuleId = modelInputConnection.from.moduleId;
          }
        }

        // Mark modified module and all downstream modules as Pending
        // Also mark connected model definition module as Pending if TrainModel is modified
        return updated.map((m) => {
          if (m.id === id || downstreamIds.includes(m.id)) {
            return {
              ...m,
              status: ModuleStatus.Pending,
              outputData: undefined,
            };
          }
          // Mark connected model definition module as Pending when TrainModel is modified
          if (
            modelDefinitionModuleId &&
            m.id === modelDefinitionModuleId &&
            MODEL_DEFINITION_TYPES.includes(m.type)
          ) {
            return {
              ...m,
              status: ModuleStatus.Pending,
              outputData: undefined,
            };
          }
          return m;
        });
      });
      setIsDirty(true);
    },
    [setModules, connections, getDownstreamModules]
  );

  const updateModule = useCallback(
    (id: string, updates: Partial<CanvasModule>) => {
      setModules(
        (prev) => prev.map((m) => (m.id === id ? { ...m, ...updates } : m)),
        true
      );
      setIsDirty(true);
    },
    [setModules]
  );

  const updateModuleName = useCallback(
    (id: string, newName: string) => {
      setModules((prev) => {
        const updated = prev.map((m) =>
          m.id === id ? { ...m, name: newName } : m
        );

        // Find all downstream modules
        const downstreamIds = getDownstreamModules(id, updated, connections);

        // Mark modified module and all downstream modules as Pending
        return updated.map((m) => {
          if (m.id === id || downstreamIds.includes(m.id)) {
            return {
              ...m,
              status: ModuleStatus.Pending,
              outputData: undefined,
            };
          }
          return m;
        });
      });
      setIsDirty(true);
    },
    [setModules, connections, getDownstreamModules]
  );

  const deleteModules = useCallback(
    (idsToDelete: string[]) => {
      setModules((prev) => prev.filter((m) => !idsToDelete.includes(m.id)));
      setConnections((prev) =>
        prev.filter(
          (c) =>
            !idsToDelete.includes(c.from.moduleId) &&
            !idsToDelete.includes(c.to.moduleId)
        )
      );
      setSelectedModuleIds((prev) =>
        prev.filter((id) => !idsToDelete.includes(id))
      );
      setIsDirty(true);
    },
    [setModules, setConnections, setSelectedModuleIds]
  );

  const handleViewDetails = (moduleId: string) => {
    const module = modules.find((m) => m.id === moduleId);
    if (module?.outputData) {
      if (module.outputData.type === "StatsModelsResultOutput") {
        setViewingStatsModelsResult(module);
      } else if (module.outputData.type === "DiversionCheckerOutput") {
        setViewingDiversionChecker(module);
      } else if (module.outputData.type === "EvaluateStatOutput") {
        setViewingEvaluateStat(module);
      } else if (module.outputData.type === "SplitDataOutput") {
        setViewingSplitDataForModule(module);
      } else if (module.outputData.type === "TrainedModelOutput") {
        setViewingTrainedModel(module);
      } else if (module.outputData.type === "XoLPriceOutput") {
        setViewingXoLPrice(module);
      } else if (module.outputData.type === "FinalXolPriceOutput") {
        setViewingFinalXolPrice(module);
      } else if (module.outputData.type === "EvaluationOutput") {
        setViewingEvaluation(module);
      } else {
        setViewingDataForModule(module);
      }
    }
  };

  const handleCloseModal = () => {
    setViewingDataForModule(null);
    setViewingSplitDataForModule(null);
    setViewingTrainedModel(null);
    setViewingStatsModelsResult(null);
    setViewingDiversionChecker(null);
    setViewingEvaluateStat(null);
    setViewingXoLPrice(null);
    setViewingFinalXolPrice(null);
    setViewingEvaluation(null);
  };

  // Model definition modules that should not be executed directly in Run All
  const MODEL_DEFINITION_TYPES: ModuleType[] = [
    // Supervised Learning Models
    ModuleType.LinearRegression,
    ModuleType.LogisticRegression,
    ModuleType.PoissonRegression,
    ModuleType.NegativeBinomialRegression,
    ModuleType.DecisionTree,
    ModuleType.RandomForest,
    ModuleType.SVM,
    ModuleType.LinearDiscriminantAnalysis,
    ModuleType.NaiveBayes,
    ModuleType.KNN,
    // Unsupervised Learning Models
    ModuleType.KMeans,
    ModuleType.HierarchicalClustering,
    ModuleType.DBSCAN,
    ModuleType.PrincipalComponentAnalysis,
    // Traditional Analysis - Statsmodels Models
    ModuleType.OLSModel,
    ModuleType.LogisticModel,
    ModuleType.PoissonModel,
    ModuleType.QuasiPoissonModel,
    ModuleType.NegativeBinomialModel,
    // Statistical Models
    ModuleType.StatModels,
  ];

  // Helper function to check if all upstream modules are successfully executed
  const areUpstreamModulesReady = useCallback(
    (
      moduleId: string,
      allModules: CanvasModule[],
      allConnections: Connection[]
    ): boolean => {
      const upstreamConnections = allConnections.filter(
        (c) => c.to.moduleId === moduleId
      );
      if (upstreamConnections.length === 0) return true; // No dependencies

      return upstreamConnections.every((conn) => {
        const sourceModule = allModules.find(
          (m) => m.id === conn.from.moduleId
        );
        // Model definition modules are always considered ready (they don't need to be executed)
        if (
          sourceModule &&
          MODEL_DEFINITION_TYPES.includes(sourceModule.type)
        ) {
          return true;
        }
        return sourceModule?.status === ModuleStatus.Success;
      });
    },
    []
  );

  const runSimulation = async (
    startModuleId: string,
    runAll: boolean = false
  ) => {
    const runQueue: string[] = [];
    const visited = new Set<string>();
    // 최신 modules 상태를 가져오기 위해 함수 내부에서 참조
    // 클로저를 통해 항상 최신 상태를 참조하도록 함
    const getCurrentModules = () => [...modules];
    let currentModules = getCurrentModules(); // Use a mutable copy for the current simulation run

    const traverse = (moduleId: string) => {
      if (visited.has(moduleId)) return;
      visited.add(moduleId);

      const module = currentModules.find((m) => m.id === moduleId);
      const isModelDefinition =
        module && MODEL_DEFINITION_TYPES.includes(module.type);

      // In Run All mode, skip model definition modules but still process their dependencies
      if (runAll && isModelDefinition) {
        // Still traverse upstream to ensure dependencies are included
        const upstreamConnections = connections.filter(
          (c) => c.to.moduleId === moduleId
        );
        const parentModules = currentModules.filter((m) =>
          upstreamConnections.some((c) => c.from.moduleId === m.id)
        );
        parentModules.forEach((p) => traverse(p.id));

        // Traverse downstream to ensure modules that depend on this model definition are included
        const downstreamConnections = connections.filter(
          (c) => c.from.moduleId === moduleId
        );
        const childModules = currentModules.filter((m) =>
          downstreamConnections.some((c) => c.to.moduleId === m.id)
        );
        childModules.forEach((child) => traverse(child.id));
        return; // Don't add model definition to queue
      }

      // Traverse upstream dependencies first
      const upstreamConnections = connections.filter(
        (c) => c.to.moduleId === moduleId
      );
      const parentModules = currentModules.filter((m) =>
        upstreamConnections.some((c) => c.from.moduleId === m.id)
      );
      parentModules.forEach((p) => traverse(p.id));

      // Add to queue if not already present
      if (!runQueue.includes(moduleId)) {
        runQueue.push(moduleId);
      }

      // In Run All mode, also traverse downstream to ensure all connected modules are included
      if (runAll) {
        const downstreamConnections = connections.filter(
          (c) => c.from.moduleId === moduleId
        );
        const childModules = currentModules.filter((m) =>
          downstreamConnections.some((c) => c.to.moduleId === m.id)
        );
        childModules.forEach((child) => {
          if (!visited.has(child.id)) {
            traverse(child.id);
          }
        });
      }
    };

    if (runAll) {
      // Run All: traverse from all root nodes to include all modules
      // Ignore startModuleId and traverse all root nodes
      const rootNodes = currentModules.filter(
        (m) => !connections.some((c) => c.to.moduleId === m.id)
      );
      if (rootNodes.length > 0) {
        // Traverse all root nodes to ensure all modules are included
        rootNodes.forEach((node) => traverse(node.id));
      } else {
        // If no root nodes (circular dependencies), traverse all modules
        currentModules.forEach((m) => traverse(m.id));
      }
    } else {
      // Individual module run: only run this module (but still check dependencies)
      traverse(startModuleId);
      // Only keep the target module in the queue for individual runs
      runQueue.length = 0;
      runQueue.push(startModuleId);
    }

    const getSingleInputData = (
      moduleId: string,
      portType: Port["type"] = "data"
    ):
      | (DataPreview | MissingHandlerOutput | EncoderOutput | NormalizerOutput)
      | null => {
      const inputConnection = connections.find((c) => {
        if (c.to.moduleId === moduleId) {
          const targetModule = currentModules.find((m) => m.id === moduleId);
          const targetPort = targetModule?.inputs.find(
            (p) => p.name === c.to.portName
          );
          return targetPort?.type === portType;
        }
        return false;
      });

      if (!inputConnection) return null;
      const sourceModule = currentModules.find(
        (sm) => sm.id === inputConnection.from.moduleId
      );
      if (!sourceModule?.outputData) return null;

      if (
        sourceModule.outputData.type === "SplitDataOutput" &&
        portType === "data"
      ) {
        const portName = inputConnection.from.portName;
        if (portName === "train_data_out") return sourceModule.outputData.train;
        if (portName === "test_data_out") return sourceModule.outputData.test;
      }

      if (
        (sourceModule.outputData.type === "DataPreview" &&
          portType === "data") ||
        (sourceModule.outputData.type === "MissingHandlerOutput" &&
          portType === "handler") ||
        (sourceModule.outputData.type === "EncoderOutput" &&
          portType === "handler") ||
        (sourceModule.outputData.type === "NormalizerOutput" &&
          portType === "handler")
      ) {
        return sourceModule.outputData;
      }

      return null;
    };

    for (const moduleId of runQueue) {
      const module = currentModules.find((m) => m.id === moduleId)!;
      const moduleName = module.name;

      // Check if upstream modules are ready (only for individual runs, not Run All)
      if (
        !runAll &&
        !areUpstreamModulesReady(moduleId, currentModules, connections)
      ) {
        addLog(
          "WARN",
          `Module [${moduleName}] cannot run: upstream modules are not ready.`
        );
        setModules((prev) =>
          prev.map((m) =>
            m.id === moduleId ? { ...m, status: ModuleStatus.Error } : m
          )
        );
        continue;
      }

      setModules((prev) =>
        prev.map((m) =>
          m.id === moduleId ? { ...m, status: ModuleStatus.Running } : m
        )
      );
      addLog("INFO", `Module [${moduleName}] execution started.`);

      await new Promise((resolve) => setTimeout(resolve, 500));

      let newStatus = ModuleStatus.Error;
      let newOutputData: CanvasModule["outputData"] | undefined = undefined;
      let logMessage = `Module [${moduleName}] failed.`;
      let logLevel: TerminalLog["level"] = "ERROR";

      try {
        if (
          module.type === ModuleType.LoadData ||
          module.type === ModuleType.XolLoading
        ) {
          const fileContent = module.parameters.fileContent as string;
          if (!fileContent)
            throw new Error(
              "No file content loaded. Please select a CSV file."
            );

          // CSV 파싱 함수 (따옴표 처리 포함)
          const parseCSVLine = (line: string): string[] => {
            const result: string[] = [];
            let current = "";
            let inQuotes = false;

            for (let i = 0; i < line.length; i++) {
              const char = line[i];
              const nextChar = line[i + 1];

              if (char === '"') {
                if (inQuotes && nextChar === '"') {
                  // 이스케이프된 따옴표
                  current += '"';
                  i++; // 다음 문자 건너뛰기
                } else {
                  // 따옴표 시작/끝
                  inQuotes = !inQuotes;
                }
              } else if (char === "," && !inQuotes) {
                // 쉼표로 필드 구분
                result.push(current.trim());
                current = "";
              } else {
                current += char;
              }
            }
            result.push(current.trim()); // 마지막 필드
            return result;
          };

          const lines = fileContent
            .trim()
            .split(/\r?\n/)
            .filter((line) => line.trim() !== "");
          if (lines.length < 1)
            throw new Error("CSV file is empty or invalid.");

          const header = parseCSVLine(lines[0]).map((h) =>
            h.replace(/^"|"$/g, "")
          );
          if (header.length === 0)
            throw new Error("CSV file has no header row.");

          const stringRows = lines.slice(1).map((line) => {
            const values = parseCSVLine(line).map((v) =>
              v.replace(/^"|"$/g, "")
            );
            const rowObj: Record<string, string> = {};
            header.forEach((col, index) => {
              rowObj[col] = values[index] || "";
            });
            return rowObj;
          });

          // 컬럼 이름 중복 처리
          const uniqueHeader: string[] = [];
          const headerCount: Record<string, number> = {};
          header.forEach((name) => {
            const originalName = name || "Unnamed";
            if (headerCount[originalName] !== undefined) {
              headerCount[originalName]++;
              uniqueHeader.push(`${originalName}_${headerCount[originalName]}`);
            } else {
              headerCount[originalName] = 0;
              uniqueHeader.push(originalName);
            }
          });

          const columns: ColumnInfo[] = uniqueHeader.map((name) => {
            const sample = stringRows
              .slice(0, 100)
              .map((r) => r[name])
              .filter(
                (v) => v !== undefined && v !== null && String(v).trim() !== ""
              );
            const allAreNumbers =
              sample.length > 0 &&
              sample.every((v) => {
                const num = Number(v);
                return !isNaN(num) && isFinite(num);
              });
            return { name, type: allAreNumbers ? "number" : "string" };
          });

          if (columns.length === 0) {
            throw new Error("No valid columns found in CSV file.");
          }

          const rows = stringRows
            .map((stringRow, rowIndex) => {
              const typedRow: Record<string, string | number | null> = {};
              for (const col of columns) {
                const val = stringRow[col.name];
                if (col.type === "number") {
                  const numVal =
                    val && String(val).trim() !== ""
                      ? parseFloat(String(val))
                      : null;
                  typedRow[col.name] =
                    numVal !== null && !isNaN(numVal) && isFinite(numVal)
                      ? numVal
                      : null;
                } else {
                  typedRow[col.name] = val || null;
                }
              }
              return typedRow;
            })
            .filter((row) => {
              // 모든 값이 null인 행 제거
              return Object.values(row).some(
                (val) => val !== null && val !== ""
              );
            });

          if (rows.length === 0) {
            throw new Error(
              "No valid data rows found in CSV file after parsing."
            );
          }

          // 전체 데이터를 저장 (View Details에서는 미리보기만 제한하여 표시)
          newOutputData = {
            type: "DataPreview",
            columns,
            totalRowCount: rows.length,
            rows: rows,
          };
        } else if (module.type === ModuleType.SelectData) {
          const inputData = getSingleInputData(module.id) as DataPreview;
          if (inputData) {
            const selections =
              (module.parameters.columnSelections as Record<
                string,
                { selected: boolean; type: string }
              >) || {};
            const isConfigured = Object.keys(selections).length > 0;

            const newColumns: ColumnInfo[] = [];
            inputData.columns.forEach((col) => {
              const selection = selections[col.name];
              // If the module is unconfigured, default to selecting all columns. Otherwise, respect the selection.
              if (!isConfigured || selection?.selected) {
                newColumns.push({
                  name: col.name,
                  type: selection?.type ?? col.type,
                });
              }
            });

            if (
              isConfigured &&
              newColumns.length === 0 &&
              inputData.columns.length > 0
            ) {
              throw new Error("No columns selected.");
            }

            const newRows = (inputData.rows || []).map((row) => {
              const newRow: Record<string, any> = {};
              newColumns.forEach((col) => {
                const originalValue = row[col.name];
                let newValue = originalValue; // Default to original

                if (col.type === "number") {
                  // Preserve null/undefined, and convert empty strings to null.
                  // If conversion to number fails (NaN), also treat as null.
                  if (
                    originalValue === null ||
                    originalValue === undefined ||
                    String(originalValue).trim() === ""
                  ) {
                    newValue = null;
                  } else {
                    const num = Number(originalValue);
                    newValue = isNaN(num) ? null : num;
                  }
                } else if (col.type === "string") {
                  // Preserve null/undefined instead of converting to "null" or empty string.
                  newValue =
                    originalValue === null || originalValue === undefined
                      ? null
                      : String(originalValue);
                }
                // For any other data types, the original value is preserved by default.

                newRow[col.name] = newValue;
              });
              return newRow;
            });
            newOutputData = {
              type: "DataPreview",
              columns: newColumns,
              totalRowCount: inputData.totalRowCount,
              rows: newRows,
            };
          } else {
            throw new Error(
              "Input data not available or is of the wrong type."
            );
          }
        } else if (module.type === ModuleType.HandleMissingValues) {
          const inputData = getSingleInputData(module.id) as DataPreview;
          if (!inputData) throw new Error("Input data not available.");

          const { method, strategy, columns, n_neighbors } = module.parameters;

          // Pyodide를 사용하여 Python으로 결측치 처리 통계 계산
          try {
            addLog(
              "INFO",
              "Pyodide를 사용하여 Python으로 결측치 처리 통계 계산 중..."
            );

            const pyodideModule = await import("./utils/pyodideRunner");
            const { handleMissingValuesPython } = pyodideModule;

            const result = await handleMissingValuesPython(
              inputData.rows || [],
              method || "impute",
              strategy || "mean",
              columns || null,
              parseInt(n_neighbors) || 5,
              60000 // 타임아웃: 60초
            );

            newOutputData = {
              type: "MissingHandlerOutput",
              method,
              strategy: strategy || "mean",
              n_neighbors: parseInt(n_neighbors) || 5,
              metric: module.parameters.metric,
              imputation_values: result.imputation_values,
            };

            addLog("SUCCESS", "Python으로 결측치 처리 통계 계산 완료");
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            addLog("ERROR", `Python HandleMissingValues 실패: ${errorMessage}`);
            throw new Error(`결측치 처리 실패: ${errorMessage}`);
          }
        } else if (module.type === ModuleType.TransformData) {
          const handler = getSingleInputData(module.id, "handler") as
            | MissingHandlerOutput
            | EncoderOutput
            | NormalizerOutput;
          const inputData = getSingleInputData(
            module.id,
            "data"
          ) as DataPreview;

          if (!handler)
            throw new Error(
              "A handler module must be connected to 'handler_in'."
            );
          if (!inputData)
            throw new Error("A data module must be connected to 'data_in'.");

          const { exclude_columns = [] } = module.parameters;

          // Pyodide를 사용하여 Python으로 변환 적용
          try {
            addLog("INFO", "Pyodide를 사용하여 Python으로 변환 적용 중...");

            const pyodideModule = await import("./utils/pyodideRunner");
            const { applyTransformPython } = pyodideModule;

            const result = await applyTransformPython(
              inputData.rows || [],
              handler,
              exclude_columns || [],
              60000 // 타임아웃: 60초
            );

            newOutputData = {
              ...inputData,
              columns: result.columns,
              rows: result.rows,
              totalRowCount: result.rows.length,
            };

            addLog("SUCCESS", "Python으로 변환 적용 완료");
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            addLog("ERROR", `Python TransformData 실패: ${errorMessage}`);
            throw new Error(`변환 적용 실패: ${errorMessage}`);
          }
        } else if (module.type === ModuleType.EncodeCategorical) {
          const inputData = getSingleInputData(module.id) as DataPreview;
          if (!inputData) throw new Error("Input data not available.");
          const {
            method,
            columns: targetColumns,
            ordinal_mapping: ordinalMappingStr,
            drop,
            handle_unknown,
          } = module.parameters;

          const columnsToEncode =
            targetColumns && targetColumns.length > 0
              ? targetColumns
              : inputData.columns
                  .filter((c) => c.type === "string")
                  .map((c) => c.name);

          // Pyodide를 사용하여 Python으로 인코딩 매핑 생성
          try {
            addLog(
              "INFO",
              "Pyodide를 사용하여 Python으로 인코딩 매핑 생성 중..."
            );

            const pyodideModule = await import("./utils/pyodideRunner");
            const { encodeCategoricalPython } = pyodideModule;

            let ordinalMapping: Record<string, string[]> | null = null;
            if (ordinalMappingStr) {
              try {
                ordinalMapping = JSON.parse(ordinalMappingStr);
              } catch (e) {
                ordinalMapping = null;
              }
            }

            const result = await encodeCategoricalPython(
              inputData.rows || [],
              method || "label",
              columnsToEncode.length > 0 ? columnsToEncode : null,
              ordinalMapping,
              drop || "first",
              handle_unknown || "ignore",
              60000 // 타임아웃: 60초
            );

            newOutputData = {
              type: "EncoderOutput",
              method,
              mappings: result.mappings,
              columns_to_encode: columnsToEncode,
              drop: drop,
              handle_unknown: handle_unknown,
            };

            addLog("SUCCESS", "Python으로 인코딩 매핑 생성 완료");
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            addLog("ERROR", `Python EncodeCategorical 실패: ${errorMessage}`);
            throw new Error(`인코딩 매핑 생성 실패: ${errorMessage}`);
          }
        } else if (module.type === ModuleType.NormalizeData) {
          const inputData = getSingleInputData(module.id) as DataPreview;
          if (!inputData) throw new Error("Input data not available.");

          const selections =
            (module.parameters.columnSelections as Record<
              string,
              { selected: boolean }
            >) || {};
          const method = module.parameters.method as NormalizerOutput["method"];

          const columnsToNormalize = inputData.columns
            .filter(
              (col) => selections[col.name]?.selected && col.type === "number"
            )
            .map((col) => col.name);

          // Pyodide를 사용하여 Python으로 정규화 통계 계산
          try {
            addLog(
              "INFO",
              "Pyodide를 사용하여 Python으로 정규화 통계 계산 중..."
            );

            const pyodideModule = await import("./utils/pyodideRunner");
            const { normalizeDataPython } = pyodideModule;

            const result = await normalizeDataPython(
              inputData.rows || [],
              method || "MinMax",
              columnsToNormalize,
              60000 // 타임아웃: 60초
            );

            newOutputData = {
              type: "NormalizerOutput",
              method,
              stats: result.stats,
            };

            addLog("SUCCESS", "Python으로 정규화 통계 계산 완료");
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            addLog("ERROR", `Python NormalizeData 실패: ${errorMessage}`);
            throw new Error(`정규화 통계 계산 실패: ${errorMessage}`);
          }
        } else if (module.type === ModuleType.TransitionData) {
          const inputData = getSingleInputData(module.id) as DataPreview;
          if (!inputData) throw new Error("Input data not available.");

          const transformations =
            (module.parameters.transformations as Record<string, string>) || {};

          // Pyodide를 사용하여 Python으로 수학적 변환 수행
          try {
            addLog(
              "INFO",
              "Pyodide를 사용하여 Python으로 데이터 변환 수행 중..."
            );

            const pyodideModule = await import("./utils/pyodideRunner");
            const { transformDataPython } = pyodideModule;

            const result = await transformDataPython(
              inputData.rows || [],
              transformations,
              60000 // 타임아웃: 60초
            );

            newOutputData = {
              type: "DataPreview",
              columns: result.columns,
              totalRowCount: result.rows.length,
              rows: result.rows,
            };

            addLog("SUCCESS", "Python으로 데이터 변환 완료");
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            addLog("ERROR", `Python TransitionData 실패: ${errorMessage}`);
            throw new Error(`데이터 변환 실패: ${errorMessage}`);
          }
        } else if (module.type === ModuleType.ResampleData) {
          const inputData = getSingleInputData(module.id) as DataPreview;
          if (!inputData) throw new Error("Input data not available.");

          const { method, target_column } = module.parameters;
          if (!target_column)
            throw new Error("Target Column parameter is not set.");

          const inputRows = inputData.rows || [];
          if (inputRows.length === 0) {
            newOutputData = { ...inputData }; // Pass through empty data
          } else {
            const groups: Record<string, Record<string, any>[]> = {};
            inputRows.forEach((row) => {
              const key = String(row[target_column]);
              if (!groups[key]) {
                groups[key] = [];
              }
              groups[key].push(row);
            });

            let newRows: Record<string, any>[] = [];

            if (method === "SMOTE") {
              const counts = Object.values(groups).map((g) => g.length);
              const maxCount = Math.max(...counts);

              for (const key in groups) {
                const classRows = groups[key];
                newRows.push(...classRows);
                const diff = maxCount - classRows.length;
                for (let i = 0; i < diff; i++) {
                  // Simple random over-sampling as a simulation of SMOTE
                  newRows.push(
                    classRows[Math.floor(Math.random() * classRows.length)]
                  );
                }
              }
            } else if (method === "NearMiss") {
              const counts = Object.values(groups).map((g) => g.length);
              const minCount = Math.min(...counts);

              for (const key in groups) {
                const classRows = groups[key];
                // Shuffle for random undersampling
                for (let i = classRows.length - 1; i > 0; i--) {
                  const j = Math.floor(Math.random() * (i + 1));
                  [classRows[i], classRows[j]] = [classRows[j], classRows[i]];
                }
                newRows.push(...classRows.slice(0, minCount));
              }
            }

            // Final shuffle of the entire dataset
            for (let i = newRows.length - 1; i > 0; i--) {
              const j = Math.floor(Math.random() * (i + 1));
              [newRows[i], newRows[j]] = [newRows[j], newRows[i]];
            }

            newOutputData = {
              type: "DataPreview",
              columns: inputData.columns,
              totalRowCount: newRows.length,
              rows: newRows,
            };
          }
        } else if (module.type === ModuleType.SplitData) {
          const inputData = getSingleInputData(module.id) as DataPreview;
          if (!inputData) throw new Error("Input data not available.");

          const {
            train_size,
            random_state,
            shuffle,
            stratify,
            stratify_column,
          } = module.parameters;
          const inputRows = inputData.rows || [];

          // Pyodide를 사용하여 브라우저에서 직접 Python 실행
          // Python의 sklearn.train_test_split과 동일한 결과를 보장합니다.
          // 타임아웃 발생 시 Node.js 백엔드로 전환합니다.

          let useNodeBackend = false;
          let pyodideErrorForNode = "";
          const totalTimeout = 180000; // 전체 타임아웃: 180초 (3분)
          const startTime = Date.now();

          try {
            // Pyodide 동적 import
            addLog(
              "INFO",
              "Pyodide를 사용하여 Python으로 데이터 분할 중... (최대 3분)"
            );

            const pyodideModule = await import("./utils/pyodideRunner");
            const { splitDataPython } = pyodideModule;

            // 전체 타임아웃을 포함한 Python 실행 시도
            const executionPromise = splitDataPython(
              inputRows,
              parseFloat(train_size),
              parseInt(random_state),
              shuffle === "True" || shuffle === true,
              stratify === "True" || stratify === true,
              stratify_column || null,
              120000 // Python 실행 타임아웃: 120초 (2분)
            );

            // 전체 타임아웃 래퍼
            const timeoutPromise = new Promise<{
              trainIndices: number[];
              testIndices: number[];
            }>((_, reject) => {
              const elapsed = Date.now() - startTime;
              const remaining = totalTimeout - elapsed;
              if (remaining <= 0) {
                reject(new Error("전체 실행 타임아웃 (3분 초과)"));
              } else {
                setTimeout(
                  () => reject(new Error("전체 실행 타임아웃 (3분 초과)")),
                  remaining
                );
              }
            });

            const { trainIndices, testIndices } = await Promise.race([
              executionPromise,
              timeoutPromise,
            ]);

            const elapsedTime = Date.now() - startTime;
            addLog(
              "INFO",
              `Pyodide 실행 완료 (소요 시간: ${(elapsedTime / 1000).toFixed(
                1
              )}초)`
            );

            // Python에서 받은 인덱스를 사용하여 데이터 분할
            const trainRows = trainIndices.map((i: number) => inputRows[i]);
            const testRows = testIndices.map((i: number) => inputRows[i]);

            const totalTrainCount = Math.floor(
              inputData.totalRowCount * parseFloat(train_size)
            );
            const totalTestCount = inputData.totalRowCount - totalTrainCount;

            const trainData: DataPreview = {
              type: "DataPreview",
              columns: inputData.columns,
              totalRowCount: totalTrainCount,
              rows: trainRows,
            };
            const testData: DataPreview = {
              type: "DataPreview",
              columns: inputData.columns,
              totalRowCount: totalTestCount,
              rows: testRows,
            };

            newOutputData = {
              type: "SplitDataOutput",
              train: trainData,
              test: testData,
            };
            addLog(
              "SUCCESS",
              "Python으로 데이터 분할 완료 (sklearn.train_test_split 사용)"
            );
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            const elapsedTime = Date.now() - startTime;

            // Pyodide 에러 메시지 저장
            pyodideErrorForNode = errorMessage;

            // 타임아웃이거나 Pyodide 실패 시 Node.js 백엔드로 전환
            // "Failed to fetch"는 네트워크 오류이므로 Pyodide 오류로 간주하고 Node.js 백엔드로 전환
            if (
              errorMessage.includes("타임아웃") ||
              errorMessage.includes("timeout") ||
              errorMessage.includes("Timeout") ||
              errorMessage.includes("Failed to fetch") ||
              errorMessage.includes("NetworkError")
            ) {
              addLog(
                "WARN",
                `Pyodide 타임아웃/오류 발생 (${(elapsedTime / 1000).toFixed(
                  1
                )}초 경과), Node.js 백엔드로 전환: ${errorMessage}`
              );
              useNodeBackend = true;
            } else {
              addLog(
                "WARN",
                `Pyodide 실행 실패 (${(elapsedTime / 1000).toFixed(
                  1
                )}초 경과), Node.js 백엔드로 전환: ${errorMessage}`
              );
              useNodeBackend = true;
            }
          }

          // Node.js 백엔드로 전환
          if (useNodeBackend) {
            try {
              addLog(
                "INFO",
                "Node.js 백엔드를 통해 Python으로 데이터 분할 중... (최대 2분)"
              );

              // Node.js 백엔드 API 호출 (타임아웃: 120초)
              const nodeBackendPromise = fetch("/api/split-data", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  data: inputRows,
                  train_size: parseFloat(train_size),
                  random_state: parseInt(random_state),
                  shuffle: shuffle === "True" || shuffle === true,
                  stratify: stratify === "True" || stratify === true,
                  stratify_column: stratify_column || null,
                }),
              });

              const nodeTimeoutPromise = new Promise<Response>((_, reject) =>
                setTimeout(
                  () => reject(new Error("Node.js 백엔드 타임아웃 (2분 초과)")),
                  120000
                )
              );

              const response = await Promise.race([
                nodeBackendPromise,
                nodeTimeoutPromise,
              ]);

              if (response.ok) {
                const { trainIndices, testIndices } = await response.json();

                // Node.js 백엔드에서 받은 인덱스를 사용하여 데이터 분할
                const trainRows = trainIndices.map((i: number) => inputRows[i]);
                const testRows = testIndices.map((i: number) => inputRows[i]);

                const totalTrainCount = Math.floor(
                  inputData.totalRowCount * parseFloat(train_size)
                );
                const totalTestCount =
                  inputData.totalRowCount - totalTrainCount;

                const trainData: DataPreview = {
                  type: "DataPreview",
                  columns: inputData.columns,
                  totalRowCount: totalTrainCount,
                  rows: trainRows,
                };
                const testData: DataPreview = {
                  type: "DataPreview",
                  columns: inputData.columns,
                  totalRowCount: totalTestCount,
                  rows: testRows,
                };

                newOutputData = {
                  type: "SplitDataOutput",
                  train: trainData,
                  test: testData,
                };
                addLog(
                  "SUCCESS",
                  "Node.js 백엔드로 데이터 분할 완료 (sklearn.train_test_split 사용)"
                );
              } else {
                const errorText = await response.text();
                throw new Error(
                  `Node.js 백엔드 응답 오류: ${response.status} - ${errorText}`
                );
              }
            } catch (nodeError: any) {
              // Node.js 백엔드도 실패하면 에러 발생
              const nodeErrorMessage = nodeError.message || String(nodeError);

              // Pyodide 에러 메시지 (이전 catch 블록에서 저장된 에러)
              const pyodideErrorMsg =
                typeof pyodideErrorForNode !== "undefined"
                  ? pyodideErrorForNode
                  : "알 수 없는 Pyodide 오류";

              // Node.js 백엔드 에러 메시지
              let nodeErrorMsg = "";
              if (
                nodeErrorMessage.includes("Failed to fetch") ||
                nodeErrorMessage.includes("NetworkError") ||
                nodeErrorMessage.includes("ERR_CONNECTION_REFUSED")
              ) {
                nodeErrorMsg =
                  'Express 서버(포트 3001)를 찾을 수 없습니다. 터미널에서 "pnpm run server" 또는 "pnpm run dev:full" 명령어로 Express 서버를 실행하세요.';
              } else if (nodeErrorMessage.includes("타임아웃")) {
                nodeErrorMsg = `Express 서버 타임아웃: ${nodeErrorMessage}`;
              } else {
                nodeErrorMsg = `Express 서버 오류: ${nodeErrorMessage}`;
              }

              // js_tuning_options 관련 에러인 경우 더 명확한 메시지 제공
              let enhancedPyodideError = pyodideErrorMsg;
              if (
                pyodideErrorMsg.includes("js_tuning_options") ||
                pyodideErrorMsg.includes("KeyError")
              ) {
                enhancedPyodideError = `내부 오류 (이미 수정됨): ${pyodideErrorMsg}. 페이지를 새로고침하고 다시 시도해주세요.`;
              }

              throw new Error(
                `데이터 분할 실패: Pyodide와 Express 서버 모두 실패했습니다.\n\nPyodide 오류: ${enhancedPyodideError}\n\nExpress 서버 오류: ${nodeErrorMsg}\n\n해결 방법:\n1. 페이지를 새로고침하고 다시 시도\n2. Express 서버 실행: "pnpm run server" 또는 "pnpm run dev:full"\n3. Python이 설치되어 있고 sklearn, pandas가 설치되어 있는지 확인: "pip install scikit-learn pandas"`
              );
            }
          }
        } else if (module.type === ModuleType.Statistics) {
          const inputData = getSingleInputData(module.id) as DataPreview;
          if (!inputData || !inputData.rows) {
            throw new Error(
              "Input data not available or is of the wrong type."
            );
          }

          // Pyodide를 사용하여 Python으로 통계 계산
          try {
            addLog("INFO", "Pyodide를 사용하여 Python으로 통계 계산 중...");

            const pyodideModule = await import("./utils/pyodideRunner");
            const { calculateStatisticsPython } = pyodideModule;

            const result = await calculateStatisticsPython(
              inputData.rows,
              inputData.columns,
              60000 // 타임아웃: 60초
            );

            newOutputData = {
              type: "StatisticsOutput",
              stats: result.stats,
              correlation: result.correlation,
              columns: inputData.columns,
            };
            addLog("SUCCESS", "Python으로 통계 계산 완료");
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            addLog("ERROR", `Python 통계 계산 실패: ${errorMessage}`);
            throw new Error(`통계 계산 실패: ${errorMessage}`);
          }
        } else if (module.type === ModuleType.TrainModel) {
          const modelInputConnection = connections.find(
            (c) => c.to.moduleId === module.id && c.to.portName === "model_in"
          );
          const dataInputConnection = connections.find(
            (c) => c.to.moduleId === module.id && c.to.portName === "data_in"
          );

          if (!modelInputConnection || !dataInputConnection) {
            throw new Error(
              "Both 'model_in' and 'data_in' ports must be connected."
            );
          }

          const modelSourceModule = currentModules.find(
            (m) => m.id === modelInputConnection.from.moduleId
          );
          if (!modelSourceModule)
            throw new Error("Model source module not found.");

          const dataSourceModule = currentModules.find(
            (m) => m.id === dataInputConnection.from.moduleId
          );
          if (!dataSourceModule || !dataSourceModule.outputData)
            throw new Error("Data source module has no output.");

          let inputData: DataPreview | null = null;
          if (dataSourceModule.outputData.type === "DataPreview") {
            inputData = dataSourceModule.outputData;
          } else if (dataSourceModule.outputData.type === "SplitDataOutput") {
            const portName = dataInputConnection.from.portName;
            if (portName === "train_data_out") {
              inputData = dataSourceModule.outputData.train;
            } else if (portName === "test_data_out") {
              inputData = dataSourceModule.outputData.test;
            }
          }

          if (!inputData)
            throw new Error(
              "Input data not available or is of the wrong type."
            );

          const { feature_columns, label_column } = module.parameters;
          if (
            !feature_columns ||
            feature_columns.length === 0 ||
            !label_column
          ) {
            throw new Error("Feature and label columns are not configured.");
          }

          const ordered_feature_columns = inputData.columns
            .map((c) => c.name)
            .filter((name) => feature_columns.includes(name));

          if (ordered_feature_columns.length === 0) {
            throw new Error("No valid feature columns found in the data.");
          }

          let trainedModelOutput: TrainedModelOutput | undefined = undefined;
          let intercept = 0;
          const coefficients: Record<string, number> = {};
          const metrics: Record<string, number> = {};

          const modelIsClassification = isClassification(
            modelSourceModule.type,
            modelSourceModule.parameters.model_purpose
          );
          const modelIsRegression = !modelIsClassification;

          // Prepare data for training
          const rows = inputData.rows || [];
          if (rows.length === 0) {
            throw new Error("No data rows available for training.");
          }

          // Extract feature matrix X and target vector y
          const X: number[][] = [];
          const y: number[] = [];

          if (!rows || rows.length === 0) {
            throw new Error("Input data has no rows.");
          }

          if (
            !ordered_feature_columns ||
            ordered_feature_columns.length === 0
          ) {
            throw new Error("No feature columns specified.");
          }

          for (let rowIdx = 0; rowIdx < rows.length; rowIdx++) {
            const row = rows[rowIdx];
            if (!row) {
              continue; // Skip null/undefined rows
            }

            const featureRow: number[] = [];
            let hasValidFeatures = true;

            for (
              let colIdx = 0;
              colIdx < ordered_feature_columns.length;
              colIdx++
            ) {
              const col = ordered_feature_columns[colIdx];
              if (!col) {
                hasValidFeatures = false;
                break;
              }
              const value = row[col];
              if (
                typeof value === "number" &&
                !isNaN(value) &&
                value !== null &&
                value !== undefined
              ) {
                featureRow.push(value);
              } else {
                hasValidFeatures = false;
                break;
              }
            }

            if (!hasValidFeatures) {
              continue; // Skip rows with invalid features
            }

            if (featureRow.length !== ordered_feature_columns.length) {
              continue; // Skip rows with incomplete features
            }

            const labelValue = row[label_column];
            if (
              typeof labelValue === "number" &&
              !isNaN(labelValue) &&
              labelValue !== null &&
              labelValue !== undefined
            ) {
              X.push(featureRow);
              y.push(labelValue);
            }
          }

          if (X.length === 0) {
            throw new Error(
              `No valid data rows found after filtering. Checked ${
                rows.length
              } rows. Ensure feature columns (${ordered_feature_columns.join(
                ", "
              )}) and label column (${label_column}) contain valid numeric values.`
            );
          }

          if (X.length < ordered_feature_columns.length) {
            throw new Error(
              `Insufficient data: need at least ${ordered_feature_columns.length} samples for ${ordered_feature_columns.length} features, but only found ${X.length} valid rows.`
            );
          }

          // tuningSummary를 초기화 (모든 모델 타입에서 사용 가능하도록)
          let tuningSummary: TrainedModelOutput["tuningSummary"] = undefined;

          if (modelIsRegression) {
            // Pyodide를 사용하여 Python으로 Linear Regression 훈련
            if (modelSourceModule.type === ModuleType.LinearRegression) {
              const fitIntercept =
                modelSourceModule.parameters.fit_intercept === "True";
              const modelType =
                modelSourceModule.parameters.model_type || "LinearRegression";
              const alpha =
                parseFloat(modelSourceModule.parameters.alpha) || 1.0;
              const l1_ratio =
                parseFloat(modelSourceModule.parameters.l1_ratio) || 0.5;
              const parseCandidates = (
                raw: any,
                fallback: number[]
              ): number[] => {
                if (Array.isArray(raw)) {
                  const parsed = raw
                    .map((val) => {
                      const num =
                        typeof val === "number" ? val : parseFloat(val);
                      return isNaN(num) ? null : num;
                    })
                    .filter((num): num is number => num !== null);
                  return parsed.length ? parsed : fallback;
                }
                if (typeof raw === "string") {
                  const parsed = raw
                    .split(",")
                    .map((part) => parseFloat(part.trim()))
                    .filter((num) => !isNaN(num));
                  return parsed.length ? parsed : fallback;
                }
                if (typeof raw === "number" && !isNaN(raw)) {
                  return [raw];
                }
                return fallback;
              };
              const tuningEnabled =
                modelSourceModule.parameters.tuning_enabled === "True";
              const tuningOptions = tuningEnabled
                ? {
                    enabled: true,
                    strategy: "GridSearch" as const,
                    alphaCandidates: parseCandidates(
                      modelSourceModule.parameters.alpha_candidates,
                      [alpha]
                    ),
                    l1RatioCandidates:
                      modelType === "ElasticNet"
                        ? parseCandidates(
                            modelSourceModule.parameters.l1_ratio_candidates,
                            [l1_ratio]
                          )
                        : undefined,
                    cvFolds: Math.max(
                      2,
                      parseInt(modelSourceModule.parameters.cv_folds, 10) || 5
                    ),
                    scoringMetric:
                      modelSourceModule.parameters.scoring_metric ||
                      "neg_mean_squared_error",
                  }
                : undefined;

              if (X.length < ordered_feature_columns.length) {
                throw new Error(
                  `Insufficient data: need at least ${ordered_feature_columns.length} samples for ${ordered_feature_columns.length} features, but got ${X.length}.`
                );
              }

              try {
                addLog(
                  "INFO",
                  `Pyodide를 사용하여 Python으로 ${modelType} 모델 훈련 중...`
                );

                const pyodideModule = await import("./utils/pyodideRunner");
                const { fitLinearRegressionPython } = pyodideModule;

                const fitResult = await fitLinearRegressionPython(
                  X,
                  y,
                  modelType,
                  fitIntercept,
                  alpha,
                  l1_ratio,
                  ordered_feature_columns, // feature columns 전달
                  60000, // 타임아웃: 60초
                  tuningOptions
                );

                if (
                  !fitResult.coefficients ||
                  fitResult.coefficients.length !==
                    ordered_feature_columns.length
                ) {
                  throw new Error(
                    `Coefficient count mismatch: expected ${
                      ordered_feature_columns.length
                    }, got ${fitResult.coefficients?.length || 0}.`
                  );
                }

                intercept = fitResult.intercept;
                ordered_feature_columns.forEach((col, idx) => {
                  if (fitResult.coefficients[idx] !== undefined) {
                    coefficients[col] = fitResult.coefficients[idx];
                  } else {
                    throw new Error(
                      `Missing coefficient for feature ${col} at index ${idx}.`
                    );
                  }
                });
                tuningSummary = fitResult.tuning
                  ? {
                      enabled: Boolean(fitResult.tuning.enabled),
                      strategy: fitResult.tuning.strategy,
                      bestParams: fitResult.tuning.bestParams,
                      bestScore:
                        typeof fitResult.tuning.bestScore === "number"
                          ? fitResult.tuning.bestScore
                          : undefined,
                      scoringMetric: fitResult.tuning.scoringMetric,
                      candidates: Array.isArray(fitResult.tuning.candidates)
                        ? fitResult.tuning.candidates
                        : undefined,
                    }
                  : undefined;
                if (tuningSummary?.enabled && tuningSummary.bestParams) {
                  addLog(
                    "INFO",
                    `Hyperparameter tuning selected params: ${Object.entries(
                      tuningSummary.bestParams
                    )
                      .map(([k, v]) => `${k}=${v}`)
                      .join(", ")}.`
                  );
                }

                // Python에서 계산된 메트릭 사용
                const r2Value =
                  typeof fitResult.metrics["R-squared"] === "number"
                    ? fitResult.metrics["R-squared"]
                    : parseFloat(fitResult.metrics["R-squared"]);
                const mseValue =
                  typeof fitResult.metrics["Mean Squared Error"] === "number"
                    ? fitResult.metrics["Mean Squared Error"]
                    : parseFloat(fitResult.metrics["Mean Squared Error"]);
                const rmseValue =
                  typeof fitResult.metrics["Root Mean Squared Error"] ===
                  "number"
                    ? fitResult.metrics["Root Mean Squared Error"]
                    : parseFloat(fitResult.metrics["Root Mean Squared Error"]);

                metrics["R-squared"] = parseFloat(r2Value.toFixed(4));
                metrics["Mean Squared Error"] = parseFloat(mseValue.toFixed(4));
                metrics["Root Mean Squared Error"] = parseFloat(
                  rmseValue.toFixed(4)
                );

                addLog("SUCCESS", `Python으로 ${modelType} 모델 훈련 완료`);
              } catch (error: any) {
                const errorMessage = error.message || String(error);
                addLog(
                  "ERROR",
                  `Python LinearRegression 훈련 실패: ${errorMessage}`
                );
                throw new Error(`모델 훈련 실패: ${errorMessage}`);
              }
            } else if (
              modelSourceModule.type === ModuleType.PoissonModel ||
              modelSourceModule.type === ModuleType.QuasiPoissonModel ||
              modelSourceModule.type === ModuleType.NegativeBinomialModel ||
              modelSourceModule.type === ModuleType.PoissonRegression ||
              modelSourceModule.type === ModuleType.NegativeBinomialRegression
            ) {
              // statsmodels를 사용한 포아송/음이항/Quasi-Poisson 회귀
              let distributionType: string;
              let maxIter: number;
              let disp: number;

              if (modelSourceModule.type === ModuleType.PoissonModel) {
                distributionType = "Poisson";
                maxIter =
                  parseInt(modelSourceModule.parameters.max_iter, 10) || 100;
                disp = 1.0;
              } else if (
                modelSourceModule.type === ModuleType.QuasiPoissonModel
              ) {
                distributionType = "QuasiPoisson";
                maxIter =
                  parseInt(modelSourceModule.parameters.max_iter, 10) || 100;
                disp = 1.0;
              } else if (
                modelSourceModule.type === ModuleType.NegativeBinomialModel
              ) {
                distributionType = "NegativeBinomial";
                maxIter =
                  parseInt(modelSourceModule.parameters.max_iter, 10) || 100;
                disp = parseFloat(modelSourceModule.parameters.disp) || 1.0;
              } else {
                // 기존 모듈 (deprecated)
                distributionType =
                  modelSourceModule.parameters.distribution_type ||
                  (modelSourceModule.type === ModuleType.PoissonRegression
                    ? "Poisson"
                    : "NegativeBinomial");
                maxIter =
                  parseInt(modelSourceModule.parameters.max_iter, 10) || 100;
                disp = parseFloat(modelSourceModule.parameters.disp) || 1.0;
              }

              try {
                addLog(
                  "INFO",
                  `Pyodide를 사용하여 Python으로 ${distributionType} 회귀 모델 훈련 중 (statsmodels)...`
                );

                const pyodideModule = await import("./utils/pyodideRunner");
                const { fitCountRegressionStatsmodels } = pyodideModule;

                const fitResult = await fitCountRegressionStatsmodels(
                  X,
                  y,
                  distributionType,
                  ordered_feature_columns,
                  maxIter,
                  disp,
                  60000 // 타임아웃: 60초
                );

                intercept = fitResult.intercept;
                Object.entries(fitResult.coefficients).forEach(
                  ([col, coef]) => {
                    coefficients[col] = coef;
                  }
                );

                // 통계량 설정
                Object.entries(fitResult.metrics).forEach(([key, value]) => {
                  if (typeof value === "number") {
                    metrics[key] = parseFloat(value.toFixed(4));
                  } else {
                    metrics[key] = value;
                  }
                });

                // TrainedModelOutput에 summary 정보 추가 (StatsModelsResultOutput 형식으로)
                trainedModelOutput = {
                  type: "TrainedModelOutput",
                  modelType: modelSourceModule.type,
                  modelPurpose: "regression",
                  coefficients,
                  intercept,
                  metrics,
                  featureColumns: ordered_feature_columns,
                  labelColumn: label_column,
                  tuningSummary: undefined,
                  // statsmodels 결과를 StatsModelsResultOutput 형식으로 저장
                  statsModelsResult: {
                    type: "StatsModelsResultOutput",
                    summary: fitResult.summary,
                    modelType: distributionType as StatsModelFamily,
                    labelColumn: label_column,
                    featureColumns: ordered_feature_columns,
                  },
                };

                addLog(
                  "SUCCESS",
                  `Python으로 ${distributionType} 회귀 모델 훈련 완료 (statsmodels)`
                );
              } catch (error: any) {
                const errorMessage = error.message || String(error);
                addLog(
                  "ERROR",
                  `Python ${distributionType} 회귀 훈련 실패: ${errorMessage}`
                );
                throw new Error(`모델 훈련 실패: ${errorMessage}`);
              }
            } else {
              // For other regression models, use simulation for now
              intercept = Math.random() * 10;
              ordered_feature_columns.forEach((col) => {
                coefficients[col] = Math.random() * 5 - 2.5;
              });
              metrics["R-squared"] = 0.65 + Math.random() * 0.25;
              metrics["Mean Squared Error"] = 150 - Math.random() * 100;
              metrics["Root Mean Squared Error"] = Math.sqrt(
                metrics["Mean Squared Error"]
              );
            }
          } else if (modelIsClassification) {
            // Pyodide를 사용하여 Python으로 Logistic Regression 훈련
            if (modelSourceModule.type === ModuleType.LogisticRegression) {
              const penalty = modelSourceModule.parameters.penalty || "l2";
              const C = parseFloat(modelSourceModule.parameters.C) || 1.0;
              const solver = modelSourceModule.parameters.solver || "lbfgs";
              const maxIter =
                parseInt(modelSourceModule.parameters.max_iter, 10) || 100;

              const parseCandidates = (
                raw: any,
                fallback: number[]
              ): number[] => {
                if (Array.isArray(raw)) {
                  const parsed = raw
                    .map((val) => {
                      const num =
                        typeof val === "number" ? val : parseFloat(val);
                      return isNaN(num) ? null : num;
                    })
                    .filter((num): num is number => num !== null);
                  return parsed.length ? parsed : fallback;
                }
                if (typeof raw === "string") {
                  const parsed = raw
                    .split(",")
                    .map((part) => parseFloat(part.trim()))
                    .filter((num) => !isNaN(num));
                  return parsed.length ? parsed : fallback;
                }
                if (typeof raw === "number" && !isNaN(raw)) {
                  return [raw];
                }
                return fallback;
              };
              const tuningEnabled =
                modelSourceModule.parameters.tuning_enabled === "True";
              const tuningOptions = tuningEnabled
                ? {
                    enabled: true,
                    strategy: "GridSearch" as const,
                    cCandidates: parseCandidates(
                      modelSourceModule.parameters.c_candidates,
                      [C]
                    ),
                    l1RatioCandidates:
                      penalty === "elasticnet"
                        ? parseCandidates(
                            modelSourceModule.parameters.l1_ratio_candidates,
                            [0.5]
                          )
                        : undefined,
                    cvFolds: Math.max(
                      2,
                      parseInt(modelSourceModule.parameters.cv_folds, 10) || 5
                    ),
                    scoringMetric:
                      modelSourceModule.parameters.scoring_metric || "accuracy",
                  }
                : undefined;

              if (X.length < ordered_feature_columns.length) {
                throw new Error(
                  `Insufficient data: need at least ${ordered_feature_columns.length} samples for ${ordered_feature_columns.length} features, but got ${X.length}.`
                );
              }

              try {
                addLog(
                  "INFO",
                  `Pyodide를 사용하여 Python으로 Logistic Regression 모델 훈련 중...`
                );

                const pyodideModule = await import("./utils/pyodideRunner");
                const { fitLogisticRegressionPython } = pyodideModule;

                const fitResult = await fitLogisticRegressionPython(
                  X,
                  y,
                  penalty,
                  C,
                  solver,
                  maxIter,
                  ordered_feature_columns,
                  60000, // 타임아웃: 60초
                  tuningOptions
                );

                // Logistic Regression은 다중 클래스를 지원하므로 coefficients가 2D 배열일 수 있음
                if (
                  !fitResult.coefficients ||
                  !Array.isArray(fitResult.coefficients)
                ) {
                  throw new Error(
                    `Invalid coefficients: expected array, got ${typeof fitResult.coefficients}.`
                  );
                }

                // 이진 분류인 경우
                if (
                  fitResult.coefficients.length === 1 &&
                  fitResult.coefficients[0].length ===
                    ordered_feature_columns.length
                ) {
                  intercept = fitResult.intercept[0];
                  ordered_feature_columns.forEach((col, idx) => {
                    if (fitResult.coefficients[0][idx] !== undefined) {
                      coefficients[col] = fitResult.coefficients[0][idx];
                    } else {
                      throw new Error(
                        `Missing coefficient for feature ${col} at index ${idx}.`
                      );
                    }
                  });
                } else {
                  // 다중 클래스인 경우 첫 번째 클래스의 계수 사용
                  intercept = fitResult.intercept[0] || 0;
                  ordered_feature_columns.forEach((col, idx) => {
                    if (
                      fitResult.coefficients[0] &&
                      fitResult.coefficients[0][idx] !== undefined
                    ) {
                      coefficients[col] = fitResult.coefficients[0][idx];
                    } else {
                      coefficients[col] = 0;
                    }
                  });
                }

                tuningSummary = fitResult.tuning
                  ? {
                      enabled: Boolean(fitResult.tuning.enabled),
                      strategy: fitResult.tuning.strategy,
                      bestParams: fitResult.tuning.bestParams,
                      bestScore:
                        typeof fitResult.tuning.bestScore === "number"
                          ? fitResult.tuning.bestScore
                          : undefined,
                      scoringMetric: fitResult.tuning.scoringMetric,
                      candidates: Array.isArray(fitResult.tuning.candidates)
                        ? fitResult.tuning.candidates
                        : undefined,
                    }
                  : undefined;
                if (tuningSummary?.enabled && tuningSummary.bestParams) {
                  addLog(
                    "INFO",
                    `Hyperparameter tuning selected params: ${Object.entries(
                      tuningSummary.bestParams
                    )
                      .map(([k, v]) => `${k}=${v}`)
                      .join(", ")}.`
                  );
                }

                // Python에서 계산된 메트릭 사용
                metrics["Accuracy"] = parseFloat(
                  (fitResult.metrics["Accuracy"] || 0).toFixed(4)
                );
                metrics["Precision"] = parseFloat(
                  (fitResult.metrics["Precision"] || 0).toFixed(4)
                );
                metrics["Recall"] = parseFloat(
                  (fitResult.metrics["Recall"] || 0).toFixed(4)
                );
                metrics["F1-Score"] = parseFloat(
                  (fitResult.metrics["F1-Score"] || 0).toFixed(4)
                );
                if (fitResult.metrics["ROC-AUC"] !== undefined) {
                  metrics["ROC-AUC"] = parseFloat(
                    fitResult.metrics["ROC-AUC"].toFixed(4)
                  );
                }

                addLog(
                  "SUCCESS",
                  `Python으로 Logistic Regression 모델 훈련 완료`
                );
              } catch (error: any) {
                const errorMessage = error.message || String(error);
                addLog(
                  "ERROR",
                  `Python LogisticRegression 훈련 실패: ${errorMessage}`
                );
                throw new Error(`모델 훈련 실패: ${errorMessage}`);
              }
            } else if (modelSourceModule.type === ModuleType.KNN) {
              // Pyodide를 사용하여 Python으로 KNN 훈련
              const modelPurpose =
                modelSourceModule.parameters.model_purpose || "classification";
              const nNeighbors =
                parseInt(modelSourceModule.parameters.n_neighbors, 10) || 3;
              const weights =
                modelSourceModule.parameters.weights || "uniform";
              const algorithm =
                modelSourceModule.parameters.algorithm || "auto";
              const metric =
                modelSourceModule.parameters.metric || "minkowski";

              if (X.length < ordered_feature_columns.length) {
                throw new Error(
                  `Insufficient data: need at least ${ordered_feature_columns.length} samples for ${ordered_feature_columns.length} features, but got ${X.length}.`
                );
              }

              try {
                addLog(
                  "INFO",
                  `Pyodide를 사용하여 Python으로 KNN 모델 훈련 중...`
                );

                const pyodideModule = await import("./utils/pyodideRunner");
                const { fitKNNPython } = pyodideModule;

                const fitResult = await fitKNNPython(
                  X,
                  y,
                  modelPurpose,
                  nNeighbors,
                  weights,
                  algorithm,
                  metric,
                  ordered_feature_columns,
                  60000 // 타임아웃: 60초
                );

                // KNN은 coefficients와 intercept가 없으므로 메트릭만 사용
                // coefficients와 intercept는 빈 값으로 설정
                intercept = 0;
                ordered_feature_columns.forEach((col) => {
                  coefficients[col] = 0;
                });

                // Python에서 계산된 메트릭 사용
                if (modelPurpose === "classification") {
                  metrics["Accuracy"] = parseFloat(
                    (fitResult.metrics["Accuracy"] || 0).toFixed(4)
                  );
                  metrics["Precision"] = parseFloat(
                    (fitResult.metrics["Precision"] || 0).toFixed(4)
                  );
                  metrics["Recall"] = parseFloat(
                    (fitResult.metrics["Recall"] || 0).toFixed(4)
                  );
                  metrics["F1-Score"] = parseFloat(
                    (fitResult.metrics["F1-Score"] || 0).toFixed(4)
                  );
                  if (fitResult.metrics["ROC-AUC"] !== undefined) {
                    metrics["ROC-AUC"] = parseFloat(
                      fitResult.metrics["ROC-AUC"].toFixed(4)
                    );
                  }
                } else {
                  metrics["R-squared"] = parseFloat(
                    (fitResult.metrics["R-squared"] || 0).toFixed(4)
                  );
                  metrics["Mean Squared Error"] = parseFloat(
                    (fitResult.metrics["Mean Squared Error"] || 0).toFixed(4)
                  );
                  metrics["Root Mean Squared Error"] = parseFloat(
                    (fitResult.metrics["Root Mean Squared Error"] || 0).toFixed(4)
                  );
                  metrics["Mean Absolute Error"] = parseFloat(
                    (fitResult.metrics["Mean Absolute Error"] || 0).toFixed(4)
                  );
                }

                addLog(
                  "SUCCESS",
                  `Python으로 KNN 모델 훈련 완료`
                );
              } catch (error: any) {
                const errorMessage = error.message || String(error);
                addLog(
                  "ERROR",
                  `Python KNN 훈련 실패: ${errorMessage}`
                );
                throw new Error(`모델 훈련 실패: ${errorMessage}`);
              }
            } else {
              // For other classification models, use simulation for now
              intercept = Math.random() - 0.5;
              ordered_feature_columns.forEach((col) => {
                coefficients[col] = Math.random() * 2 - 1;
              });
              metrics["Accuracy"] = 0.75 + Math.random() * 0.2;
              metrics["Precision"] = 0.7 + Math.random() * 0.25;
              metrics["Recall"] = 0.7 + Math.random() * 0.25;
              metrics["F1-Score"] =
                (2 * (metrics["Precision"] * metrics["Recall"])) /
                (metrics["Precision"] + metrics["Recall"]);
            }
          } else {
            throw new Error(
              `Training simulation for model type '${modelSourceModule.type}' is not implemented, or its 'model_purpose' parameter is not set correctly.`
            );
          }

          // trainedModelOutput이 이미 설정되지 않은 경우에만 기본값으로 생성
          if (!trainedModelOutput) {
            trainedModelOutput = {
              type: "TrainedModelOutput",
              modelType: modelSourceModule.type,
              modelPurpose: modelIsClassification
                ? "classification"
                : "regression",
              coefficients,
              intercept,
              metrics,
              featureColumns: ordered_feature_columns,
              labelColumn: label_column,
              tuningSummary,
            };
          }

          newOutputData = trainedModelOutput;
        } else if (module.type === ModuleType.ScoreModel) {
          const modelInputConnection = connections.find(
            (c) => c.to.moduleId === module.id && c.to.portName === "model_in"
          );
          const dataInputConnection = connections.find(
            (c) => c.to.moduleId === module.id && c.to.portName === "data_in"
          );

          if (!modelInputConnection || !dataInputConnection) {
            throw new Error(
              "Both 'model_in' and 'data_in' ports must be connected."
            );
          }

          const trainedModelSourceModule = currentModules.find(
            (m) => m.id === modelInputConnection.from.moduleId
          );
          if (
            !trainedModelSourceModule ||
            !trainedModelSourceModule.outputData ||
            trainedModelSourceModule.outputData.type !== "TrainedModelOutput"
          ) {
            throw new Error(
              "A successfully trained model must be connected to 'model_in'."
            );
          }

          const dataSourceModule = currentModules.find(
            (m) => m.id === dataInputConnection.from.moduleId
          );
          if (!dataSourceModule || !dataSourceModule.outputData)
            throw new Error("Data source module has no output.");

          let inputData: DataPreview | null = null;
          if (dataSourceModule.outputData.type === "DataPreview") {
            inputData = dataSourceModule.outputData;
          } else if (dataSourceModule.outputData.type === "SplitDataOutput") {
            const portName = dataInputConnection.from.portName;
            if (portName === "train_data_out") {
              inputData = dataSourceModule.outputData.train;
            } else if (portName === "test_data_out") {
              inputData = dataSourceModule.outputData.test;
            }
          }

          if (!inputData)
            throw new Error(
              "Input data for scoring not available or is of the wrong type."
            );

          const trainedModel = trainedModelSourceModule.outputData;
          const modelIsClassification = isClassification(
            trainedModel.modelType,
            trainedModel.modelPurpose
          );
          const labelColumn = trainedModel.labelColumn;

          // KNN 모델의 경우 별도 처리
          if (trainedModel.modelType === ModuleType.KNN) {
            // Train Model 모듈에서 훈련 데이터 가져오기
            const trainModelModule = currentModules.find(
              (m) => m.id === trainedModelSourceModule.id
            );
            
            if (!trainModelModule) {
              throw new Error("Train Model module not found.");
            }

            // Train Model의 입력 데이터 찾기
            const trainDataInputConnection = connections.find(
              (c) =>
                c.to.moduleId === trainModelModule.id &&
                c.to.portName === "data_in"
            );

            if (!trainDataInputConnection) {
              throw new Error("Training data connection not found for KNN model.");
            }

            const trainDataSourceModule = currentModules.find(
              (m) => m.id === trainDataInputConnection.from.moduleId
            );

            if (!trainDataSourceModule || !trainDataSourceModule.outputData) {
              throw new Error("Training data source module not found.");
            }

            let trainingData: DataPreview | null = null;
            if (trainDataSourceModule.outputData.type === "DataPreview") {
              trainingData = trainDataSourceModule.outputData;
            } else if (trainDataSourceModule.outputData.type === "SplitDataOutput") {
              const portName = trainDataInputConnection.from.portName;
              if (portName === "train_data_out") {
                trainingData = trainDataSourceModule.outputData.train;
              } else if (portName === "test_data_out") {
                trainingData = trainDataSourceModule.outputData.test;
              }
            }

            if (!trainingData) {
              throw new Error("Training data not available for KNN model.");
            }

            // KNN 모델 정의 모듈 찾기
            const modelDefConnection = connections.find(
              (c) =>
                c.to.moduleId === trainModelModule.id &&
                c.to.portName === "model_in"
            );

            if (!modelDefConnection) {
              throw new Error("KNN model definition connection not found.");
            }

            const knnModelDefModule = currentModules.find(
              (m) => m.id === modelDefConnection.from.moduleId
            );

            if (!knnModelDefModule) {
              throw new Error("KNN model definition module not found.");
            }

            const modelPurpose =
              knnModelDefModule.parameters.model_purpose || "classification";
            const nNeighbors =
              parseInt(knnModelDefModule.parameters.n_neighbors, 10) || 3;
            const weights = knnModelDefModule.parameters.weights || "uniform";
            const algorithm =
              knnModelDefModule.parameters.algorithm || "auto";
            const metric = knnModelDefModule.parameters.metric || "minkowski";

            try {
              addLog(
                "INFO",
                "Pyodide를 사용하여 Python으로 KNN 모델 예측 수행 중..."
              );

              const pyodideModule = await import("./utils/pyodideRunner");
              const { scoreKNNPython } = pyodideModule;

              const result = await scoreKNNPython(
                inputData.rows || [],
                trainedModel.featureColumns,
                labelColumn,
                modelIsClassification ? "classification" : "regression",
                nNeighbors,
                weights,
                algorithm,
                metric,
                trainingData.rows || [],
                trainedModel.featureColumns,
                labelColumn,
                60000 // 타임아웃: 60초
              );

              newOutputData = {
                type: "DataPreview",
                columns: result.columns,
                totalRowCount: inputData.totalRowCount,
                rows: result.rows,
              };

              addLog("SUCCESS", "Python으로 KNN 모델 예측 완료");
            } catch (error: any) {
              const errorMessage = error.message || String(error);
              addLog("ERROR", `Python KNN ScoreModel 실패: ${errorMessage}`);
              throw new Error(`모델 예측 실패: ${errorMessage}`);
            }
          } else {
            // 기존 방식 (coefficients/intercept 사용)
            // Pyodide를 사용하여 Python으로 예측 수행
            try {
              addLog(
                "INFO",
                "Pyodide를 사용하여 Python으로 모델 예측 수행 중..."
              );

              const pyodideModule = await import("./utils/pyodideRunner");
              const { scoreModelPython } = pyodideModule;

              const result = await scoreModelPython(
                inputData.rows || [],
                trainedModel.featureColumns,
                trainedModel.coefficients,
                trainedModel.intercept,
                labelColumn,
                modelIsClassification ? "classification" : "regression",
                60000 // 타임아웃: 60초
              );

              newOutputData = {
                type: "DataPreview",
                columns: result.columns,
                totalRowCount: inputData.totalRowCount,
                rows: result.rows,
              };

              addLog("SUCCESS", "Python으로 모델 예측 완료");

              // 연결된 Evaluate Model의 파라미터 자동 설정
              const evaluateModelConnections = connections.filter(
                (c) =>
                  c.from.moduleId === module.id &&
                  currentModules.find((m) => m.id === c.to.moduleId)?.type ===
                    ModuleType.EvaluateModel
              );

              for (const evalConn of evaluateModelConnections) {
                const evalModule = currentModules.find(
                  (m) => m.id === evalConn.to.moduleId
                );
                if (evalModule) {
                  const evalParams = evalModule.parameters || {};
                  const updates: Record<string, any> = {};

                  const inputColumns = result.columns.map((c) => c.name);

                  // label_column 자동 설정 (항상 업데이트)
                  if (inputColumns.includes(labelColumn)) {
                    updates.label_column = labelColumn;
                  } else if (inputColumns.length > 0) {
                    updates.label_column = inputColumns[0];
                  }

                  // prediction_column 자동 설정 (항상 업데이트)
                  if (modelIsClassification) {
                    const probaColumn = `${labelColumn}_Predict_Proba_1`;
                    if (inputColumns.includes(probaColumn)) {
                      updates.prediction_column = probaColumn;
                    } else if (inputColumns.includes("Predict")) {
                      updates.prediction_column = "Predict";
                    }
                  } else {
                    if (inputColumns.includes("Predict")) {
                      updates.prediction_column = "Predict";
                    }
                  }

                  // model_type 자동 설정 (항상 업데이트)
                  const detectedModelType = modelIsClassification
                    ? "classification"
                    : "regression";
                  updates.model_type = detectedModelType;

                  // threshold 기본값 설정 (분류 모델인 경우, 값이 없을 때만)
                  // threshold가 이미 설정되어 있으면 절대 변경하지 않음
                  if (
                    modelIsClassification &&
                    (evalParams.threshold === undefined ||
                      evalParams.threshold === null)
                  ) {
                    // threshold가 없을 때만 기본값 설정
                    updates.threshold = 0.5;
                  }
                  // threshold가 이미 설정되어 있으면 updates에 추가하지 않음

                  // 파라미터 업데이트 (threshold는 절대 덮어쓰지 않음)
                  if (Object.keys(updates).length > 0) {
                    setModules(
                      (prev) =>
                        prev.map((m) => {
                          if (m.id === evalModule.id) {
                            // threshold를 제외한 파라미터만 업데이트
                            const finalUpdates = { ...updates };
                            const existingThreshold = m.parameters?.threshold;

                            // threshold가 이미 있으면 절대 덮어쓰지 않음
                            if (
                              existingThreshold !== undefined &&
                              existingThreshold !== null
                            ) {
                              delete finalUpdates.threshold;
                            }

                            // threshold를 제외한 파라미터만 업데이트하고, threshold는 기존 값 유지
                            return {
                              ...m,
                              parameters: {
                                ...m.parameters,
                                ...finalUpdates,
                                // threshold는 기존 값 명시적으로 유지
                                threshold:
                                  existingThreshold !== undefined &&
                                  existingThreshold !== null
                                    ? existingThreshold
                                    : finalUpdates.threshold !== undefined
                                    ? finalUpdates.threshold
                                    : m.parameters?.threshold,
                              },
                            };
                          }
                          return m;
                        }),
                      true
                    );

                    // 파라미터 업데이트만 하고 자동 재실행은 하지 않음
                    // 사용자가 수동으로 실행하거나, Score Model이 완료된 후에 실행되도록 함
                    addLog(
                      "INFO",
                      `Evaluate Model [${evalModule.name}] 파라미터가 자동으로 설정되었습니다. 실행하려면 모듈을 클릭하세요.`
                    );
                  }
                }
              }
            } catch (error: any) {
              const errorMessage = error.message || String(error);
              addLog("ERROR", `Python ScoreModel 실패: ${errorMessage}`);
              throw new Error(`모델 예측 실패: ${errorMessage}`);
            }
          }
        } else if (module.type === ModuleType.EvaluateModel) {
          const inputData = getSingleInputData(module.id) as DataPreview;
          if (!inputData)
            throw new Error("Input data for evaluation not available.");

          // 최신 모듈 상태에서 파라미터 가져오기 (threshold 변경 반영)
          // getCurrentModules()를 통해 항상 최신 상태를 가져옴
          const latestModules = getCurrentModules();
          const latestModule =
            latestModules.find((m) => m.id === module.id) || module;
          let { label_column, prediction_column, model_type, threshold } =
            latestModule.parameters;

          // threshold가 설정되어 있으면 로그 출력 (디버깅용)
          if (threshold !== undefined && threshold !== null) {
            addLog(
              "INFO",
              `Evaluate Model [${module.name}] 실행 시 threshold: ${threshold} (최신 상태에서 가져옴)`
            );
          } else {
            addLog(
              "INFO",
              `Evaluate Model [${module.name}] threshold가 설정되지 않음`
            );
          }

          // 연결된 Train Model을 찾아서 modelPurpose를 자동으로 감지 및 기본값 설정
          let detectedModelType: "classification" | "regression" =
            model_type === "regression" ? "regression" : "classification";
          let trainModelLabelColumn: string | null = null;

          // Evaluate Model의 입력 연결 찾기 (보통 Score Model)
          const inputConnection = connections.find(
            (c) => c.to.moduleId === module.id
          );
          if (inputConnection) {
            const sourceModule = currentModules.find(
              (m) => m.id === inputConnection.from.moduleId
            );

            // Score Model인 경우, 그 Score Model이 연결된 Train Model 찾기
            if (sourceModule?.type === ModuleType.ScoreModel) {
              const modelInputConnection = connections.find(
                (c) =>
                  c.to.moduleId === sourceModule.id &&
                  c.to.portName === "model_in"
              );
              if (modelInputConnection) {
                const trainModelModule = currentModules.find(
                  (m) =>
                    m.id === modelInputConnection.from.moduleId &&
                    m.outputData?.type === "TrainedModelOutput"
                );
                if (
                  trainModelModule?.outputData?.type === "TrainedModelOutput"
                ) {
                  const trainedModel = trainModelModule.outputData;
                  trainModelLabelColumn = trainedModel.labelColumn;

                  // modelPurpose가 있으면 사용, 없으면 modelType으로 추론
                  if (trainedModel.modelPurpose) {
                    detectedModelType = trainedModel.modelPurpose;
                    addLog(
                      "INFO",
                      `연결된 모델 타입 자동 감지: ${detectedModelType} (${trainModelModule.name})`
                    );
                  } else {
                    // modelType으로 분류 모델인지 확인
                    const isClassModel = isClassification(
                      trainedModel.modelType,
                      trainedModel.modelPurpose
                    );
                    detectedModelType = isClassModel
                      ? "classification"
                      : "regression";
                    addLog(
                      "INFO",
                      `모델 타입 자동 감지: ${detectedModelType} (${trainModelModule.name})`
                    );
                  }
                }
              }
            }
          }

          // 자동 기본값 설정
          const inputColumns = inputData.columns.map((c) => c.name);
          const paramUpdates: Record<string, any> = {};

          // label_column 자동 설정
          if (!label_column) {
            if (
              trainModelLabelColumn &&
              inputColumns.includes(trainModelLabelColumn)
            ) {
              label_column = trainModelLabelColumn;
              paramUpdates.label_column = label_column;
              addLog("INFO", `Label column 자동 설정: ${label_column}`);
            } else if (inputColumns.length > 0) {
              label_column = inputColumns[0];
              paramUpdates.label_column = label_column;
              addLog("INFO", `Label column 자동 설정: ${label_column}`);
            }
          }

          // prediction_column 자동 설정
          if (!prediction_column) {
            if (
              detectedModelType === "classification" &&
              trainModelLabelColumn
            ) {
              // 분류 모델: {label_column}_Predict_Proba_1 찾기
              const probaColumn = `${trainModelLabelColumn}_Predict_Proba_1`;
              if (inputColumns.includes(probaColumn)) {
                prediction_column = probaColumn;
                paramUpdates.prediction_column = prediction_column;
                addLog(
                  "INFO",
                  `Prediction column 자동 설정: ${prediction_column} (확률값)`
                );
              } else if (inputColumns.includes("Predict")) {
                prediction_column = "Predict";
                paramUpdates.prediction_column = prediction_column;
                addLog(
                  "INFO",
                  `Prediction column 자동 설정: ${prediction_column}`
                );
              }
            } else {
              // 회귀 모델: Predict 사용
              if (inputColumns.includes("Predict")) {
                prediction_column = "Predict";
                paramUpdates.prediction_column = prediction_column;
                addLog(
                  "INFO",
                  `Prediction column 자동 설정: ${prediction_column}`
                );
              }
            }
          }

          // model_type 자동 설정
          if (model_type !== detectedModelType) {
            paramUpdates.model_type = detectedModelType;
          }

          // threshold 기본값 설정 (분류 모델인 경우, 값이 없을 때만)
          // threshold가 이미 설정되어 있으면 절대 덮어쓰지 않음
          if (detectedModelType === "classification") {
            if (threshold === undefined || threshold === null) {
              // threshold가 없을 때만 기본값 설정
              threshold = 0.5;
              // paramUpdates에 추가하지 않음 (사용자가 변경한 값이 있을 수 있으므로)
              // 대신 threshold 변수만 업데이트하여 평가에 사용
              addLog(
                "INFO",
                `Evaluate Model [${module.name}] threshold 기본값 사용: ${threshold} (파라미터에는 저장하지 않음)`
              );
            } else {
              // threshold가 이미 설정되어 있으면 그 값을 사용
              addLog(
                "INFO",
                `Evaluate Model [${module.name}] threshold 사용: ${threshold}`
              );
            }
          }

          // 자동으로 설정한 파라미터들을 모듈에 저장
          // threshold는 절대 paramUpdates에 추가하지 않음 (사용자가 변경한 값 유지)
          if (Object.keys(paramUpdates).length > 0) {
            setModules(
              (prev) =>
                prev.map((m) => {
                  if (m.id === module.id) {
                    // threshold를 제외한 파라미터만 업데이트
                    const finalParamUpdates = { ...paramUpdates };
                    // threshold가 paramUpdates에 있으면 제거
                    delete finalParamUpdates.threshold;

                    // 기존 threshold 값 확인 (절대 변경하지 않음)
                    const existingThreshold = m.parameters?.threshold;

                    // threshold는 기존 값을 명시적으로 유지 (절대 변경하지 않음)
                    const updatedParameters = {
                      ...m.parameters,
                      ...finalParamUpdates,
                      // threshold는 기존 값 유지 (변경하지 않음)
                      threshold:
                        existingThreshold !== undefined &&
                        existingThreshold !== null
                          ? existingThreshold
                          : threshold !== undefined && threshold !== null
                          ? threshold
                          : 0.5,
                    };

                    addLog(
                      "INFO",
                      `Evaluate Model [${module.name}] 파라미터 업데이트 후 threshold: ${updatedParameters.threshold} (기존: ${existingThreshold})`
                    );

                    return { ...m, parameters: updatedParameters };
                  }
                  return m;
                }),
              true
            );
          }

          if (!label_column || !prediction_column) {
            throw new Error(
              "Label and prediction columns must be configured for evaluation."
            );
          }

          const rows = inputData.rows || [];
          if (rows.length === 0)
            throw new Error("No rows in input data to evaluate.");

          // Pyodide를 사용하여 Python으로 평가 메트릭 계산
          try {
            addLog(
              "INFO",
              "Pyodide를 사용하여 Python으로 모델 평가 수행 중..."
            );

            const pyodideModule = await import("./utils/pyodideRunner");
            const { evaluateModelPython } = pyodideModule;

            // 분류 모델인 경우 여러 threshold에 대한 precision/recall도 계산
            const calculateThresholdMetrics =
              detectedModelType === "classification";

            const result = await evaluateModelPython(
              rows,
              label_column,
              prediction_column,
              detectedModelType, // 자동 감지된 모델 타입 사용
              threshold, // threshold 전달 (분류 모델인 경우)
              120000, // 타임아웃: 120초 (여러 threshold 계산 시 시간이 더 걸림)
              calculateThresholdMetrics // 여러 threshold에 대한 precision/recall 계산
            );

            const { thresholdMetrics, ...metrics } = result;

            addLog("SUCCESS", "Python으로 모델 평가 완료");

            // 혼동행렬 추출
            const confusionMatrix =
              detectedModelType === "classification" &&
              typeof metrics["TP"] === "number" &&
              typeof metrics["FP"] === "number" &&
              typeof metrics["TN"] === "number" &&
              typeof metrics["FN"] === "number"
                ? {
                    tp: metrics["TP"] as number,
                    fp: metrics["FP"] as number,
                    tn: metrics["TN"] as number,
                    fn: metrics["FN"] as number,
                  }
                : undefined;

            newOutputData = {
              type: "EvaluationOutput",
              modelType: detectedModelType, // 자동 감지된 모델 타입 사용
              metrics,
              confusionMatrix,
              threshold:
                detectedModelType === "classification" ? threshold : undefined,
              thresholdMetrics: thresholdMetrics, // 여러 threshold에 대한 precision/recall
            };
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            addLog("ERROR", `Python EvaluateModel 실패: ${errorMessage}`);
            throw new Error(`모델 평가 실패: ${errorMessage}`);
          }
        } else if (module.type === ModuleType.OLSModel) {
          newOutputData = {
            type: "ModelDefinitionOutput",
            modelFamily: "statsmodels",
            modelType: "OLS",
            parameters: {},
          };
          addLog(
            "INFO",
            `모델 정의 모듈 '${module.name}' (OLS)이 생성되었습니다.`
          );
        } else if (module.type === ModuleType.LogisticModel) {
          newOutputData = {
            type: "ModelDefinitionOutput",
            modelFamily: "statsmodels",
            modelType: "Logit",
            parameters: {},
          };
          addLog(
            "INFO",
            `모델 정의 모듈 '${module.name}' (Logistic)이 생성되었습니다.`
          );
        } else if (module.type === ModuleType.PoissonModel) {
          newOutputData = {
            type: "ModelDefinitionOutput",
            modelFamily: "statsmodels",
            modelType: "Poisson",
            parameters: {
              max_iter: module.parameters.max_iter || 100,
            },
          };
          addLog(
            "INFO",
            `모델 정의 모듈 '${module.name}' (Poisson)이 생성되었습니다.`
          );
        } else if (module.type === ModuleType.QuasiPoissonModel) {
          newOutputData = {
            type: "ModelDefinitionOutput",
            modelFamily: "statsmodels",
            modelType: "QuasiPoisson",
            parameters: {
              max_iter: module.parameters.max_iter || 100,
            },
          };
          addLog(
            "INFO",
            `모델 정의 모듈 '${module.name}' (Quasi-Poisson)이 생성되었습니다.`
          );
        } else if (module.type === ModuleType.NegativeBinomialModel) {
          newOutputData = {
            type: "ModelDefinitionOutput",
            modelFamily: "statsmodels",
            modelType: "NegativeBinomial",
            parameters: {
              max_iter: module.parameters.max_iter || 100,
              disp: module.parameters.disp || 1.0,
            },
          };
          addLog(
            "INFO",
            `모델 정의 모듈 '${module.name}' (Negative Binomial)이 생성되었습니다.`
          );
        } else if (module.type === ModuleType.StatModels) {
          newOutputData = {
            type: "ModelDefinitionOutput",
            modelFamily: "statsmodels",
            modelType: module.parameters.model,
            parameters: {},
          };
          addLog(
            "INFO",
            `모델 정의 모듈 '${module.name}' (${module.parameters.model})이 생성되었습니다.`
          );
        } else if (module.type === ModuleType.ResultModel) {
          const modelInputConnection = connections.find(
            (c) => c.to.moduleId === module.id && c.to.portName === "model_in"
          );
          const dataInputConnection = connections.find(
            (c) => c.to.moduleId === module.id && c.to.portName === "data_in"
          );
          if (!modelInputConnection || !dataInputConnection)
            throw new Error(
              "Both 'model_in' and 'data_in' ports must be connected."
            );

          const modelSourceModule = currentModules.find(
            (m) => m.id === modelInputConnection.from.moduleId
          );
          if (!modelSourceModule)
            throw new Error("Model source module not found.");

          // 모델 정의 모듈이 output이 없으면 자동으로 생성
          if (
            MODEL_DEFINITION_TYPES.includes(modelSourceModule.type) &&
            !modelSourceModule.outputData
          ) {
            // 모델 정의 output 자동 생성
            let modelType: string;
            let parameters: Record<string, any> = {};

            if (modelSourceModule.type === ModuleType.OLSModel) {
              modelType = "OLS";
            } else if (modelSourceModule.type === ModuleType.LogisticModel) {
              modelType = "Logit";
            } else if (modelSourceModule.type === ModuleType.PoissonModel) {
              modelType = "Poisson";
              parameters = {
                max_iter: modelSourceModule.parameters.max_iter || 100,
              };
            } else if (
              modelSourceModule.type === ModuleType.QuasiPoissonModel
            ) {
              modelType = "QuasiPoisson";
              parameters = {
                max_iter: modelSourceModule.parameters.max_iter || 100,
              };
            } else if (
              modelSourceModule.type === ModuleType.NegativeBinomialModel
            ) {
              modelType = "NegativeBinomial";
              parameters = {
                max_iter: modelSourceModule.parameters.max_iter || 100,
                disp: modelSourceModule.parameters.disp || 1.0,
              };
            } else if (modelSourceModule.type === ModuleType.StatModels) {
              modelType = modelSourceModule.parameters.model || "Gamma";
            } else {
              throw new Error(
                `Unsupported model definition type: ${modelSourceModule.type}`
              );
            }

            // 모델 정의 모듈의 output 생성
            const modelDefinitionOutput = {
              type: "ModelDefinitionOutput" as const,
              modelFamily: "statsmodels" as const,
              modelType: modelType as any,
              parameters,
            };

            // 현재 모듈 목록 업데이트
            currentModules = currentModules.map((m) =>
              m.id === modelSourceModule.id
                ? {
                    ...m,
                    outputData: modelDefinitionOutput,
                    status: ModuleStatus.Success,
                  }
                : m
            );

            // 상태 업데이트
            setModules((prevModules) =>
              prevModules.map((m) =>
                m.id === modelSourceModule.id
                  ? {
                      ...m,
                      outputData: modelDefinitionOutput,
                      status: ModuleStatus.Success,
                    }
                  : m
              )
            );

            addLog(
              "INFO",
              `모델 정의 모듈 '${modelSourceModule.name}'의 output이 자동 생성되었습니다.`
            );
          }

          // 업데이트된 모듈에서 다시 찾기
          const updatedModelSourceModule = currentModules.find(
            (m) => m.id === modelInputConnection.from.moduleId
          );

          if (
            !updatedModelSourceModule ||
            updatedModelSourceModule.outputData?.type !==
              "ModelDefinitionOutput"
          )
            throw new Error("A Stat Models module must be connected.");

          const modelDefinition = updatedModelSourceModule.outputData;
          if (modelDefinition.modelFamily !== "statsmodels")
            throw new Error("Connected model is not a statsmodels type.");

          const dataSourceModule = currentModules.find(
            (m) => m.id === dataInputConnection.from.moduleId
          );
          if (!dataSourceModule || !dataSourceModule.outputData)
            throw new Error("Data source module has no output.");

          let inputData: DataPreview | null = null;
          if (dataSourceModule.outputData.type === "DataPreview")
            inputData = dataSourceModule.outputData;
          else if (dataSourceModule.outputData.type === "SplitDataOutput") {
            const portName = dataInputConnection.from.portName;
            inputData =
              portName === "train_data_out"
                ? dataSourceModule.outputData.train
                : dataSourceModule.outputData.test;
          }
          if (!inputData) throw new Error("Input data not available.");

          const { feature_columns, label_column } = module.parameters;
          if (!feature_columns || feature_columns.length === 0 || !label_column)
            throw new Error("Feature and label columns must be configured.");

          const ordered_feature_columns = inputData.columns
            .map((c) => c.name)
            .filter((name) => feature_columns.includes(name));

          const modelType = modelDefinition.modelType;
          const modelParams = modelDefinition.parameters || {};

          // 모든 모델 타입을 Python으로 실행
          try {
            addLog(
              "INFO",
              `Pyodide를 사용하여 Python으로 ${modelType} 모델 피팅 중 (statsmodels)...`
            );

            // 데이터 검증
            if (!inputData.rows || inputData.rows.length === 0) {
              throw new Error("입력 데이터가 비어있습니다.");
            }
            if (ordered_feature_columns.length === 0) {
              throw new Error("특성 컬럼이 선택되지 않았습니다.");
            }

            const X = (inputData.rows || []).map((row) =>
              ordered_feature_columns.map((col) => {
                const val = row[col];
                if (typeof val !== "number" || isNaN(val)) {
                  addLog(
                    "WARNING",
                    `컬럼 '${col}'의 값이 숫자가 아니거나 NaN입니다. 0으로 대체합니다.`
                  );
                  return 0;
                }
                return val;
              })
            );
            const y = (inputData.rows || []).map((row) => {
              const val = row[label_column];
              if (typeof val !== "number" || isNaN(val)) {
                addLog(
                  "WARNING",
                  `레이블 컬럼 '${label_column}'의 값이 숫자가 아니거나 NaN입니다. 0으로 대체합니다.`
                );
                return 0;
              }
              return val;
            });

            // 데이터 크기 검증
            if (X.length === 0 || y.length === 0) {
              throw new Error("데이터가 비어있습니다.");
            }
            if (X.length !== y.length) {
              throw new Error(
                `X와 y의 길이가 일치하지 않습니다: X=${X.length}, y=${y.length}`
              );
            }
            if (X[0].length === 0) {
              throw new Error("특성 데이터가 비어있습니다.");
            }

            addLog(
              "INFO",
              `데이터 준비 완료: ${X.length}개 샘플, ${X[0].length}개 특성`
            );

            const pyodideModule = await import("./utils/pyodideRunner");
            const { runStatsModel } = pyodideModule;

            // 모델 파라미터 추출
            const maxIter = modelParams.max_iter || 100;
            const disp = modelParams.disp || 1.0;

            const fitResult = await runStatsModel(
              X,
              y,
              modelType === "Logit" ? "Logistic" : modelType,
              ordered_feature_columns,
              60000, // 타임아웃: 60초
              maxIter,
              disp
            );

            // 결과 변환
            const summaryCoefficients: StatsModelsResultOutput["summary"]["coefficients"] =
              {};
            Object.entries(fitResult.summary.coefficients).forEach(
              ([paramName, coefData]) => {
                summaryCoefficients[paramName] = {
                  coef: coefData.coef,
                  "std err": coefData["std err"],
                  t: coefData.t,
                  z: coefData.z,
                  "P>|t|": coefData["P>|t|"],
                  "P>|z|": coefData["P>|z|"],
                  "[0.025": coefData["[0.025"],
                  "0.975]": coefData["0.975]"],
                };
              }
            );

            const metrics: StatsModelsResultOutput["summary"]["metrics"] = {};
            Object.entries(fitResult.summary.metrics).forEach(
              ([key, value]) => {
                if (typeof value === "number") {
                  metrics[key] = value.toFixed(6);
                } else {
                  metrics[key] = value;
                }
              }
            );

            newOutputData = {
              type: "StatsModelsResultOutput",
              modelType: modelDefinition.modelType,
              summary: { coefficients: summaryCoefficients, metrics },
              featureColumns: ordered_feature_columns,
              labelColumn: label_column,
            };

            addLog(
              "SUCCESS",
              `Python으로 ${modelType} 모델 피팅 완료 (statsmodels)`
            );
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            addLog(
              "ERROR",
              `Python ${modelType} 모델 피팅 실패: ${errorMessage}`
            );
            throw new Error(`모델 피팅 실패: ${errorMessage}`);
          }
        } else if (module.type === ModuleType.PredictModel) {
          const modelInputConnection = connections.find(
            (c) => c.to.moduleId === module.id && c.to.portName === "model_in"
          );
          const dataInputConnection = connections.find(
            (c) => c.to.moduleId === module.id && c.to.portName === "data_in"
          );

          if (!modelInputConnection || !dataInputConnection) {
            throw new Error(
              "Both 'model_in' and 'data_in' ports must be connected."
            );
          }

          const modelSourceModule = currentModules.find(
            (m) => m.id === modelInputConnection.from.moduleId
          );
          if (
            !modelSourceModule ||
            !modelSourceModule.outputData ||
            modelSourceModule.outputData.type !== "StatsModelsResultOutput"
          ) {
            throw new Error(
              "A successful Result Model module must be connected to 'model_in'."
            );
          }
          const modelOutput = modelSourceModule.outputData;

          const dataSourceModule = currentModules.find(
            (m) => m.id === dataInputConnection.from.moduleId
          );
          if (!dataSourceModule || !dataSourceModule.outputData)
            throw new Error("Data source module has no output.");

          let inputData: DataPreview | null = null;
          if (dataSourceModule.outputData.type === "DataPreview") {
            inputData = dataSourceModule.outputData;
          } else if (dataSourceModule.outputData.type === "SplitDataOutput") {
            const portName = dataInputConnection.from.portName;
            if (portName === "train_data_out") {
              inputData = dataSourceModule.outputData.train;
            } else if (portName === "test_data_out") {
              inputData = dataSourceModule.outputData.test;
            }
          }

          if (!inputData)
            throw new Error(
              "Input data for prediction not available or is of the wrong type."
            );

          const labelColumn = modelOutput.labelColumn;
          const predictColName = "y_pred_prob";

          // Logistic, Poisson, Quasi-Poisson, Negative Binomial 모델의 경우 y_pred 열 추가
          const needsRoundedColumn = [
            "Logit",
            "Logistic",
            "Poisson",
            "QuasiPoisson",
            "NegativeBinomial",
          ].includes(modelOutput.modelType);
          const roundedColName = "y_pred";

          const newColumns: ColumnInfo[] = [...inputData.columns];
          if (!newColumns.some((c) => c.name === predictColName)) {
            newColumns.push({ name: predictColName, type: "number" });
          }
          if (
            needsRoundedColumn &&
            !newColumns.some((c) => c.name === roundedColName)
          ) {
            newColumns.push({ name: roundedColName, type: "number" });
          }
          const inputRows = inputData.rows || [];

          const newRows = inputRows.map((row) => {
            let linearPredictor =
              modelOutput.summary.coefficients["const"]?.coef ?? 0;

            for (const feature of modelOutput.featureColumns) {
              const featureValue = row[feature] as number;
              const coefficient =
                modelOutput.summary.coefficients[feature]?.coef;
              if (
                typeof featureValue === "number" &&
                typeof coefficient === "number"
              ) {
                linearPredictor += featureValue * coefficient;
              }
            }

            let prediction: number;
            switch (modelOutput.modelType) {
              case "OLS":
                prediction = linearPredictor;
                break;
              case "Logit":
              case "Logistic":
                prediction = sigmoid(linearPredictor);
                break;
              case "Poisson":
              case "QuasiPoisson":
              case "NegativeBinomial":
              case "Gamma":
              case "Tweedie":
                prediction = Math.exp(linearPredictor);
                break;
              default:
                prediction = NaN;
            }

            const resultRow: Record<string, any> = {
              ...row,
              [predictColName]: parseFloat(prediction.toFixed(4)),
            };

            // Logistic, Poisson, Quasi-Poisson, Negative Binomial 모델의 경우 반올림된 정수 추가
            if (needsRoundedColumn) {
              resultRow[roundedColName] = Math.round(prediction);
            }

            return resultRow;
          });

          newOutputData = {
            type: "DataPreview",
            columns: newColumns,
            totalRowCount: inputData.totalRowCount,
            rows: newRows,
          };
        } else if (module.type === ModuleType.DiversionChecker) {
          const dataInputConnection = connections.find(
            (c) => c.to.moduleId === module.id && c.to.portName === "data_in"
          );

          if (!dataInputConnection) {
            throw new Error("'data_in' port must be connected.");
          }

          const dataSourceModule = currentModules.find(
            (m) => m.id === dataInputConnection.from.moduleId
          );
          if (!dataSourceModule || !dataSourceModule.outputData)
            throw new Error("Data source module has no output.");

          let inputData: DataPreview | null = null;
          if (dataSourceModule.outputData.type === "DataPreview") {
            inputData = dataSourceModule.outputData;
          } else if (dataSourceModule.outputData.type === "SplitDataOutput") {
            const portName = dataInputConnection.from.portName;
            inputData =
              portName === "train_data_out"
                ? dataSourceModule.outputData.train
                : dataSourceModule.outputData.test;
          }
          if (!inputData) throw new Error("Input data not available.");

          const { feature_columns, label_column, max_iter } = module.parameters;
          if (!feature_columns || feature_columns.length === 0 || !label_column)
            throw new Error("Feature and label columns must be configured.");

          try {
            addLog(
              "INFO",
              "Pyodide를 사용하여 Python으로 Diversion Checker 실행 중..."
            );

            const ordered_feature_columns = inputData.columns
              .map((c) => c.name)
              .filter((name) => feature_columns.includes(name));

            const X = (inputData.rows || []).map((row) =>
              ordered_feature_columns.map((col) => {
                const val = row[col];
                return typeof val === "number" ? val : 0;
              })
            );
            const y = (inputData.rows || []).map((row) => {
              const val = row[label_column];
              return typeof val === "number" ? val : 0;
            });

            // 데이터 검증
            if (X.length === 0 || y.length === 0) {
              throw new Error(
                "입력 데이터가 비어있습니다. 데이터 소스를 확인해주세요."
              );
            }
            if (X[0].length !== ordered_feature_columns.length) {
              throw new Error("특성 컬럼 수가 데이터와 일치하지 않습니다.");
            }
            if (X.length !== y.length) {
              throw new Error(
                "특성 데이터와 레이블 데이터의 행 수가 일치하지 않습니다."
              );
            }

            const pyodideModule = await import("./utils/pyodideRunner");
            const { runDiversionChecker } = pyodideModule;

            const result = await runDiversionChecker(
              X,
              y,
              ordered_feature_columns,
              label_column,
              max_iter || 100,
              120000 // 타임아웃: 120초
            );

            // runDiversionChecker는 Python 결과를 그대로 반환하므로 snake_case를 사용
            const pythonResults = result.results as any;
            newOutputData = {
              type: "DiversionCheckerOutput",
              phi: result.phi,
              recommendation: result.recommendation,
              poissonAic: result.poissonAic,
              negativeBinomialAic: result.negativeBinomialAic,
              aicComparison: result.aicComparison,
              cameronTrivediCoef: result.cameronTrivediCoef,
              cameronTrivediPvalue: result.cameronTrivediPvalue,
              cameronTrivediConclusion: result.cameronTrivediConclusion,
              methodsUsed: result.methodsUsed,
              results: {
                phi: pythonResults.phi,
                phi_interpretation:
                  pythonResults.phi_interpretation ||
                  `φ = ${result.phi.toFixed(6)}`,
                recommendation: pythonResults.recommendation,
                poisson_aic: pythonResults.poisson_aic ?? null,
                negative_binomial_aic:
                  pythonResults.negative_binomial_aic ?? null,
                cameron_trivedi_coef: pythonResults.cameron_trivedi_coef,
                cameron_trivedi_pvalue: pythonResults.cameron_trivedi_pvalue,
                cameron_trivedi_conclusion:
                  pythonResults.cameron_trivedi_conclusion,
              },
            } as DiversionCheckerOutput;

            addLog("SUCCESS", "Diversion Checker 실행 완료");
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            addLog("ERROR", `Diversion Checker 실행 실패: ${errorMessage}`);
            throw new Error(`Diversion Checker 실행 실패: ${errorMessage}`);
          }
        } else if (module.type === ModuleType.EvaluateStat) {
          const dataInputConnection = connections.find(
            (c) => c.to.moduleId === module.id && c.to.portName === "data_in"
          );

          if (!dataInputConnection) {
            throw new Error("'data_in' port must be connected.");
          }

          const dataSourceModule = currentModules.find(
            (m) => m.id === dataInputConnection.from.moduleId
          );
          if (!dataSourceModule || !dataSourceModule.outputData)
            throw new Error("Data source module has no output.");

          let inputData: DataPreview | null = null;
          if (dataSourceModule.outputData.type === "DataPreview") {
            inputData = dataSourceModule.outputData;
          } else if (dataSourceModule.outputData.type === "SplitDataOutput") {
            const portName = dataInputConnection.from.portName;
            inputData =
              portName === "train_data_out"
                ? dataSourceModule.outputData.train
                : dataSourceModule.outputData.test;
          }
          if (!inputData) throw new Error("Input data not available.");

          const { label_column, prediction_column, model_type } =
            module.parameters;
          if (!label_column || !prediction_column) {
            throw new Error(
              "Label column and prediction column must be configured."
            );
          }

          // 예측 컬럼이 데이터에 있는지 확인
          if (!inputData.columns.some((c) => c.name === prediction_column)) {
            throw new Error(
              `Prediction column '${prediction_column}' not found in data.`
            );
          }
          if (!inputData.columns.some((c) => c.name === label_column)) {
            throw new Error(
              `Label column '${label_column}' not found in data.`
            );
          }

          try {
            addLog(
              "INFO",
              "Pyodide를 사용하여 Python으로 Evaluate Stat 실행 중..."
            );

            const pyodideModule = await import("./utils/pyodideRunner");
            const { evaluateStatsPython } = pyodideModule;

            const result = await evaluateStatsPython(
              inputData.rows || [],
              label_column,
              prediction_column,
              model_type || "",
              120000 // 타임아웃: 120초
            );

            newOutputData = {
              type: "EvaluateStatOutput",
              modelType: model_type,
              metrics: result.metrics,
              residuals: result.residuals,
              deviance: result.deviance,
              pearsonChi2: result.pearsonChi2,
              dispersion: result.dispersion,
              aic: result.aic,
              bic: result.bic,
              logLikelihood: result.logLikelihood,
            } as EvaluateStatOutput;

            addLog("SUCCESS", "Evaluate Stat 실행 완료");
          } catch (error: any) {
            const errorMessage = error.message || String(error);
            addLog("ERROR", `Evaluate Stat 실행 실패: ${errorMessage}`);
            throw new Error(`Evaluate Stat 실행 실패: ${errorMessage}`);
          }
        } else if (
          module.type === ModuleType.KMeans ||
          module.type === ModuleType.HierarchicalClustering
        ) {
          const inputData = getSingleInputData(module.id) as DataPreview;
          if (!inputData) throw new Error("Input data not available.");

          const { n_clusters, feature_columns } = module.parameters;
          if (
            !feature_columns ||
            !Array.isArray(feature_columns) ||
            feature_columns.length === 0
          )
            throw new Error("Feature columns must be selected for clustering.");

          const newRows = (inputData.rows || []).map((row) => ({
            ...row,
            cluster: Math.floor(Math.random() * n_clusters),
          }));
          const newColumns = [
            ...inputData.columns,
            { name: "cluster", type: "number" },
          ];

          const clusterAssignments: DataPreview = {
            ...inputData,
            columns: newColumns,
            rows: newRows,
          };

          if (module.type === ModuleType.KMeans) {
            const centroids = Array.from({ length: n_clusters }, (_, i) => {
              const centroid: Record<string, number> = {};
              feature_columns.forEach((col: string) => {
                centroid[col] = Math.random() * 100;
              });
              return centroid;
            });
            newOutputData = {
              type: "KMeansOutput",
              clusterAssignments,
              centroids,
              model: {},
            } as KMeansOutput;
          } else {
            newOutputData = {
              type: "HierarchicalClusteringOutput",
              clusterAssignments,
            } as HierarchicalClusteringOutput;
          }
        } else if (module.type === ModuleType.DBSCAN) {
          const inputData = getSingleInputData(module.id) as DataPreview;
          if (!inputData) throw new Error("Input data not available.");
          const { feature_columns } = module.parameters;
          if (
            !feature_columns ||
            !Array.isArray(feature_columns) ||
            feature_columns.length === 0
          )
            throw new Error("Feature columns must be selected.");

          const n_clusters = Math.floor(Math.random() * 4) + 2; // Random clusters 2-5
          let n_noise = 0;

          const newRows = (inputData.rows || []).map((row) => {
            // 10% chance of being noise
            const isNoise = Math.random() < 0.1;
            let cluster = -1;
            if (isNoise) {
              n_noise++;
            } else {
              cluster = Math.floor(Math.random() * n_clusters);
            }
            return { ...row, cluster };
          });

          const newColumns = [
            ...inputData.columns,
            { name: "cluster", type: "number" },
          ];
          const clusterAssignments: DataPreview = {
            ...inputData,
            columns: newColumns,
            rows: newRows,
          };
          newOutputData = {
            type: "DBSCANOutput",
            clusterAssignments,
            n_clusters,
            n_noise,
          } as DBSCANOutput;
        } else if (module.type === ModuleType.PrincipalComponentAnalysis) {
          const inputData = getSingleInputData(module.id) as DataPreview;
          if (!inputData) throw new Error("Input data not available.");
          const { n_components, feature_columns } = module.parameters;
          let columnsToProcess = feature_columns;
          if (
            !columnsToProcess ||
            !Array.isArray(columnsToProcess) ||
            columnsToProcess.length === 0
          ) {
            columnsToProcess = inputData.columns
              .filter((c) => c.type === "number")
              .map((c) => c.name);
          }
          if (columnsToProcess.length < n_components)
            throw new Error(
              "Number of components cannot be greater than number of features."
            );

          const newColumns: ColumnInfo[] = Array.from(
            { length: n_components },
            (_, i) => ({
              name: `PC${i + 1}`,
              type: "number",
            })
          );

          const newRows = (inputData.rows || []).map((row) => {
            const newRow: Record<string, number> = {};
            for (let i = 0; i < n_components; i++) {
              newRow[`PC${i + 1}`] = Math.random() * 10 - 5;
            }
            return newRow;
          });

          // Mock explained variance
          let remainingVariance = 1;
          const explainedVarianceRatio = Array.from(
            { length: n_components },
            (_, i) => {
              const explained =
                Math.random() * (remainingVariance / 2) + (i === 0 ? 0.4 : 0);
              remainingVariance -= explained;
              return explained;
            }
          );

          const transformedData: DataPreview = {
            type: "DataPreview",
            columns: newColumns,
            totalRowCount: inputData.totalRowCount,
            rows: newRows,
          };
          newOutputData = {
            type: "PCAOutput",
            transformedData,
            explainedVarianceRatio,
          } as PCAOutput;
        } else if (module.type === ModuleType.FitLossDistribution) {
          const inputData = getSingleInputData(module.id) as DataPreview;
          if (!inputData) throw new Error("Input data not available.");

          const { loss_column, distribution_type } = module.parameters;
          if (!loss_column)
            throw new Error("Loss column parameter is not set.");
          if (
            !inputData.columns.some(
              (c) => c.name === loss_column && c.type === "number"
            )
          ) {
            throw new Error(
              `Selected loss column '${loss_column}' is not a numeric column in the input data.`
            );
          }

          // Mock fitting process
          let params: Record<string, number> = {};
          if (distribution_type === "Pareto") {
            params = {
              alpha: 1.5 + Math.random() * 0.5,
              x_m: 100000 + Math.random() * 50000,
            };
          } else if (distribution_type === "Lognormal") {
            params = {
              mu: 12 + Math.random(),
              sigma: 1.2 + Math.random() * 0.5,
            };
          }

          newOutputData = {
            type: "FittedDistributionOutput",
            distributionType: distribution_type,
            parameters: params,
            lossColumn: loss_column,
          };
        } else if (module.type === ModuleType.GenerateExposureCurve) {
          const distInputConnection = connections.find(
            (c) => c.to.moduleId === module.id && c.to.portName === "dist_in"
          );
          const dataInputConnection = connections.find(
            (c) => c.to.moduleId === module.id && c.to.portName === "data_in"
          );
          if (!distInputConnection || !dataInputConnection)
            throw new Error(
              "Both 'dist_in' and 'data_in' ports must be connected."
            );

          const distSourceModule = currentModules.find(
            (m) => m.id === distInputConnection.from.moduleId
          );
          if (
            !distSourceModule ||
            distSourceModule.outputData?.type !== "FittedDistributionOutput"
          )
            throw new Error("A fitted distribution must be connected.");
          const distOutput = distSourceModule.outputData;

          const dataSourceModule = currentModules.find(
            (m) => m.id === dataInputConnection.from.moduleId
          );
          let inputData: DataPreview | null = null;
          if (dataSourceModule?.outputData?.type === "DataPreview")
            inputData = dataSourceModule.outputData;
          if (!inputData)
            throw new Error("Input data not available from 'data_in'.");

          const lossColumn = distOutput.lossColumn;
          const totalExpectedLoss = (inputData.rows || []).reduce(
            (sum, row) => sum + (Number(row[lossColumn]) || 0),
            0
          );

          // Mock curve generation
          const curve: ExposureCurveOutput["curve"] = [];
          for (let i = 0; i <= 20; i++) {
            const retention = i * (totalExpectedLoss / 10);
            // Simple mock decay function
            const loss_pct = Math.exp(-i * 0.25);
            curve.push({ retention, loss_pct });
          }

          newOutputData = {
            type: "ExposureCurveOutput",
            curve,
            totalExpectedLoss,
          };
        } else if (module.type === ModuleType.PriceXoLLayer) {
          const curveInputConnection = connections.find(
            (c) => c.to.moduleId === module.id && c.to.portName === "curve_in"
          );
          if (!curveInputConnection)
            throw new Error("'curve_in' port must be connected.");

          const curveSourceModule = currentModules.find(
            (m) => m.id === curveInputConnection.from.moduleId
          );
          if (
            !curveSourceModule ||
            curveSourceModule.outputData?.type !== "ExposureCurveOutput"
          )
            throw new Error("An exposure curve must be connected.");
          const curveOutput = curveSourceModule.outputData;

          const { retention, limit, loading_factor } = module.parameters;
          if (
            typeof retention !== "number" ||
            typeof limit !== "number" ||
            typeof loading_factor !== "number"
          ) {
            throw new Error(
              "Retention, Limit, and Loading Factor must be configured as numbers."
            );
          }

          // Mock interpolation
          const interpolate = (x: number) =>
            Math.exp(-x / (curveOutput.totalExpectedLoss / 2));

          const pct_at_retention = interpolate(retention);
          const pct_at_limit = interpolate(retention + limit);

          const layer_loss_pct = pct_at_retention - pct_at_limit;
          const expectedLayerLoss =
            curveOutput.totalExpectedLoss * layer_loss_pct;
          const rateOnLinePct =
            limit > 0 ? (expectedLayerLoss / limit) * 100 : 0;
          const premium = expectedLayerLoss * loading_factor;

          newOutputData = {
            type: "XoLPriceOutput",
            retention,
            limit,
            expectedLayerLoss,
            rateOnLinePct,
            premium,
          };
        } else if (module.type === ModuleType.ApplyThreshold) {
          const inputData = getSingleInputData(module.id) as DataPreview;
          if (!inputData) throw new Error("Input data not available.");
          const { threshold, loss_column } = module.parameters;
          if (!loss_column || typeof threshold !== "number")
            throw new Error("Threshold and Loss Column must be set.");

          const newRows = (inputData.rows || []).filter(
            (row) => (row[loss_column] as number) >= threshold
          );
          newOutputData = {
            ...inputData,
            rows: newRows,
            totalRowCount: newRows.length,
          };
        } else if (module.type === ModuleType.DefineXolContract) {
          const {
            deductible,
            limit,
            reinstatements,
            aggDeductible,
            expenseRatio,
          } = module.parameters;
          newOutputData = {
            type: "XolContractOutput",
            deductible,
            limit,
            reinstatements,
            aggDeductible,
            expenseRatio,
          };
        } else if (module.type === ModuleType.CalculateCededLoss) {
          const dataConnection = connections.find(
            (c) => c.to.moduleId === module.id && c.to.portName === "data_in"
          );
          const contractConnection = connections.find(
            (c) =>
              c.to.moduleId === module.id && c.to.portName === "contract_in"
          );
          if (!dataConnection || !contractConnection)
            throw new Error("Both data and contract inputs must be connected.");

          const dataSource = currentModules.find(
            (m) => m.id === dataConnection.from.moduleId
          );
          if (!dataSource || dataSource.outputData?.type !== "DataPreview")
            throw new Error("Input data is not valid.");

          const contractSource = currentModules.find(
            (m) => m.id === contractConnection.from.moduleId
          );
          if (
            !contractSource ||
            contractSource.outputData?.type !== "XolContractOutput"
          )
            throw new Error("Input contract is not valid.");

          const inputData = dataSource.outputData;
          const contract = contractSource.outputData;
          const { loss_column } = module.parameters;

          const newRows = (inputData.rows || []).map((row) => {
            const loss = row[loss_column] as number;
            const ceded_loss = Math.min(
              contract.limit,
              Math.max(0, loss - contract.deductible)
            );
            return { ...row, ceded_loss: ceded_loss };
          });

          const newColumns = [...inputData.columns];
          if (!newColumns.some((c) => c.name === "ceded_loss")) {
            newColumns.push({ name: "ceded_loss", type: "number" });
          }

          newOutputData = { ...inputData, columns: newColumns, rows: newRows };
        } else if (module.type === ModuleType.PriceXolContract) {
          const dataConnection = connections.find(
            (c) => c.to.moduleId === module.id && c.to.portName === "data_in"
          );
          const contractConnection = connections.find(
            (c) =>
              c.to.moduleId === module.id && c.to.portName === "contract_in"
          );
          if (!dataConnection || !contractConnection)
            throw new Error("Both data and contract inputs must be connected.");

          const dataSource = currentModules.find(
            (m) => m.id === dataConnection.from.moduleId
          );
          if (!dataSource || dataSource.outputData?.type !== "DataPreview")
            throw new Error("Input data is not valid.");

          const contractSource = currentModules.find(
            (m) => m.id === contractConnection.from.moduleId
          );
          if (
            !contractSource ||
            contractSource.outputData?.type !== "XolContractOutput"
          )
            throw new Error("Input contract is not valid.");

          const inputData = dataSource.outputData;
          const contract = contractSource.outputData;
          const { year_column, ceded_loss_column, volatility_loading } =
            module.parameters;

          const yearlyLosses: Record<string, number> = {};
          (inputData.rows || []).forEach((row) => {
            const year = row[year_column];
            const cededLoss = row[ceded_loss_column] as number;
            if (year && typeof cededLoss === "number") {
              yearlyLosses[year] = (yearlyLosses[year] || 0) + cededLoss;
            }
          });

          const yearlyLossValues = Object.values(yearlyLosses);
          if (yearlyLossValues.length === 0)
            throw new Error(
              "No yearly losses to analyze. Check input data and column names."
            );

          const sum = yearlyLossValues.reduce((a, b) => a + b, 0);
          const mean = sum / yearlyLossValues.length;
          const stdDev = Math.sqrt(
            yearlyLossValues
              .map((x) => Math.pow(x - mean, 2))
              .reduce((a, b) => a + b, 0) / (yearlyLossValues.length - 1 || 1)
          );

          const expectedLoss = mean;
          const volatilityMargin = stdDev * (volatility_loading / 100);
          const purePremium = expectedLoss + volatilityMargin;
          const expenseLoading =
            purePremium / (1 - contract.expenseRatio) - purePremium;
          const finalPremium = purePremium + expenseLoading;

          newOutputData = {
            type: "FinalXolPriceOutput",
            expectedLoss,
            stdDev,
            volatilityMargin,
            purePremium,
            expenseLoading,
            finalPremium,
          };
        } else {
          const inputConnection = connections.find(
            (c) => c.to.moduleId === module.id
          );
          if (inputConnection) {
            const sourceModule = currentModules.find(
              (sm) => sm.id === inputConnection.from.moduleId
            );
            if (sourceModule?.outputData?.type === "DataPreview") {
              newOutputData = sourceModule.outputData;
            } else if (sourceModule?.status !== ModuleStatus.Success) {
              throw new Error(
                `Upstream module [${sourceModule?.name}] did not run successfully.`
              );
            }
          }
        }
        newStatus = ModuleStatus.Success;
        logLevel = "SUCCESS";
        logMessage = `Module [${moduleName}] executed successfully.`;
      } catch (error: any) {
        newStatus = ModuleStatus.Error;
        logLevel = "ERROR";
        logMessage = `Module [${moduleName}] failed: ${error.message}`;
      }

      const finalModuleState = {
        ...module,
        status: newStatus,
        outputData: newOutputData,
      };

      // Update the mutable array for the current run
      const moduleIndex = currentModules.findIndex((m) => m.id === moduleId);
      if (moduleIndex !== -1) {
        currentModules[moduleIndex] = finalModuleState;
      }

      // If TrainModel succeeded, mark the connected model definition module as Success
      // If TrainModel is set to Pending, also mark the connected model definition module as Pending
      let modelDefinitionModuleId: string | null = null;
      let shouldUpdateModelDefinition = false;
      let modelDefinitionNewStatus: ModuleStatus | null = null;

      if (module.type === ModuleType.TrainModel) {
        const modelInputConnection = connections.find(
          (c) => c.to.moduleId === moduleId && c.to.portName === "model_in"
        );
        if (modelInputConnection) {
          modelDefinitionModuleId = modelInputConnection.from.moduleId;
          if (newStatus === ModuleStatus.Success) {
            shouldUpdateModelDefinition = true;
            modelDefinitionNewStatus = ModuleStatus.Success;
          } else if (
            newStatus === ModuleStatus.Pending ||
            newStatus === ModuleStatus.Error
          ) {
            // When TrainModel becomes Pending or Error, mark connected model definition module as Pending
            shouldUpdateModelDefinition = true;
            modelDefinitionNewStatus = ModuleStatus.Pending;
          }
        }
      }

      // Update React's state for the UI
      setModules((prev) =>
        prev.map((m) => {
          if (m.id === moduleId) {
            // viewingEvaluation이 열려있고 이 모듈이면 자동으로 업데이트
            if (viewingEvaluation && viewingEvaluation.id === moduleId) {
              setViewingEvaluation(finalModuleState);
            }
            return finalModuleState;
          }
          // Update connected model definition module status when TrainModel status changes
          if (
            shouldUpdateModelDefinition &&
            modelDefinitionModuleId &&
            m.id === modelDefinitionModuleId &&
            MODEL_DEFINITION_TYPES.includes(m.type) &&
            modelDefinitionNewStatus
          ) {
            return {
              ...m,
              status: modelDefinitionNewStatus,
              outputData:
                modelDefinitionNewStatus === ModuleStatus.Pending
                  ? undefined
                  : m.outputData,
            };
          }
          return m;
        })
      );
      addLog(logLevel, logMessage);

      if (newStatus === ModuleStatus.Error) {
        break;
      }
    }
  };

  const handleRunAll = () => {
    const rootNodes = modules.filter(
      (m) => !connections.some((c) => c.to.moduleId === m.id)
    );
    if (rootNodes.length > 0) {
      addLog(
        "INFO",
        `Project Run All started with ${rootNodes.length} root node(s)...`
      );
      setModules((prev) =>
        prev.map((m) => ({
          ...m,
          status: ModuleStatus.Pending,
          outputData: undefined,
        }))
      );
      // Run all modules starting from all root nodes
      // Pass the first root node ID, but runAll=true will traverse all root nodes
      runSimulation(rootNodes[0].id, true);
    } else if (modules.length > 0) {
      addLog(
        "WARN",
        "Circular dependency or no root nodes found. Starting from all modules."
      );
      setModules((prev) =>
        prev.map((m) => ({
          ...m,
          status: ModuleStatus.Pending,
          outputData: undefined,
        }))
      );
      // When no root nodes, runAll will traverse all modules
      runSimulation(modules[0].id, true);
    } else {
      addLog("WARN", "No modules on canvas to run.");
    }
  };

  const adjustScale = (delta: number) => {
    setScale((prev) => Math.max(0.2, Math.min(2, prev + delta)));
  };

  const selectedModule =
    modules.find(
      (m) => m.id === selectedModuleIds[selectedModuleIds.length - 1]
    ) || null;

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const activeElement = document.activeElement;
      const isEditingText =
        activeElement &&
        (activeElement.tagName === "INPUT" ||
          activeElement.tagName === "TEXTAREA" ||
          (activeElement as HTMLElement).isContentEditable);
      if (isEditingText) return;

      if (e.ctrlKey || e.metaKey) {
        if (e.key === "a" || e.key === "A") {
          e.preventDefault();
          // Ctrl+A: 모든 모듈 선택
          const allModuleIds = modules.map((m) => m.id);
          setSelectedModuleIds(allModuleIds);
          addLog("INFO", `모든 모듈 선택됨 (${allModuleIds.length}개)`);
        } else if (e.key === "z") {
          e.preventDefault();
          undo();
        } else if (e.key === "y") {
          e.preventDefault();
          redo();
        } else if (e.key === "c") {
          if (selectedModuleIds.length > 0) {
            e.preventDefault();
            pasteOffset.current = 0;
            const selectedModules = modules.filter((m) =>
              selectedModuleIds.includes(m.id)
            );
            const selectedIdsSet = new Set(selectedModuleIds);
            const internalConnections = connections.filter(
              (c) =>
                selectedIdsSet.has(c.from.moduleId) &&
                selectedIdsSet.has(c.to.moduleId)
            );
            setClipboard({
              modules: JSON.parse(JSON.stringify(selectedModules)),
              connections: JSON.parse(JSON.stringify(internalConnections)),
            });
            addLog(
              "INFO",
              `${selectedModuleIds.length} module(s) copied to clipboard.`
            );
          }
        } else if (e.key === "v") {
          e.preventDefault();
          if (clipboard) {
            pasteOffset.current += 30;
            const idMap: Record<string, string> = {};
            const newModules = clipboard.modules.map((mod) => {
              const newId = `${mod.type}-${Date.now()}-${Math.random()
                .toString(36)
                .substring(2, 7)}`;
              idMap[mod.id] = newId;
              return {
                ...mod,
                id: newId,
                position: {
                  x: mod.position.x + pasteOffset.current,
                  y: mod.position.y + pasteOffset.current,
                },
                status: ModuleStatus.Pending,
                outputData: undefined,
              };
            });
            const newConnections = clipboard.connections.map((conn) => ({
              ...conn,
              id: `conn-${Date.now()}-${Math.random()
                .toString(36)
                .substring(2, 7)}`,
              from: { ...conn.from, moduleId: idMap[conn.from.moduleId] },
              to: { ...conn.to, moduleId: idMap[conn.to.moduleId] },
            }));

            setModules((prev) => [...prev, ...newModules]);
            setConnections((prev) => [...prev, ...newConnections]);
            setSelectedModuleIds(newModules.map((m) => m.id));
            addLog(
              "INFO",
              `${newModules.length} module(s) pasted from clipboard.`
            );
          }
        }
      } else if (selectedModuleIds.length > 0) {
        if (e.key === "Delete" || e.key === "Backspace") {
          e.preventDefault();
          deleteModules([...selectedModuleIds]);
        }
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
    selectedModuleIds,
    undo,
    redo,
    setModules,
    setConnections,
    setSelectedModuleIds,
    modules,
    connections,
    clipboard,
    deleteModules,
    addLog,
  ]);

  useEffect(() => {
    const handleWindowMouseMove = (e: globalThis.MouseEvent) => {
      if (isDraggingControlPanel.current && canvasContainerRef.current) {
        const containerRect =
          canvasContainerRef.current.getBoundingClientRect();
        setControlPanelPos({
          x: e.clientX - containerRect.left - controlPanelDragOffset.current.x,
          y: e.clientY - containerRect.top - controlPanelDragOffset.current.y,
        });
      }
    };

    const handleWindowMouseUp = () => {
      isDraggingControlPanel.current = false;
    };

    window.addEventListener("mousemove", handleWindowMouseMove);
    window.addEventListener("mouseup", handleWindowMouseUp);

    return () => {
      window.removeEventListener("mousemove", handleWindowMouseMove);
      window.removeEventListener("mouseup", handleWindowMouseUp);
    };
  }, []);

  const handleControlPanelMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation(); // Prevent canvas panning
    isDraggingControlPanel.current = true;

    const rect = e.currentTarget.getBoundingClientRect();
    controlPanelDragOffset.current = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };

    // If it's the first drag (currently centered via CSS), set explicit coordinates relative to container
    if (!controlPanelPos && canvasContainerRef.current) {
      const containerRect = canvasContainerRef.current.getBoundingClientRect();
      setControlPanelPos({
        x: rect.left - containerRect.left,
        y: rect.top - containerRect.top,
      });
    }
  };

  return (
    <div className="bg-gray-900 text-white h-screen w-screen flex flex-col overflow-hidden">
      {isAiGenerating && (
        <div className="fixed inset-0 bg-black bg-opacity-70 flex flex-col items-center justify-center z-50">
          <div role="status">
            <svg
              aria-hidden="true"
              className="w-12 h-12 text-gray-200 animate-spin fill-blue-600"
              viewBox="0 0 100 101"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z"
                fill="currentColor"
              />
              <path
                d="M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0492C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z"
                fill="currentFill"
              />
            </svg>
            <span className="sr-only">Loading...</span>
          </div>
          <p className="mt-4 text-lg font-semibold text-white">
            AI가 최적의 파이프라인을 설계하고 있습니다...
          </p>
        </div>
      )}

      <header className="flex flex-col px-4 py-1.5 bg-gray-900 border-b border-gray-700 flex-shrink-0 z-20 relative overflow-visible">
        {/* 첫 번째 줄: 제목 및 모델 이름 */}
        <div className="flex items-center w-full">
          <div className="flex items-center gap-2 md:gap-4 flex-1 min-w-0">
            <LogoIcon className="h-5 w-5 md:h-6 md:w-6 text-blue-400 flex-shrink-0" />
            <h1 className="text-base md:text-xl font-bold text-blue-300 tracking-wide flex-shrink-0">
              Insure Auto Flow
            </h1>
            <div className="flex items-center gap-2 flex-shrink-0">
              <span className="text-gray-600 hidden md:inline">|</span>
              {isEditingProjectName ? (
                <input
                  value={projectName}
                  onChange={(e) => setProjectName(e.target.value)}
                  onBlur={() => setIsEditingProjectName(false)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === "Escape") {
                      setIsEditingProjectName(false);
                    }
                  }}
                  className="bg-gray-800 text-sm md:text-lg font-semibold text-white px-2 py-1 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 min-w-0"
                  autoFocus
                />
              ) : (
                <h2
                  onClick={() => setIsEditingProjectName(true)}
                  className="text-sm md:text-lg font-semibold text-gray-300 hover:bg-gray-700 px-2 py-1 rounded-md cursor-pointer truncate"
                  title="Click to edit project name"
                >
                  {projectName}
                </h2>
              )}
            </div>
          </div>
        </div>

        {/* 두 번째 줄: Load, Save 등 버튼들 */}
        <div className="flex items-center justify-end gap-2 w-full overflow-x-auto scrollbar-hide mt-1">
          <button
            onClick={undo}
            disabled={!canUndo}
            className="p-1.5 text-gray-300 hover:bg-gray-700 rounded-md disabled:text-gray-600 disabled:cursor-not-allowed transition-colors flex-shrink-0"
            title="Undo (Ctrl+Z)"
          >
            <ArrowUturnLeftIcon className="h-5 w-5" />
          </button>
          <button
            onClick={redo}
            disabled={!canRedo}
            className="p-1.5 text-gray-300 hover:bg-gray-700 rounded-md disabled:text-gray-600 disabled:cursor-not-allowed transition-colors flex-shrink-0"
            title="Redo (Ctrl+Y)"
          >
            <ArrowUturnRightIcon className="h-5 w-5" />
          </button>
          <div className="h-5 border-l border-gray-700"></div>
          <button
            onClick={handleSetFolder}
            className="flex items-center gap-2 px-3 py-1.5 text-xs bg-gray-700 hover:bg-gray-600 rounded-md font-semibold transition-colors flex-shrink-0"
            title="Set Save Folder"
          >
            <FolderOpenIcon className="h-4 w-4" />
            <span>Set Folder</span>
          </button>
          <button
            onClick={handleLoadPipeline}
            className="flex items-center gap-2 px-3 py-1.5 text-xs bg-gray-700 hover:bg-gray-600 rounded-md font-semibold transition-colors flex-shrink-0"
            title="Load Pipeline"
          >
            <FolderOpenIcon className="h-4 w-4" />
            <span>Load</span>
          </button>
          <button
            onClick={handleSavePipeline}
            disabled={!isDirty}
            className={`flex items-center gap-2 px-3 py-1.5 text-xs rounded-md font-semibold transition-colors flex-shrink-0 ${
              !isDirty
                ? "bg-gray-600 cursor-not-allowed opacity-50"
                : "bg-gray-700 hover:bg-gray-600"
            }`}
            title="Save Pipeline"
          >
            {saveButtonText === "Save" ? (
              <CodeBracketIcon className="h-4 w-4" />
            ) : (
              <CheckIcon className="h-4 w-4" />
            )}
            <span>{saveButtonText}</span>
          </button>
        </div>

        {/* 세 번째 줄: 햄버거 버튼(왼쪽) 및 AI 버튼 2개, Run All, 설정 버튼(오른쪽) */}
        <div className="flex items-center justify-between gap-1 md:gap-2 w-full mt-1 overflow-visible">
          <div className="flex items-center gap-1 md:gap-2">
            <button
              onClick={() => setIsLeftPanelVisible((v) => !v)}
              className="p-1.5 text-gray-300 hover:bg-gray-700 rounded-md transition-colors flex-shrink-0"
              aria-label="Toggle modules panel"
              title="Toggle Modules Panel"
            >
              <Bars3Icon className="h-5 w-5" />
            </button>
            <div className="h-5 border-l border-gray-700"></div>
            <div
              className="relative flex-shrink-0"
              ref={sampleMenuRef}
              style={{ zIndex: 1000 }}
            >
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  console.log(
                    "Samples button clicked, current state:",
                    isSampleMenuOpen,
                    "SAMPLE_MODELS:",
                    SAMPLE_MODELS
                  );
                  setIsSampleMenuOpen((prev) => {
                    console.log("Toggling from", prev, "to", !prev);
                    return !prev;
                  });
                }}
                className={`flex items-center gap-2 px-3 py-1.5 text-xs rounded-md font-semibold transition-colors cursor-pointer ${
                  isSampleMenuOpen
                    ? "bg-purple-600 text-white"
                    : "bg-gray-700 hover:bg-gray-600 text-gray-200"
                }`}
                title="Load Sample Model"
                type="button"
              >
                <SparklesIcon className="h-4 w-4" />
                <span>Samples</span>
              </button>
              {isSampleMenuOpen && (
                <div
                  className="absolute top-full left-0 mt-1 bg-gray-800 border border-gray-700 rounded-md shadow-xl min-w-[200px] max-h-[600px] overflow-y-auto"
                  style={{ zIndex: 9999 }}
                >
                  {/* Samples 폴더의 파일 목록 */}
                  {isLoadingSamples ? (
                    <div className="px-4 py-2 text-sm text-gray-400">
                      Loading samples...
                    </div>
                  ) : folderSamples.length > 0 ? (
                    <>
                      <div className="px-4 py-2 text-xs text-gray-500 uppercase font-bold border-b border-gray-700">
                        Samples Folder ({folderSamples.length})
                      </div>
                      {folderSamples.map((sample) => (
                        <button
                          key={sample.filename}
                          onClick={(e) => {
                            e.stopPropagation();
                            console.log(
                              "Loading sample:",
                              sample.name,
                              "from file:",
                              sample.filename
                            );
                            handleLoadSample(
                              sample.name,
                              "folder",
                              sample.filename
                            );
                          }}
                          className="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 transition-colors cursor-pointer"
                          type="button"
                          title={sample.filename}
                        >
                          {sample.name}
                        </button>
                      ))}
                      <div className="border-b border-gray-700 my-1"></div>
                    </>
                  ) : (
                    <div className="px-4 py-2 text-xs text-gray-500">
                      No samples in folder
                    </div>
                  )}

                  {/* 공유 Samples 목록 */}
                  {(() => {
                    try {
                      const savedSamples = getSavedSamples();
                      if (savedSamples && savedSamples.length > 0) {
                        return (
                          <>
                            <div className="px-4 py-2 text-xs text-gray-500 uppercase font-bold border-b border-gray-700">
                              Shared Samples
                            </div>
                            {savedSamples.map((sample: any) => (
                              <button
                                key={sample.name}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleLoadSample(sample.name, "samples");
                                }}
                                className="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 transition-colors cursor-pointer"
                                type="button"
                              >
                                {sample.name}
                              </button>
                            ))}
                            <div className="border-b border-gray-700 my-1"></div>
                          </>
                        );
                      }
                      return null;
                    } catch (error) {
                      console.error("Error rendering saved samples:", error);
                      return null;
                    }
                  })()}

                  {/* 기본 Samples 목록 */}
                  {SAMPLE_MODELS && SAMPLE_MODELS.length > 0 ? (
                    <>
                      <div className="px-4 py-2 text-xs text-gray-500 uppercase font-bold border-b border-gray-700">
                        Default Samples
                      </div>
                      {SAMPLE_MODELS.map((sample: any) => (
                        <button
                          key={sample.name}
                          onClick={(e) => {
                            e.stopPropagation();
                            handleLoadSample(sample.name, "samples");
                          }}
                          className="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 last:rounded-b-md transition-colors cursor-pointer"
                          type="button"
                        >
                          {sample.name}
                        </button>
                      ))}
                    </>
                  ) : (
                    <div className="px-4 py-2 text-sm text-gray-400 last:rounded-b-md">
                      No samples available
                    </div>
                  )}
                </div>
              )}
            </div>
            {/* My Work 버튼 */}
            <div
              className="relative flex-shrink-0"
              ref={myWorkMenuRef}
              style={{ zIndex: 1000 }}
            >
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setIsMyWorkMenuOpen((prev) => !prev);
                }}
                className={`flex items-center gap-2 px-3 py-1.5 text-xs rounded-md font-semibold transition-colors cursor-pointer ${
                  isMyWorkMenuOpen
                    ? "bg-purple-600 text-white"
                    : "bg-gray-700 hover:bg-gray-600 text-gray-200"
                }`}
                title="My Work"
                type="button"
              >
                <FolderOpenIcon className="h-4 w-4" />
                <span>My Work</span>
              </button>
              {isMyWorkMenuOpen && (
                <div
                  className="absolute top-full left-0 mt-1 bg-gray-800 border border-gray-700 rounded-md shadow-xl min-w-[200px]"
                  style={{ zIndex: 9999 }}
                >
                  {/* 파일에서 불러오기 */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      const input = document.createElement("input");
                      input.type = "file";
                      input.accept = ".json,.mla";
                      input.onchange = (event: Event) => {
                        const target = event.target as HTMLInputElement;
                        const file = target.files?.[0];
                        if (!file) return;

                        const reader = new FileReader();
                        reader.onload = (e: ProgressEvent<FileReader>) => {
                          try {
                            const content = e.target?.result as string;
                            if (!content) {
                              addLog("ERROR", "파일이 비어있습니다.");
                              return;
                            }
                            const savedState = JSON.parse(content);
                            if (savedState.modules && savedState.connections) {
                              resetModules(savedState.modules);
                              _setConnections(savedState.connections);
                              if (savedState.projectName) {
                                setProjectName(savedState.projectName);
                              }
                              setSelectedModuleIds([]);
                              setIsDirty(false);
                              addLog(
                                "SUCCESS",
                                `파일 '${file.name}'을 불러왔습니다.`
                              );
                              setIsMyWorkMenuOpen(false);
                            } else if (savedState.name && savedState.modules) {
                              // Sample 형식인 경우
                              handleLoadSample(savedState.name, "mywork");
                              setIsMyWorkMenuOpen(false);
                            } else {
                              addLog("WARN", "올바르지 않은 파일 형식입니다.");
                            }
                          } catch (error) {
                            console.error("Failed to load file:", error);
                            addLog("ERROR", "파일을 불러오는데 실패했습니다.");
                          }
                        };
                        reader.readAsText(file);
                      };
                      input.click();
                    }}
                    className="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 transition-colors cursor-pointer flex items-center gap-2 border-b border-gray-700"
                    type="button"
                  >
                    <FolderOpenIcon className="w-4 h-4 text-blue-400" />
                    <span>파일에서 불러오기</span>
                  </button>

                  {/* 현재 모델 저장 (개인용) */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      if (modules.length === 0) {
                        addLog(
                          "WARN",
                          "저장할 모델이 없습니다. 먼저 모듈을 추가해주세요."
                        );
                        setIsMyWorkMenuOpen(false);
                        return;
                      }

                      const modelName = prompt(
                        "모델 이름을 입력하세요:",
                        projectName || "My Model"
                      );
                      if (!modelName || !modelName.trim()) {
                        setIsMyWorkMenuOpen(false);
                        return;
                      }

                      const trimmedName = modelName.trim();

                      // 기존 모델 목록 가져오기
                      const existingModelsStr =
                        localStorage.getItem("myWorkModels");
                      let existingModels: any[] = [];
                      if (existingModelsStr) {
                        try {
                          existingModels = JSON.parse(existingModelsStr);
                          if (!Array.isArray(existingModels)) {
                            existingModels = [];
                          }
                        } catch (parseError) {
                          console.error(
                            "Failed to parse existing models:",
                            parseError
                          );
                          existingModels = [];
                        }
                      }

                      // 동일한 이름의 모델이 있는지 확인
                      const existingModel = existingModels.find(
                        (m: any) => m.name === trimmedName
                      );
                      if (existingModel) {
                        const shouldOverwrite = window.confirm(
                          `모델 "${trimmedName}"이 이미 존재합니다. 덮어쓰시겠습니까?`
                        );
                        if (!shouldOverwrite) {
                          setIsMyWorkMenuOpen(false);
                          return;
                        }
                      }

                      const savedModel = {
                        name: trimmedName,
                        modules: modules.map((m) => ({
                          type: m.type,
                          position: m.position,
                          name: m.name,
                          parameters: m.parameters,
                        })),
                        connections: connections
                          .map((c) => {
                            const fromIndex = modules.findIndex(
                              (m) => m.id === c.from.moduleId
                            );
                            const toIndex = modules.findIndex(
                              (m) => m.id === c.to.moduleId
                            );
                            return {
                              fromModuleIndex: fromIndex,
                              fromPort: c.from.portName,
                              toModuleIndex: toIndex,
                              toPort: c.to.portName,
                            };
                          })
                          .filter(
                            (c) =>
                              c.fromModuleIndex >= 0 && c.toModuleIndex >= 0
                          ),
                      };

                      // 같은 이름의 모델이 있으면 제거하고 새로 추가
                      const filteredModels = existingModels.filter(
                        (m: any) => m.name !== trimmedName
                      );
                      const updatedModels = [...filteredModels, savedModel];

                      localStorage.setItem(
                        "myWorkModels",
                        JSON.stringify(updatedModels)
                      );
                      setMyWorkModels(updatedModels);
                      addLog(
                        "SUCCESS",
                        `모델 "${trimmedName}"이 저장되었습니다. (개인용)`
                      );
                      setIsMyWorkMenuOpen(false);
                    }}
                    className="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 transition-colors cursor-pointer flex items-center gap-2 border-b border-gray-700"
                    type="button"
                  >
                    <PlusIcon className="w-4 h-4 text-blue-400" />
                    <span>현재 모델 저장</span>
                  </button>

                  {/* 초기 화면으로 설정 */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      const currentModel = {
                        name: projectName || "My Model",
                        modules: modules.map((m) => ({
                          type: m.type,
                          position: m.position,
                          name: m.name,
                          parameters: m.parameters,
                        })),
                        connections: connections.map((c) => ({
                          fromModuleIndex: modules.findIndex(
                            (m) => m.id === c.from.moduleId
                          ),
                          fromPort: c.from.portName,
                          toModuleIndex: modules.findIndex(
                            (m) => m.id === c.to.moduleId
                          ),
                          toPort: c.to.portName,
                        })),
                      };
                      localStorage.setItem(
                        "initialModel",
                        JSON.stringify(currentModel)
                      );
                      addLog("SUCCESS", "초기 화면으로 설정되었습니다.");
                      setIsMyWorkMenuOpen(false);
                    }}
                    className="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 transition-colors cursor-pointer flex items-center gap-2 border-b border-gray-700"
                    type="button"
                  >
                    <StarIcon className="w-4 h-4 text-yellow-400" />
                    <span className="text-green-400">초기 화면으로 설정</span>
                  </button>

                  {/* 구분선 */}
                  <div className="border-b border-gray-700 my-1"></div>

                  {/* 저장된 모델 목록 */}
                  {myWorkModels && myWorkModels.length > 0 ? (
                    myWorkModels.map((saved: any) => (
                      <button
                        key={saved.name}
                        onClick={(e) => {
                          e.stopPropagation();
                          handleLoadSample(saved.name, "mywork");
                          setIsMyWorkMenuOpen(false);
                        }}
                        className="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 last:rounded-b-md transition-colors cursor-pointer"
                        type="button"
                      >
                        {saved.name}
                      </button>
                    ))
                  ) : (
                    <div className="px-4 py-2 text-sm text-gray-400 last:rounded-b-md">
                      저장된 모델이 없습니다
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
          <div className="flex items-center gap-1 md:gap-2 ml-auto">
            <button
              onClick={() => setIsCodePanelVisible((v) => !v)}
              className="flex items-center gap-1 md:gap-2 px-1.5 md:px-2 py-0.5 md:py-1 text-[5px] md:text-[8px] bg-gray-600 hover:bg-gray-700 rounded-md font-semibold transition-colors flex-shrink-0"
              title="View Full Pipeline Code"
            >
              <CodeBracketIcon className="h-1.5 w-1.5 md:h-2.5 md:w-2.5" />
              <span className="whitespace-nowrap">전체 코드</span>
            </button>
            <button
              onClick={() => setIsGoalModalOpen(true)}
              className="flex items-center gap-1 md:gap-2 px-1.5 md:px-2 py-0.5 md:py-1 text-[5px] md:text-[8px] bg-purple-600 hover:bg-purple-700 rounded-md font-semibold transition-colors flex-shrink-0"
              title="Generate pipeline from a goal"
            >
              <SparklesIcon className="h-1.5 w-1.5 md:h-2.5 md:w-2.5" />
              <span className="whitespace-nowrap">
                AI로 파이프라인 생성하기
              </span>
            </button>
            <button
              onClick={() => setIsDataModalOpen(true)}
              className="flex items-center gap-1 md:gap-2 px-1.5 md:px-2 py-0.5 md:py-1 text-[5px] md:text-[8px] bg-indigo-600 hover:bg-indigo-700 rounded-md font-semibold transition-colors flex-shrink-0"
              title="Generate pipeline from data"
            >
              <SparklesIcon className="h-1.5 w-1.5 md:h-2.5 md:w-2.5" />
              <span className="whitespace-nowrap">
                AI로 데이터 분석 실행하기
              </span>
            </button>
            <button
              onClick={handleRunAll}
              className="flex items-center gap-1 md:gap-2 px-1.5 md:px-2 py-0.5 md:py-1 text-[7px] md:text-xs bg-green-600 hover:bg-green-500 rounded-md font-bold text-white transition-colors flex-shrink-0"
            >
              <PlayIcon className="h-2.5 w-2.5 md:h-3.5 md:w-3.5" />
              <span className="hidden sm:inline">Run All</span>
            </button>
            <button
              onClick={handleToggleRightPanel}
              className="p-1 md:p-1.5 text-gray-300 hover:bg-gray-700 rounded-md transition-colors flex-shrink-0"
              title="Toggle Properties Panel"
            >
              <CogIcon className="h-4 w-4 md:h-5 md:w-5" />
            </button>
          </div>
        </div>
      </header>

      <div className="flex-grow min-h-0 relative">
        <main
          ref={canvasContainerRef}
          className="w-full h-full canvas-bg relative overflow-hidden"
        >
          <Canvas
            modules={modules}
            connections={connections}
            setConnections={setConnections}
            selectedModuleIds={selectedModuleIds}
            setSelectedModuleIds={setSelectedModuleIds}
            updateModulePositions={updateModulePositions}
            onModuleDrop={createModule}
            scale={scale}
            setScale={setScale}
            pan={pan}
            setPan={setPan}
            canvasContainerRef={canvasContainerRef}
            onViewDetails={handleViewDetails}
            onModuleDoubleClick={handleModuleDoubleClick}
            onRunModule={(moduleId) => runSimulation(moduleId, false)}
            onDeleteModule={(id) => deleteModules([id])}
            onUpdateModuleName={updateModuleName}
            onUpdateModule={updateModule}
            suggestion={suggestion}
            onAcceptSuggestion={acceptSuggestion}
            onClearSuggestion={clearSuggestion}
            onStartSuggestion={handleSuggestModule}
            areUpstreamModulesReady={areUpstreamModulesReady}
          />
          <div
            onMouseDown={handleControlPanelMouseDown}
            style={{
              transform: controlPanelPos
                ? `translate(${controlPanelPos.x}px, ${controlPanelPos.y}px)`
                : "translate(-50%, 0)",
              cursor: "grab",
            }}
            className={`absolute bottom-8 left-1/2 -translate-x-1/2 bg-gray-900/80 backdrop-blur-md rounded-full px-4 py-2 flex items-center gap-4 shadow-2xl z-50 border border-gray-700 select-none transition-transform active:scale-95 ${
              controlPanelPos ? "" : ""
            }`}
          >
            <div className="flex items-center gap-1">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  adjustScale(-0.1);
                }}
                className="p-2 hover:bg-gray-700/50 rounded-full text-gray-400 hover:text-white transition-colors"
                title="Zoom Out"
              >
                <MinusIcon className="w-5 h-5" />
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setScale(1);
                  setPan({ x: 0, y: 0 });
                }}
                className="px-2 text-sm font-medium text-gray-300 hover:text-white min-w-[3rem] text-center"
                title="Reset View"
              >
                {Math.round(scale * 100)}%
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  adjustScale(0.1);
                }}
                className="p-2 hover:bg-gray-700/50 rounded-full text-gray-400 hover:text-white transition-colors"
                title="Zoom In"
              >
                <PlusIcon className="w-5 h-5" />
              </button>
            </div>

            <div className="w-px h-4 bg-gray-700"></div>

            <div className="flex items-center gap-1">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleFitToView();
                }}
                className="p-2 hover:bg-gray-700/50 rounded-full text-gray-400 hover:text-white transition-colors"
                title="Fit to View"
              >
                <ArrowsPointingOutIcon className="w-5 h-5" />
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleRearrangeModules();
                }}
                className="p-2 hover:bg-gray-700/50 rounded-full text-gray-400 hover:text-white transition-colors"
                title="Auto Layout"
              >
                <SparklesIcon className="w-5 h-5" />
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleRotateModules();
                }}
                className="p-2 hover:bg-gray-700/50 rounded-full text-gray-400 hover:text-white transition-colors"
                title="Rotate Modules"
              >
                <ArrowPathIcon className="w-5 h-5" />
              </button>
            </div>
          </div>
        </main>

        {/* -- Unified Side Panels -- */}
        {/* Code Panel - Rightmost */}
        <PipelineCodePanel
          modules={modules}
          connections={connections}
          isVisible={isCodePanelVisible}
          onToggle={() => setIsCodePanelVisible((v) => !v)}
        />

        {/* Toolbox Panel */}
        <div
          className={`absolute top-0 left-0 h-full z-10 transition-transform duration-300 ease-in-out ${
            isLeftPanelVisible ? "translate-x-0" : "-translate-x-full"
          }`}
          style={{ left: 0 }}
        >
          <Toolbox
            onModuleDoubleClick={handleModuleToolboxDoubleClick}
            onFontSizeChange={handleFontSizeChange}
          />
        </div>

        <div
          className={`absolute top-0 right-0 h-full z-10 transition-transform duration-300 ease-in-out ${
            isRightPanelVisible ? "translate-x-0" : "translate-x-full"
          }`}
        >
          <div
            className="flex h-full"
            style={{ width: `${rightPanelWidth}px` }}
          >
            <div
              onMouseDown={handleResizeMouseDown}
              className="flex-shrink-0 w-1.5 cursor-col-resize bg-gray-700 hover:bg-blue-500 transition-colors"
              title="Resize Panel"
            />
            <div className="flex-grow h-full min-w-0">
              <PropertiesPanel
                module={selectedModule}
                projectName={projectName}
                updateModuleParameters={updateModuleParameters}
                updateModuleName={updateModuleName}
                logs={terminalLogs}
                modules={modules}
                connections={connections}
                activeTab={activePropertiesTab}
                setActiveTab={setActivePropertiesTab}
                onViewDetails={handleViewDetails}
                folderHandle={folderHandleRef.current}
              />
            </div>
          </div>
        </div>
      </div>

      <AIPipelineFromGoalModal
        isOpen={isGoalModalOpen}
        onClose={() => setIsGoalModalOpen(false)}
        onSubmit={(goal) => {
          setIsGoalModalOpen(false);
          handleGeneratePipelineFromGoal(goal);
        }}
      />
      <AIPipelineFromDataModal
        isOpen={isDataModalOpen}
        onClose={() => setIsDataModalOpen(false)}
        onSubmit={(goal, fileContent, fileName) => {
          setIsDataModalOpen(false);
          handleGeneratePipelineFromData(goal, fileContent, fileName);
        }}
      />

      <AIPlanDisplayModal
        isOpen={!!aiPlan}
        onClose={() => setAiPlan(null)}
        plan={aiPlan || ""}
      />

      {/* -- Modals -- */}
      {viewingDataForModule &&
        (viewingDataForModule.outputData?.type === "DataPreview" ||
          viewingDataForModule.outputData?.type === "KMeansOutput" ||
          viewingDataForModule.outputData?.type ===
            "HierarchicalClusteringOutput" ||
          viewingDataForModule.outputData?.type === "DBSCANOutput" ||
          viewingDataForModule.outputData?.type === "PCAOutput") && (
          <ErrorBoundary>
            <DataPreviewModal
              module={viewingDataForModule}
              projectName={projectName}
              onClose={handleCloseModal}
            />
          </ErrorBoundary>
        )}
      {viewingDataForModule &&
        viewingDataForModule.outputData?.type === "StatisticsOutput" && (
          <StatisticsPreviewModal
            module={viewingDataForModule}
            projectName={projectName}
            onClose={handleCloseModal}
          />
        )}
      {viewingSplitDataForModule && (
        <SplitDataPreviewModal
          module={viewingSplitDataForModule}
          onClose={handleCloseModal}
        />
      )}
      {viewingTrainedModel && (
        <TrainedModelPreviewModal
          module={viewingTrainedModel}
          projectName={projectName}
          onClose={handleCloseModal}
        />
      )}
      {viewingStatsModelsResult && (
        <StatsModelsResultPreviewModal
          module={viewingStatsModelsResult}
          projectName={projectName}
          onClose={handleCloseModal}
        />
      )}
      {viewingDiversionChecker && (
        <DiversionCheckerPreviewModal
          module={viewingDiversionChecker}
          projectName={projectName}
          onClose={handleCloseModal}
        />
      )}
      {viewingEvaluateStat && (
        <EvaluateStatPreviewModal
          module={viewingEvaluateStat}
          projectName={projectName}
          onClose={handleCloseModal}
        />
      )}
      {viewingXoLPrice && (
        <XoLPricePreviewModal
          module={viewingXoLPrice}
          onClose={handleCloseModal}
        />
      )}
      {viewingFinalXolPrice && (
        <FinalXolPricePreviewModal
          module={viewingFinalXolPrice}
          onClose={handleCloseModal}
        />
      )}
      {viewingEvaluation &&
        (() => {
          // 최신 모듈 상태를 가져와서 모달에 전달
          const latestModule =
            modules.find((m) => m.id === viewingEvaluation.id) ||
            viewingEvaluation;
          return (
            <EvaluationPreviewModal
              module={latestModule}
              onClose={handleCloseModal}
              onThresholdChange={async (moduleId, newThreshold) => {
                // threshold 변경 시 파라미터만 업데이트 (재계산하지 않음)
                addLog(
                  "INFO",
                  `Evaluate Model threshold 변경: ${newThreshold.toFixed(
                    2
                  )} (재계산 없음)`
                );

                // threshold를 명시적으로 설정 (재계산하지 않음)
                setModules(
                  (prev) =>
                    prev.map((m) => {
                      if (m.id === moduleId) {
                        const updated = {
                          ...m,
                          parameters: {
                            ...m.parameters,
                            threshold: newThreshold,
                          },
                        };
                        addLog(
                          "INFO",
                          `Evaluate Model [${m.name}] threshold 업데이트: ${updated.parameters.threshold}`
                        );
                        return updated;
                      }
                      return m;
                    }),
                  true
                );
              }}
            />
          );
        })()}
    </div>
  );
};

export default App;
