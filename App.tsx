

import React, { useState, useCallback, MouseEvent, useEffect, useRef } from 'react';
import { Toolbox } from './components/Toolbox';
import { Canvas } from './components/Canvas';
import { PropertiesPanel } from './components/PropertiesPanel';
// fix: Add missing 'Port' type to handle portType argument in getSingleInputData.
import { CanvasModule, ModuleType, Connection, ModuleStatus, StatisticsOutput, DataPreview, ColumnInfo, SplitDataOutput, TrainedModelOutput, ModelDefinitionOutput, StatsModelsResultOutput, FittedDistributionOutput, ExposureCurveOutput, XoLPriceOutput, XolContractOutput, FinalXolPriceOutput, EvaluationOutput, KMeansOutput, HierarchicalClusteringOutput, PCAOutput, DBSCANOutput, MissingHandlerOutput, Port, EncoderOutput, NormalizerOutput } from './types';
import { DEFAULT_MODULES, TOOLBOX_MODULES, SAMPLE_MODELS } from './constants';
import { LogoIcon, PlayIcon, CodeBracketIcon, FolderOpenIcon, PlusIcon, MinusIcon, Bars3Icon, CogIcon, ArrowUturnLeftIcon, ArrowUturnRightIcon, SparklesIcon, ArrowsPointingOutIcon, Squares2X2Icon, CheckIcon } from './components/icons';
import useHistoryState from './hooks/useHistoryState';
import { DataPreviewModal } from './components/DataPreviewModal';
import { StatisticsPreviewModal } from './components/StatisticsPreviewModal';
import { SplitDataPreviewModal } from './components/SplitDataPreviewModal';
import { TrainedModelPreviewModal } from './components/TrainedModelPreviewModal';
import { StatsModelsResultPreviewModal } from './components/StatsModelsResultPreviewModal';
import { XoLPricePreviewModal } from './components/XoLPricePreviewModal';
import { FinalXolPricePreviewModal } from './components/FinalXolPricePreviewModal';
import { EvaluationPreviewModal } from './components/EvaluationPreviewModal';
import { AIPipelineFromGoalModal } from './components/AIPipelineFromGoalModal';
import { AIPipelineFromDataModal } from './components/AIPipelineFromDataModal';
import { AIPlanDisplayModal } from './components/AIPlanDisplayModal';
import { GoogleGenAI, Type } from "@google/genai";


type TerminalLog = {
    id: number;
    level: 'INFO' | 'WARN' | 'ERROR' | 'SUCCESS';
    message: string;
    timestamp: string;
};

type PropertiesTab = 'properties' | 'preview' | 'code';

// --- Helper Functions ---
// Note: All mathematical/statistical calculations are now performed using Pyodide (Python)
// JavaScript is only used for UI rendering and data structure transformations that don't modify Python results

// Helper function to determine model type
const isClassification = (modelType: ModuleType, modelPurpose?: 'classification' | 'regression'): boolean => {
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
    if (dualPurposeTypes.includes(modelType) && modelPurpose === 'classification') {
        return true;
    }
    return false;
};

// All regression and statistical calculations are now performed using Pyodide (Python)
// These JavaScript implementations have been removed to ensure Python-compatible results


const App: React.FC = () => {
  const [modules, setModules, undo, redo, resetModules, canUndo, canRedo] = useHistoryState<CanvasModule[]>([]);
  const [connections, _setConnections] = useState<Connection[]>([]);
  const [selectedModuleIds, setSelectedModuleIds] = useState<string[]>([]);
  const [terminalLogs, setTerminalLogs] = useState<TerminalLog[]>([]);
  const [projectName, setProjectName] = useState('Data Analysis');
  const [isEditingProjectName, setIsEditingProjectName] = useState(false);
  
  const [scale, setScale] = useState(0.8);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [viewingDataForModule, setViewingDataForModule] = useState<CanvasModule | null>(null);
  const [viewingSplitDataForModule, setViewingSplitDataForModule] = useState<CanvasModule | null>(null);
  const [viewingTrainedModel, setViewingTrainedModel] = useState<CanvasModule | null>(null);
  const [viewingStatsModelsResult, setViewingStatsModelsResult] = useState<CanvasModule | null>(null);
  const [viewingXoLPrice, setViewingXoLPrice] = useState<CanvasModule | null>(null);
  const [viewingFinalXolPrice, setViewingFinalXolPrice] = useState<CanvasModule | null>(null);
  const [viewingEvaluation, setViewingEvaluation] = useState<CanvasModule | null>(null);

  const [isAiGenerating, setIsAiGenerating] = useState(false);
  const [isGoalModalOpen, setIsGoalModalOpen] = useState(false);
  const [isDataModalOpen, setIsDataModalOpen] = useState(false);
  const [aiPlan, setAiPlan] = useState<string | null>(null);
  const [isSampleMenuOpen, setIsSampleMenuOpen] = useState(false);
  const sampleMenuRef = useRef<HTMLDivElement>(null);

  const [isLeftPanelVisible, setIsLeftPanelVisible] = useState(false);
  const [isRightPanelVisible, setIsRightPanelVisible] = useState(false);
  const [activePropertiesTab, setActivePropertiesTab] = useState<PropertiesTab>('properties');
  const [rightPanelWidth, setRightPanelWidth] = useState(384); // w-96 in Tailwind is 384px
  
  const canvasContainerRef = useRef<HTMLDivElement>(null);
  const folderHandleRef = useRef<FileSystemDirectoryHandle | null>(null);
  const [suggestion, setSuggestion] = useState<{ module: CanvasModule, connection: Connection } | null>(null);
  const [clipboard, setClipboard] = useState<{ modules: CanvasModule[], connections: Connection[] } | null>(null);
  const pasteOffset = useRef(0);

  const [isDirty, setIsDirty] = useState(false);
  const [saveButtonText, setSaveButtonText] = useState('Save');
  
  // Draggable control panel state
  const [controlPanelPos, setControlPanelPos] = useState<{ x: number; y: number } | null>(null);
  const isDraggingControlPanel = useRef(false);
  const controlPanelDragOffset = useRef({ x: 0, y: 0 });

  const setConnections = useCallback((value: React.SetStateAction<Connection[]>) => {
    const prevConnections = connections;
    const newConnections = typeof value === 'function' ? value(prevConnections) : value;
    
    // If a connection to TrainModel is removed, mark connected model definition module as Pending
    const removedConnections = prevConnections.filter(c => 
      !newConnections.some(nc => nc.id === c.id)
    );
    
    removedConnections.forEach(removedConn => {
      if (removedConn.to.moduleId) {
        const trainModelModule = modules.find(m => m.id === removedConn.to.moduleId && m.type === ModuleType.TrainModel);
        if (trainModelModule && removedConn.to.portName === 'model_in') {
          const modelDefinitionModuleId = removedConn.from.moduleId;
          setModules(prev => prev.map(m => {
            if (m.id === modelDefinitionModuleId && MODEL_DEFINITION_TYPES.includes(m.type)) {
              return { ...m, status: ModuleStatus.Pending, outputData: undefined };
            }
            return m;
          }));
        }
      }
    });
    
    _setConnections(newConnections);
    setIsDirty(true);
  }, [connections, modules, setModules]);

  // fix: Moved 'addLog' before 'handleSuggestModule' to fix "used before its declaration" error.
  const addLog = useCallback((level: TerminalLog['level'], message: string) => {
    setTerminalLogs(prev => [
      ...prev,
      {
        id: Date.now(),
        level,
        message,
        timestamp: new Date().toLocaleTimeString(),
      }
    ]);
    if (level === 'ERROR' || level === 'WARN') {
      setIsRightPanelVisible(true);
    }
  }, []);

  const handleSuggestModule = useCallback(async (fromModuleId: string, fromPortName: string) => {
    clearSuggestion();
    const fromModule = modules.find(m => m.id === fromModuleId);
    if (!fromModule) return;

    setIsAiGenerating(true);
    addLog('INFO', `AI is suggesting a module to connect to '${fromModule.name}'...`);
    try {
        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });
        const fromPort = fromModule.outputs.find(p => p.name === fromPortName);
        if (!fromPort) throw new Error("Source port not found.");

        const availableModuleTypes = TOOLBOX_MODULES.map(m => m.type).join(', ');

        const prompt = `Given a module of type '${fromModule.type}' with an output port of type '${fromPort.type}', what is the single most logical module type to connect next?
Available module types: [${availableModuleTypes}].
Respond with ONLY the module type string, for example: 'ScoreModel'`;

        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: prompt,
        });

        const suggestedType = response.text.trim() as ModuleType;
        const defaultModule = DEFAULT_MODULES.find(m => m.type === suggestedType);
        if (!defaultModule) {
            throw new Error(`AI suggested an unknown module type: '${suggestedType}'`);
        }

        const count = modules.filter(m => m.type === suggestedType).length + 1;
        const newModule: CanvasModule = {
            id: `suggestion-${suggestedType}-${Date.now()}`,
            name: `${suggestedType} ${count}`,
            type: suggestedType,
            position: { x: fromModule.position.x, y: fromModule.position.y + 180 },
            status: ModuleStatus.Pending,
            parameters: { ...defaultModule.parameters },
            inputs: [...defaultModule.inputs],
            outputs: [...defaultModule.outputs],
        };

        const toPort = newModule.inputs.find(p => p.type === fromPort.type);
        if (!toPort) {
            throw new Error(`Suggested module '${suggestedType}' has no compatible input port for type '${fromPort.type}'.`);
        }

        const newConnection: Connection = {
            id: `suggestion-conn-${Date.now()}`,
            from: { moduleId: fromModuleId, portName: fromPortName },
            to: { moduleId: newModule.id, portName: toPort.name },
        };

        setSuggestion({ module: newModule, connection: newConnection });
        addLog('SUCCESS', `AI suggested connecting a '${suggestedType}' module.`);

    } catch (error: any) {
        console.error("AI suggestion failed:", error);
        addLog('ERROR', `AI suggestion failed: ${error.message}`);
    } finally {
        setIsAiGenerating(false);
    }
  }, [modules, addLog]);

  const acceptSuggestion = useCallback(() => {
    if (suggestion) {
        const newModuleId = suggestion.module.id.replace('suggestion-', '');
        const newConnectionId = suggestion.connection.id.replace('suggestion-', '');

        const finalModule = { ...suggestion.module, id: newModuleId };
        const finalConnection = {
            ...suggestion.connection,
            id: newConnectionId,
            to: { ...suggestion.connection.to, moduleId: newModuleId }
        };

        setModules(prev => [...prev, finalModule]);
        setConnections(prev => [...prev, finalConnection]);
        setSuggestion(null);
        setIsDirty(true);
    }
  }, [suggestion, setModules, setConnections]);

  const clearSuggestion = useCallback(() => {
    setSuggestion(null);
  }, []);

  const handleResizeMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const startWidth = rightPanelWidth;
    const startX = e.clientX;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';

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
        document.body.style.cursor = 'default';
        document.body.style.userSelect = 'auto';
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  }, [rightPanelWidth]);

  const handleToggleRightPanel = () => {
    setIsRightPanelVisible(prev => !prev);
  };

  const handleModuleDoubleClick = useCallback((id: string) => {
    setSelectedModuleIds(prev => {
        if (prev.length === 1 && prev[0] === id) {
            return prev;
        }
        return [id];
    });
    setIsRightPanelVisible(true);
    setActivePropertiesTab('properties');
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

    modules.forEach(module => {
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

    const newPanX = (canvasRect.width - contentWidth * newScale) / 2 - minX * newScale;
    const newPanY = (canvasRect.height - contentHeight * newScale) / 2 - minY * newScale;

    setScale(newScale);
    setPan({ x: newPanX, y: newPanY });
  }, [modules]);
  
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
      ModuleType.StatModels,
    ];

    // 1. Build graph representations
    const adj: Record<string, string[]> = {};
    const revAdj: Record<string, string[]> = {};
    const inDegree: Record<string, number> = {};
    modules.forEach(m => {
        adj[m.id] = [];
        revAdj[m.id] = [];
        inDegree[m.id] = 0;
    });

    connections.forEach(conn => {
        if (adj[conn.from.moduleId] && revAdj[conn.to.moduleId]) {
            adj[conn.from.moduleId].push(conn.to.moduleId);
            revAdj[conn.to.moduleId].push(conn.from.moduleId);
            inDegree[conn.to.moduleId]++;
        }
    });

    // 2. Topological sort to get execution order (top to bottom)
    const queue = modules.filter(m => inDegree[m.id] === 0).map(m => m.id);
    const sortedModuleIds: string[] = [];
    const tempInDegree = { ...inDegree };
    while (queue.length > 0) {
        const u = queue.shift()!;
        sortedModuleIds.push(u);
        (adj[u] || []).forEach(v => {
            tempInDegree[v]--;
            if (tempInDegree[v] === 0) {
                queue.push(v);
            }
        });
    }

    // Handle cycles/unreachable nodes by appending them
    if (sortedModuleIds.length < modules.length) {
        addLog('WARN', 'Cycle detected or modules are unreachable. Appending to layout.');
        modules.forEach(m => {
            if (!sortedModuleIds.includes(m.id)) {
                sortedModuleIds.push(m.id);
            }
        });
    }

    // 3. Separate auxiliary modules from regular modules
    const regularModules: string[] = [];
    const auxiliaryModules: Record<string, string> = {}; // moduleId -> parentModuleId
    
    sortedModuleIds.forEach(moduleId => {
        const module = modules.find(m => m.id === moduleId);
        if (!module) return;

        if (MODEL_DEFINITION_TYPES.includes(module.type)) {
            // Find the module it connects to (usually TrainModel)
            const connection = connections.find(c => c.from.moduleId === moduleId);
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
    const modulePositions: Record<string, { x: number, y: number }> = {};
    let currentY = initialY;

    // Place regular modules vertically from top to bottom
    regularModules.forEach((moduleId) => {
        modulePositions[moduleId] = {
            x: initialX,
            y: currentY
        };
        currentY += vSpacing;
    });

    // 5. Place auxiliary modules to the left of their parent modules
    Object.entries(auxiliaryModules).forEach(([auxModuleId, parentModuleId]) => {
        const parentPos = modulePositions[parentModuleId];
        if (parentPos) {
            modulePositions[auxModuleId] = {
                x: parentPos.x - moduleWidth - auxiliaryOffset,
                y: parentPos.y
            };
        } else {
            // If parent not found, place at initial position
            modulePositions[auxModuleId] = {
                x: initialX - moduleWidth - auxiliaryOffset,
                y: initialY
            };
        }
    });

    // 6. Update module positions
    newModules.forEach((module, index) => {
        const pos = modulePositions[module.id];
        if (pos) {
            newModules[index] = {
                ...module,
                position: pos
            };
        }
    });

    // 7. Update state
    setModules(newModules);
    setIsDirty(true);
    setTimeout(() => handleFitToView(), 0);
  }, [modules, connections, setModules, handleFitToView, addLog]);
  
  const handleGeneratePipeline = async (prompt: string, type: 'goal' | 'data', file?: { content: string, name: string }) => {
    setIsAiGenerating(true);
    addLog('INFO', 'AI pipeline generation started...');
    try {
        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });

        const moduleDescriptions: Record<string, string> = {
            LoadData: "Loads a dataset from a user-provided CSV file.",
            Statistics: "Calculates descriptive statistics and correlation matrix for a dataset.",
            SelectData: "Selects or removes specific columns from a dataset.",
            HandleMissingValues: "Handles missing (null) values in a dataset by removing rows or filling values.",
            EncodeCategorical: "Converts categorical (string) columns into numerical format for modeling.",
            NormalizeData: "Scales numerical features to a standard range (e.g., 0-1).",
            TransitionData: "Applies mathematical transformations (e.g., log, sqrt) to numeric columns.",
            SplitData: "Splits a dataset into training and testing sets.",
            LinearRegression: "Defines a scikit-learn Linear Regression model.",
            LogisticRegression: "Defines a Logistic Regression model for classification.",
            DecisionTreeClassifier: "Defines a Decision Tree model for classification.",
            StatModels: "Defines a statistical model from the statsmodels library (e.g., OLS, Logit).",
            TrainModel: "Trains a model algorithm using a training dataset.",
            ResultModel: "Fits a statistical model (from StatModels) to a dataset and shows the results summary.",
            ScoreModel: "Applies a trained ML model to a dataset to generate predictions.",
            PredictModel: "Applies a fitted statistical model to a dataset to generate predictions.",
            EvaluateModel: "Evaluates the performance of a trained model on a test dataset.",
            FitLossDistribution: "Fits a statistical distribution (e.g., Pareto) to loss data.",
            GenerateExposureCurve: "Generates an exposure curve from a fitted distribution.",
            PriceXoLLayer: "Calculates the premium for an Excess of Loss (XoL) layer using an exposure curve.",
            XolLoading: "Loads claims data specifically for experience-based XoL pricing.",
            ApplyThreshold: "Filters out claims that are below a specified monetary threshold.",
            DefineXolContract: "Defines the parameters for an XoL reinsurance contract.",
            CalculateCededLoss: "Calculates the ceded loss for each claim based on contract terms.",
            PriceXolContract: "Prices an XoL contract using the burning cost method based on historical data."
        };

        const detailedModulesString = DEFAULT_MODULES.map(defaultModule => {
            const moduleInfo = TOOLBOX_MODULES.find(m => m.type === defaultModule.type);
            const description = moduleDescriptions[defaultModule.type] || "A standard module.";
            return `
- type: ${defaultModule.type}
  name: ${moduleInfo?.name}
  description: ${description}
  inputs: ${JSON.stringify(defaultModule.inputs)}
  outputs: ${JSON.stringify(defaultModule.outputs)}
`;
        }).join('');

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
            model: 'gemini-2.5-pro',
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
                                required: ['type', 'name']
                            }
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
                                required: ['fromModuleIndex', 'fromPort', 'toModuleIndex', 'toPort']
                            }
                        }
                    },
                    required: ['plan', 'modules', 'connections']
                }
            }
        });
        
        const responseText = response.text.trim();
        const pipeline = JSON.parse(responseText);

        if (!pipeline.modules || !pipeline.connections || !pipeline.plan) {
            throw new Error("AI response is missing 'plan', 'modules', or 'connections'.");
        }
        
        setAiPlan(pipeline.plan);

        // --- Render the generated pipeline ---
        const previousState = { modules: [...modules], connections: [...connections] };
        
        const newModules: CanvasModule[] = [];
        pipeline.modules.forEach((mod: any, index: number) => {
            const defaultData = DEFAULT_MODULES.find(m => m.type === mod.type);
            if (!defaultData) {
                addLog('WARN', `AI generated an unknown module type: '${mod.type}'. Skipping.`);
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
            
            if (file && (newModule.type === ModuleType.LoadData || newModule.type === ModuleType.XolLoading)) {
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

        addLog('SUCCESS', 'AI successfully generated a new pipeline.');

    } catch (error: any) {
        console.error("AI pipeline generation failed:", error);
        addLog('ERROR', `AI generation failed: ${error.message}`);
    } finally {
        setIsAiGenerating(false);
    }
  };

  const handleGeneratePipelineFromGoal = (goal: string) => {
    handleGeneratePipeline(`Goal: ${goal}`, 'goal');
  };

  const handleGeneratePipelineFromData = (goal: string, fileContent: string, fileName: string) => {
      const lines = fileContent.trim().split('\n');
      if (lines.length === 0) {
          addLog('ERROR', 'Uploaded file is empty.');
          return;
      }
      const header = lines[0];
      const dataPrompt = `
Goal: ${goal}
---
Dataset Columns:
${header}
`;
      handleGeneratePipeline(dataPrompt, 'data', { content: fileContent, name: fileName });
  };

  const handleSavePipeline = useCallback(async () => {
    try {
        const pipelineState = { modules, connections, projectName };
        const blob = new Blob([JSON.stringify(pipelineState, null, 2)], { type: 'application/json' });
        const fileName = `${projectName.replace(/[<>:"/\\|?*]/g, '_')}.mla`;

        // fix: Cast window to `any` to access `showSaveFilePicker` which is not in default TS types.
        if ((window as any).showSaveFilePicker) {
            const handle = await (window as any).showSaveFilePicker({
                suggestedName: fileName,
                types: [{
                    description: 'ML Pipeline File',
                    accept: { 'application/json': ['.mla'] },
                }],
            });
            const writable = await handle.createWritable();
            await writable.write(blob);
            await writable.close();
            addLog('SUCCESS', `Pipeline saved to '${handle.name}'.`);
        } else {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = fileName;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            addLog('SUCCESS', `Pipeline download initiated as '${fileName}'.`);
        }

        setIsDirty(false);
        setSaveButtonText('Saved!');
        setTimeout(() => setSaveButtonText('Save'), 2000);
    } catch (error: any) {
        if (error.name !== 'AbortError') {
            console.error('Failed to save pipeline:', error);
            addLog('ERROR', `Failed to save pipeline: ${error.message}`);
        }
    }
  }, [modules, connections, projectName, addLog]);

  const handleLoadPipeline = useCallback(() => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.mla';
    input.onchange = (event: Event) => {
      const target = event.target as HTMLInputElement;
      const file = target.files?.[0];
      if (!file) {
        return;
      }

      const reader = new FileReader();
      reader.onload = (e: ProgressEvent<FileReader>) => {
        try {
          const content = e.target?.result as string;
          if (!content) {
            addLog('ERROR', 'File is empty.');
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
            addLog('SUCCESS', `Pipeline '${file.name}' loaded successfully.`);
          } else {
            addLog('WARN', 'Invalid pipeline file format.');
          }
        } catch (error) {
          console.error('Failed to load or parse pipeline file:', error);
          addLog('ERROR', 'Failed to load pipeline from file. It may be corrupted or in the wrong format.');
        }
      };
      reader.onerror = () => {
        addLog('ERROR', `Error reading file: ${reader.error}`);
      }
      reader.readAsText(file);
    };
    input.click();
  }, [resetModules, addLog]);

  const handleLoadSample = useCallback((sampleName: string) => {
    const sampleModel = SAMPLE_MODELS.find((m: any) => m.name === sampleName);
    if (!sampleModel) {
      addLog('ERROR', `Sample model "${sampleName}" not found.`);
      return;
    }

    // Convert sample model format to app format
    const newModules: CanvasModule[] = sampleModel.modules.map((m: any, index: number) => {
      const moduleId = `module-${Date.now()}-${index}`;
      const defaultModule = DEFAULT_MODULES.find(dm => dm.type === m.type);
      return {
        ...defaultModule!,
        id: moduleId,
        name: m.name || defaultModule!.name,
        position: m.position,
        status: ModuleStatus.Pending,
      };
    });

    const newConnections: Connection[] = sampleModel.connections.map((conn: any, index: number) => {
      const fromModule = newModules[conn.fromModuleIndex];
      const toModule = newModules[conn.toModuleIndex];
      return {
        id: `connection-${Date.now()}-${index}`,
        from: { moduleId: fromModule.id, portName: conn.fromPort },
        to: { moduleId: toModule.id, portName: conn.toPort },
      };
    });

    resetModules(newModules);
    _setConnections(newConnections);
    setSelectedModuleIds([]);
    setIsDirty(false);
    setProjectName(sampleName);
    setIsSampleMenuOpen(false);
    addLog('SUCCESS', `Sample model "${sampleName}" loaded successfully.`);
    setTimeout(() => handleFitToView(), 100);
  }, [resetModules, addLog, handleFitToView]);

  // Close sample menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (sampleMenuRef.current && !sampleMenuRef.current.contains(event.target as Node)) {
        setIsSampleMenuOpen(false);
      }
    };

    if (isSampleMenuOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isSampleMenuOpen]);
  
// fix: Added missing handleSetFolder function to resolve "Cannot find name" error.
  const handleSetFolder = useCallback(async () => {
    try {
      if (!('showDirectoryPicker' in window)) {
        addLog('WARN', '현재 브라우저에서는 폴더 설정 기능을 지원하지 않습니다.');
        return;
      }
      const handle = await (window as any).showDirectoryPicker();
      folderHandleRef.current = handle;
      addLog('SUCCESS', `저장 폴더가 '${handle.name}'(으)로 설정되었습니다.`);
    } catch (error: any) {
      if (error.name !== 'AbortError') {
        console.error('Failed to set save folder:', error);
        addLog('ERROR', `폴더를 설정하지 못했습니다: ${error.message}. 브라우저 권한 설정을 확인해 주세요.`);
      }
    }
  }, [addLog]);
  
  const createModule = useCallback((type: ModuleType, position: { x: number; y: number }) => {
    clearSuggestion();
    const defaultData = DEFAULT_MODULES.find(m => m.type === type);
    if (!defaultData) return;
    
    const moduleInfo = TOOLBOX_MODULES.find(m => m.type === type);
    const baseName = moduleInfo ? moduleInfo.name : type;

    const count = modules.filter(m => m.type === type).length + 1;
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

    setModules(prev => [...prev, newModule]);
    setSelectedModuleIds([newModule.id]);
    setIsDirty(true);
  }, [modules, setModules, setSelectedModuleIds, clearSuggestion]);
  
  const handleModuleToolboxDoubleClick = useCallback((type: ModuleType) => {
    if (canvasContainerRef.current) {
        const canvasRect = canvasContainerRef.current.getBoundingClientRect();
        // Position in the middle, accounting for current pan and scale
        const position = {
            x: (canvasRect.width / 2 - 128 - pan.x) / scale, // 128 is half module width (256/2)
            y: (canvasRect.height / 2 - 60 - pan.y) / scale, // 60 is half module height
        };
        createModule(type, position);
    }
  }, [createModule, scale, pan]);

  const updateModulePositions = useCallback((updates: {id: string, position: {x: number, y: number}}[]) => {
      const updatesMap = new Map(updates.map(u => [u.id, u.position]));
      setModules(prev => prev.map(m => {
          const newPos = updatesMap.get(m.id);
          return newPos ? { ...m, position: newPos } : m;
      }), true);
      setIsDirty(true);
  }, [setModules]);

  // Helper function to find all downstream modules (modules that depend on the given module)
  const getDownstreamModules = useCallback((moduleId: string, allModules: CanvasModule[], allConnections: Connection[]): string[] => {
    const downstream: string[] = [];
    const visited = new Set<string>();
    
    const traverse = (currentId: string) => {
      if (visited.has(currentId)) return;
      visited.add(currentId);
      
      // Find all modules that receive output from this module
      const outgoingConnections = allConnections.filter(c => c.from.moduleId === currentId);
      outgoingConnections.forEach(conn => {
        const targetId = conn.to.moduleId;
        if (!downstream.includes(targetId)) {
          downstream.push(targetId);
          traverse(targetId); // Recursively find downstream modules
        }
      });
    };
    
    traverse(moduleId);
    return downstream;
  }, []);

  const updateModuleParameters = useCallback((id:string, newParams: Record<string, any>) => {
    setModules(prev => {
      const updated = prev.map(m => m.id === id ? { ...m, parameters: {...m.parameters, ...newParams}} : m);
      
      // Find all downstream modules
      const downstreamIds = getDownstreamModules(id, updated, connections);
      
      // Find connected model definition module if this is TrainModel
      let modelDefinitionModuleId: string | null = null;
      const modifiedModule = updated.find(m => m.id === id);
      if (modifiedModule && modifiedModule.type === ModuleType.TrainModel) {
        const modelInputConnection = connections.find(c => c.to.moduleId === id && c.to.portName === 'model_in');
        if (modelInputConnection) {
          modelDefinitionModuleId = modelInputConnection.from.moduleId;
        }
      }
      
      // Mark modified module and all downstream modules as Pending
      // Also mark connected model definition module as Pending if TrainModel is modified
      return updated.map(m => {
        if (m.id === id || downstreamIds.includes(m.id)) {
          return { ...m, status: ModuleStatus.Pending, outputData: undefined };
        }
        // Mark connected model definition module as Pending when TrainModel is modified
        if (modelDefinitionModuleId && m.id === modelDefinitionModuleId && MODEL_DEFINITION_TYPES.includes(m.type)) {
          return { ...m, status: ModuleStatus.Pending, outputData: undefined };
        }
        return m;
      });
    });
    setIsDirty(true);
  }, [setModules, connections, getDownstreamModules]);

  const updateModuleName = useCallback((id: string, newName: string) => {
    setModules(prev => {
      const updated = prev.map(m => (m.id === id ? { ...m, name: newName } : m));
      
      // Find all downstream modules
      const downstreamIds = getDownstreamModules(id, updated, connections);
      
      // Mark modified module and all downstream modules as Pending
      return updated.map(m => {
        if (m.id === id || downstreamIds.includes(m.id)) {
          return { ...m, status: ModuleStatus.Pending, outputData: undefined };
        }
        return m;
      });
    });
    setIsDirty(true);
  }, [setModules, connections, getDownstreamModules]);

  const deleteModules = useCallback((idsToDelete: string[]) => {
    setModules(prev => prev.filter(m => !idsToDelete.includes(m.id)));
    setConnections(prev => prev.filter(c => !idsToDelete.includes(c.from.moduleId) && !idsToDelete.includes(c.to.moduleId)));
    setSelectedModuleIds(prev => prev.filter(id => !idsToDelete.includes(id)));
    setIsDirty(true);
  }, [setModules, setConnections, setSelectedModuleIds]);

    const handleViewDetails = (moduleId: string) => {
        const module = modules.find(m => m.id === moduleId);
        if (module?.outputData) {
            if (module.outputData.type === 'StatsModelsResultOutput') {
                setViewingStatsModelsResult(module);
            } else if (module.outputData.type === 'SplitDataOutput') {
                setViewingSplitDataForModule(module);
            } else if (module.outputData.type === 'TrainedModelOutput') {
                setViewingTrainedModel(module);
            } else if (module.outputData.type === 'XoLPriceOutput') {
                setViewingXoLPrice(module);
            } else if (module.outputData.type === 'FinalXolPriceOutput') {
                setViewingFinalXolPrice(module);
            } else if (module.outputData.type === 'EvaluationOutput') {
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
    // Statistical Models
    ModuleType.StatModels,
  ];

  // Helper function to check if all upstream modules are successfully executed
  const areUpstreamModulesReady = useCallback((moduleId: string, allModules: CanvasModule[], allConnections: Connection[]): boolean => {
    const upstreamConnections = allConnections.filter(c => c.to.moduleId === moduleId);
    if (upstreamConnections.length === 0) return true; // No dependencies
    
    return upstreamConnections.every(conn => {
      const sourceModule = allModules.find(m => m.id === conn.from.moduleId);
      // Model definition modules are always considered ready (they don't need to be executed)
      if (sourceModule && MODEL_DEFINITION_TYPES.includes(sourceModule.type)) {
        return true;
      }
      return sourceModule?.status === ModuleStatus.Success;
    });
  }, []);

  const runSimulation = async (startModuleId: string, runAll: boolean = false) => {
    const runQueue: string[] = [];
    const visited = new Set<string>();
    let currentModules = [...modules]; // Use a mutable copy for the current simulation run

    const traverse = (moduleId: string) => {
        if(visited.has(moduleId)) return;
        visited.add(moduleId);
        
        const module = currentModules.find(m => m.id === moduleId);
        const isModelDefinition = module && MODEL_DEFINITION_TYPES.includes(module.type);
        
        // In Run All mode, skip model definition modules but still process their dependencies
        if (runAll && isModelDefinition) {
            // Still traverse upstream to ensure dependencies are included
            const upstreamConnections = connections.filter(c => c.to.moduleId === moduleId);
            const parentModules = currentModules.filter(m => upstreamConnections.some(c => c.from.moduleId === m.id));
            parentModules.forEach(p => traverse(p.id));
            
            // Traverse downstream to ensure modules that depend on this model definition are included
            const downstreamConnections = connections.filter(c => c.from.moduleId === moduleId);
            const childModules = currentModules.filter(m => downstreamConnections.some(c => c.to.moduleId === m.id));
            childModules.forEach(child => traverse(child.id));
            return; // Don't add model definition to queue
        }
        
        // Traverse upstream dependencies first
        const upstreamConnections = connections.filter(c => c.to.moduleId === moduleId);
        const parentModules = currentModules.filter(m => upstreamConnections.some(c => c.from.moduleId === m.id));
        parentModules.forEach(p => traverse(p.id));

        // Add to queue if not already present
        if(!runQueue.includes(moduleId)) {
            runQueue.push(moduleId);
        }
        
        // In Run All mode, also traverse downstream to ensure all connected modules are included
        if (runAll) {
            const downstreamConnections = connections.filter(c => c.from.moduleId === moduleId);
            const childModules = currentModules.filter(m => downstreamConnections.some(c => c.to.moduleId === m.id));
            childModules.forEach(child => {
                if (!visited.has(child.id)) {
                    traverse(child.id);
                }
            });
        }
    };
    
    if (runAll) {
      // Run All: traverse from all root nodes to include all modules
      // Ignore startModuleId and traverse all root nodes
      const rootNodes = currentModules.filter(m => !connections.some(c => c.to.moduleId === m.id));
      if (rootNodes.length > 0) {
        // Traverse all root nodes to ensure all modules are included
        rootNodes.forEach(node => traverse(node.id));
      } else {
        // If no root nodes (circular dependencies), traverse all modules
        currentModules.forEach(m => traverse(m.id));
      }
    } else {
      // Individual module run: only run this module (but still check dependencies)
      traverse(startModuleId);
      // Only keep the target module in the queue for individual runs
      runQueue.length = 0;
      runQueue.push(startModuleId);
    }
    
    const getSingleInputData = (moduleId: string, portType: Port['type'] = 'data'): (DataPreview | MissingHandlerOutput | EncoderOutput | NormalizerOutput) | null => {
        const inputConnection = connections.find(c => {
            if (c.to.moduleId === moduleId) {
                const targetModule = currentModules.find(m => m.id === moduleId);
                const targetPort = targetModule?.inputs.find(p => p.name === c.to.portName);
                return targetPort?.type === portType;
            }
            return false;
        });

        if (!inputConnection) return null;
        const sourceModule = currentModules.find(sm => sm.id === inputConnection.from.moduleId);
        if (!sourceModule?.outputData) return null;

        if (sourceModule.outputData.type === 'SplitDataOutput' && portType === 'data') {
            const portName = inputConnection.from.portName;
            if (portName === 'train_data_out') return sourceModule.outputData.train;
            if (portName === 'test_data_out') return sourceModule.outputData.test;
        }

        if (
            (sourceModule.outputData.type === 'DataPreview' && portType === 'data') ||
            (sourceModule.outputData.type === 'MissingHandlerOutput' && portType === 'handler') ||
            (sourceModule.outputData.type === 'EncoderOutput' && portType === 'handler') ||
            (sourceModule.outputData.type === 'NormalizerOutput' && portType === 'handler')
        ) {
            return sourceModule.outputData;
        }

        return null;
    }

    for (const moduleId of runQueue) {
        const module = currentModules.find(m=>m.id===moduleId)!;
        const moduleName = module.name;

        // Check if upstream modules are ready (only for individual runs, not Run All)
        if (!runAll && !areUpstreamModulesReady(moduleId, currentModules, connections)) {
            addLog('WARN', `Module [${moduleName}] cannot run: upstream modules are not ready.`);
            setModules(prev => prev.map(m => m.id === moduleId ? {...m, status: ModuleStatus.Error} : m));
            continue;
        }

        setModules(prev => prev.map(m => m.id === moduleId ? {...m, status: ModuleStatus.Running} : m));
        addLog('INFO', `Module [${moduleName}] execution started.`);
        
        await new Promise(resolve => setTimeout(resolve, 500));
        
        let newStatus = ModuleStatus.Error;
        let newOutputData: CanvasModule['outputData'] | undefined = undefined;
        let logMessage = `Module [${moduleName}] failed.`;
        let logLevel: TerminalLog['level'] = 'ERROR';

        try {
            if (module.type === ModuleType.LoadData || module.type === ModuleType.XolLoading) {
                const fileContent = module.parameters.fileContent as string;
                if (!fileContent) throw new Error("No file content loaded. Please select a CSV file.");
                
                // CSV 파싱 함수 (따옴표 처리 포함)
                const parseCSVLine = (line: string): string[] => {
                    const result: string[] = [];
                    let current = '';
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
                        } else if (char === ',' && !inQuotes) {
                            // 쉼표로 필드 구분
                            result.push(current.trim());
                            current = '';
                        } else {
                            current += char;
                        }
                    }
                    result.push(current.trim()); // 마지막 필드
                    return result;
                };
                
                const lines = fileContent.trim().split(/\r?\n/).filter(line => line.trim() !== '');
                if (lines.length < 1) throw new Error("CSV file is empty or invalid.");

                const header = parseCSVLine(lines[0]).map(h => h.replace(/^"|"$/g, ''));
                if (header.length === 0) throw new Error("CSV file has no header row.");
                
                const stringRows = lines.slice(1).map(line => {
                    const values = parseCSVLine(line).map(v => v.replace(/^"|"$/g, ''));
                    const rowObj: Record<string, string> = {};
                    header.forEach((col, index) => {
                        rowObj[col] = values[index] || '';
                    });
                    return rowObj;
                });

                // 컬럼 이름 중복 처리
                const uniqueHeader: string[] = [];
                const headerCount: Record<string, number> = {};
                header.forEach(name => {
                    const originalName = name || 'Unnamed';
                    if (headerCount[originalName] !== undefined) {
                        headerCount[originalName]++;
                        uniqueHeader.push(`${originalName}_${headerCount[originalName]}`);
                    } else {
                        headerCount[originalName] = 0;
                        uniqueHeader.push(originalName);
                    }
                });

                const columns: ColumnInfo[] = uniqueHeader.map(name => {
                    const sample = stringRows.slice(0, 100).map(r => r[name]).filter(v => v !== undefined && v !== null && String(v).trim() !== '');
                    const allAreNumbers = sample.length > 0 && sample.every(v => {
                        const num = Number(v);
                        return !isNaN(num) && isFinite(num);
                    });
                    return { name, type: allAreNumbers ? 'number' : 'string' };
                });

                if (columns.length === 0) {
                    throw new Error("No valid columns found in CSV file.");
                }

                const rows = stringRows.map((stringRow, rowIndex) => {
                    const typedRow: Record<string, string | number | null> = {};
                    for(const col of columns) {
                        const val = stringRow[col.name];
                        if(col.type === 'number') {
                            const numVal = val && String(val).trim() !== '' ? parseFloat(String(val)) : null;
                            typedRow[col.name] = (numVal !== null && !isNaN(numVal) && isFinite(numVal)) ? numVal : null;
                        } else {
                            typedRow[col.name] = val || null;
                        }
                    }
                    return typedRow;
                }).filter(row => {
                    // 모든 값이 null인 행 제거
                    return Object.values(row).some(val => val !== null && val !== '');
                });

                if (rows.length === 0) {
                    throw new Error("No valid data rows found in CSV file after parsing.");
                }

                // 전체 데이터를 저장 (View Details에서는 미리보기만 제한하여 표시)
                newOutputData = { type: 'DataPreview', columns, totalRowCount: rows.length, rows: rows };

            } else if (module.type === ModuleType.SelectData) {
                const inputData = getSingleInputData(module.id) as DataPreview;
                if (inputData) {
                    const selections = module.parameters.columnSelections as Record<string, { selected: boolean; type: string; }> || {};
                    const isConfigured = Object.keys(selections).length > 0;

                    const newColumns: ColumnInfo[] = [];
                    inputData.columns.forEach(col => {
                        const selection = selections[col.name];
                        // If the module is unconfigured, default to selecting all columns. Otherwise, respect the selection.
                        if (!isConfigured || selection?.selected) {
                            newColumns.push({ name: col.name, type: selection?.type ?? col.type });
                        }
                    });
                    
                    if (isConfigured && newColumns.length === 0 && inputData.columns.length > 0) {
                        throw new Error("No columns selected.");
                    }
                    
                    const newRows = (inputData.rows || []).map(row => {
                        const newRow: Record<string, any> = {};
                        newColumns.forEach(col => {
                            const originalValue = row[col.name];
                            let newValue = originalValue; // Default to original
                            
                            if (col.type === 'number') {
                                // Preserve null/undefined, and convert empty strings to null.
                                // If conversion to number fails (NaN), also treat as null.
                                if (originalValue === null || originalValue === undefined || String(originalValue).trim() === '') {
                                    newValue = null;
                                } else {
                                    const num = Number(originalValue);
                                    newValue = isNaN(num) ? null : num;
                                }
                            } else if (col.type === 'string') {
                                // Preserve null/undefined instead of converting to "null" or empty string.
                                newValue = (originalValue === null || originalValue === undefined) ? null : String(originalValue);
                            }
                            // For any other data types, the original value is preserved by default.

                            newRow[col.name] = newValue;
                        });
                        return newRow;
                    });
                    newOutputData = { type: 'DataPreview', columns: newColumns, totalRowCount: inputData.totalRowCount, rows: newRows };
                } else {
                     throw new Error("Input data not available or is of the wrong type.");
                }
            } else if (module.type === ModuleType.HandleMissingValues) {
                const inputData = getSingleInputData(module.id) as DataPreview;
                if (!inputData) throw new Error("Input data not available.");

                const { method, strategy, columns, n_neighbors } = module.parameters;
                
                // Pyodide를 사용하여 Python으로 결측치 처리 통계 계산
                try {
                    addLog('INFO', 'Pyodide를 사용하여 Python으로 결측치 처리 통계 계산 중...');
                    
                    const pyodideModule = await import('./utils/pyodideRunner');
                    const { handleMissingValuesPython } = pyodideModule;
                    
                    const result = await handleMissingValuesPython(
                        inputData.rows || [],
                        method || 'impute',
                        strategy || 'mean',
                        columns || null,
                        parseInt(n_neighbors) || 5,
                        60000 // 타임아웃: 60초
                    );
                    
                    newOutputData = {
                        type: 'MissingHandlerOutput',
                        method,
                        strategy: strategy || 'mean',
                        n_neighbors: parseInt(n_neighbors) || 5,
                        metric: module.parameters.metric,
                        imputation_values: result.imputation_values,
                    };
                    
                    addLog('SUCCESS', 'Python으로 결측치 처리 통계 계산 완료');
                } catch (error: any) {
                    const errorMessage = error.message || String(error);
                    addLog('ERROR', `Python HandleMissingValues 실패: ${errorMessage}`);
                    throw new Error(`결측치 처리 실패: ${errorMessage}`);
                }

            } else if (module.type === ModuleType.TransformData) {
                const handler = getSingleInputData(module.id, 'handler') as MissingHandlerOutput | EncoderOutput | NormalizerOutput;
                const inputData = getSingleInputData(module.id, 'data') as DataPreview;

                if (!handler) throw new Error("A handler module must be connected to 'handler_in'.");
                if (!inputData) throw new Error("A data module must be connected to 'data_in'.");
                
                const { exclude_columns = [] } = module.parameters;
                
                // Pyodide를 사용하여 Python으로 변환 적용
                try {
                    addLog('INFO', 'Pyodide를 사용하여 Python으로 변환 적용 중...');
                    
                    const pyodideModule = await import('./utils/pyodideRunner');
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
                        totalRowCount: result.rows.length 
                    };
                    
                    addLog('SUCCESS', 'Python으로 변환 적용 완료');
                } catch (error: any) {
                    const errorMessage = error.message || String(error);
                    addLog('ERROR', `Python TransformData 실패: ${errorMessage}`);
                    throw new Error(`변환 적용 실패: ${errorMessage}`);
                }

            } else if (module.type === ModuleType.EncodeCategorical) {
                const inputData = getSingleInputData(module.id) as DataPreview;
                if (!inputData) throw new Error("Input data not available.");
                const { method, columns: targetColumns, ordinal_mapping: ordinalMappingStr, drop, handle_unknown } = module.parameters;
                
                const columnsToEncode = (targetColumns && targetColumns.length > 0) 
                    ? targetColumns 
                    : inputData.columns.filter(c => c.type === 'string').map(c => c.name);

                // Pyodide를 사용하여 Python으로 인코딩 매핑 생성
                try {
                    addLog('INFO', 'Pyodide를 사용하여 Python으로 인코딩 매핑 생성 중...');
                    
                    const pyodideModule = await import('./utils/pyodideRunner');
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
                        method || 'label',
                        columnsToEncode.length > 0 ? columnsToEncode : null,
                        ordinalMapping,
                        drop || 'first',
                        handle_unknown || 'ignore',
                        60000 // 타임아웃: 60초
                    );
                    
                    newOutputData = { 
                        type: 'EncoderOutput', 
                        method, 
                        mappings: result.mappings, 
                        columns_to_encode: columnsToEncode,
                        drop: drop,
                        handle_unknown: handle_unknown,
                    };
                    
                    addLog('SUCCESS', 'Python으로 인코딩 매핑 생성 완료');
                } catch (error: any) {
                    const errorMessage = error.message || String(error);
                    addLog('ERROR', `Python EncodeCategorical 실패: ${errorMessage}`);
                    throw new Error(`인코딩 매핑 생성 실패: ${errorMessage}`);
                }
            } else if (module.type === ModuleType.NormalizeData) {
                const inputData = getSingleInputData(module.id) as DataPreview;
                if (!inputData) throw new Error("Input data not available.");
                
                const selections = module.parameters.columnSelections as Record<string, { selected: boolean; }> || {};
                const method = module.parameters.method as NormalizerOutput['method'];
        
                const columnsToNormalize = inputData.columns
                    .filter(col => selections[col.name]?.selected && col.type === 'number')
                    .map(col => col.name);
                
                // Pyodide를 사용하여 Python으로 정규화 통계 계산
                try {
                    addLog('INFO', 'Pyodide를 사용하여 Python으로 정규화 통계 계산 중...');
                    
                    const pyodideModule = await import('./utils/pyodideRunner');
                    const { normalizeDataPython } = pyodideModule;
                    
                    const result = await normalizeDataPython(
                        inputData.rows || [],
                        method || 'MinMax',
                        columnsToNormalize,
                        60000 // 타임아웃: 60초
                    );
                    
                    newOutputData = { type: 'NormalizerOutput', method, stats: result.stats };
                    
                    addLog('SUCCESS', 'Python으로 정규화 통계 계산 완료');
                } catch (error: any) {
                    const errorMessage = error.message || String(error);
                    addLog('ERROR', `Python NormalizeData 실패: ${errorMessage}`);
                    throw new Error(`정규화 통계 계산 실패: ${errorMessage}`);
                }
            } else if (module.type === ModuleType.TransitionData) {
                const inputData = getSingleInputData(module.id) as DataPreview;
                if (!inputData) throw new Error("Input data not available.");

                const transformations = module.parameters.transformations as Record<string, string> || {};
                
                // Pyodide를 사용하여 Python으로 수학적 변환 수행
                try {
                    addLog('INFO', 'Pyodide를 사용하여 Python으로 데이터 변환 수행 중...');
                    
                    const pyodideModule = await import('./utils/pyodideRunner');
                    const { transformDataPython } = pyodideModule;
                    
                    const result = await transformDataPython(
                        inputData.rows || [],
                        transformations,
                        60000 // 타임아웃: 60초
                    );
                    
                    newOutputData = { 
                        type: 'DataPreview', 
                        columns: result.columns, 
                        totalRowCount: result.rows.length, 
                        rows: result.rows 
                    };
                    
                    addLog('SUCCESS', 'Python으로 데이터 변환 완료');
                } catch (error: any) {
                    const errorMessage = error.message || String(error);
                    addLog('ERROR', `Python TransitionData 실패: ${errorMessage}`);
                    throw new Error(`데이터 변환 실패: ${errorMessage}`);
                }
            } else if (module.type === ModuleType.ResampleData) {
                const inputData = getSingleInputData(module.id) as DataPreview;
                if (!inputData) throw new Error("Input data not available.");
    
                const { method, target_column } = module.parameters;
                if (!target_column) throw new Error("Target Column parameter is not set.");
    
                const inputRows = inputData.rows || [];
                if (inputRows.length === 0) {
                     newOutputData = { ...inputData }; // Pass through empty data
                } else {
                    const groups: Record<string, Record<string, any>[]> = {};
                    inputRows.forEach(row => {
                        const key = String(row[target_column]);
                        if (!groups[key]) {
                            groups[key] = [];
                        }
                        groups[key].push(row);
                    });
    
                    let newRows: Record<string, any>[] = [];
    
                    if (method === 'SMOTE') {
                        const counts = Object.values(groups).map(g => g.length);
                        const maxCount = Math.max(...counts);
                        
                        for (const key in groups) {
                            const classRows = groups[key];
                            newRows.push(...classRows);
                            const diff = maxCount - classRows.length;
                            for (let i = 0; i < diff; i++) {
                                // Simple random over-sampling as a simulation of SMOTE
                                newRows.push(classRows[Math.floor(Math.random() * classRows.length)]);
                            }
                        }
    
                    } else if (method === 'NearMiss') {
                        const counts = Object.values(groups).map(g => g.length);
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
                        type: 'DataPreview',
                        columns: inputData.columns,
                        totalRowCount: newRows.length,
                        rows: newRows
                    };
                }
            } else if (module.type === ModuleType.SplitData) {
                const inputData = getSingleInputData(module.id) as DataPreview;
                if (!inputData) throw new Error("Input data not available.");

                const { train_size, random_state, shuffle, stratify, stratify_column } = module.parameters;
                const inputRows = inputData.rows || [];
                
                // Pyodide를 사용하여 브라우저에서 직접 Python 실행
                // Python의 sklearn.train_test_split과 동일한 결과를 보장합니다.
                // 타임아웃 발생 시 Node.js 백엔드로 전환합니다.
                
                let useNodeBackend = false;
                let pyodideErrorForNode = '';
                const totalTimeout = 90000; // 전체 타임아웃: 90초
                const startTime = Date.now();
                
                try {
                    // Pyodide 동적 import
                    addLog('INFO', 'Pyodide를 사용하여 Python으로 데이터 분할 중... (최대 90초)');
                    
                    const pyodideModule = await import('./utils/pyodideRunner');
                    const { splitDataPython } = pyodideModule;
                    
                    // 전체 타임아웃을 포함한 Python 실행 시도
                    const executionPromise = splitDataPython(
                        inputRows,
                        parseFloat(train_size),
                        parseInt(random_state),
                        shuffle === 'True' || shuffle === true,
                        stratify === 'True' || stratify === true,
                        stratify_column || null,
                        60000 // Python 실행 타임아웃: 60초
                    );
                    
                    // 전체 타임아웃 래퍼
                    const timeoutPromise = new Promise<{ trainIndices: number[], testIndices: number[] }>((_, reject) => {
                        const elapsed = Date.now() - startTime;
                        const remaining = totalTimeout - elapsed;
                        if (remaining <= 0) {
                            reject(new Error('전체 실행 타임아웃 (90초 초과)'));
                        } else {
                            setTimeout(() => reject(new Error('전체 실행 타임아웃 (90초 초과)')), remaining);
                        }
                    });
                    
                    const { trainIndices, testIndices } = await Promise.race([
                        executionPromise,
                        timeoutPromise
                    ]);
                    
                    const elapsedTime = Date.now() - startTime;
                    addLog('INFO', `Pyodide 실행 완료 (소요 시간: ${(elapsedTime / 1000).toFixed(1)}초)`);
                    
                    // Python에서 받은 인덱스를 사용하여 데이터 분할
                    const trainRows = trainIndices.map((i: number) => inputRows[i]);
                    const testRows = testIndices.map((i: number) => inputRows[i]);

                    const totalTrainCount = Math.floor(inputData.totalRowCount * parseFloat(train_size));
                    const totalTestCount = inputData.totalRowCount - totalTrainCount;

                    const trainData: DataPreview = { 
                        type: 'DataPreview', 
                        columns: inputData.columns, 
                        totalRowCount: totalTrainCount, 
                        rows: trainRows 
                    };
                    const testData: DataPreview = { 
                        type: 'DataPreview', 
                        columns: inputData.columns, 
                        totalRowCount: totalTestCount, 
                        rows: testRows 
                    };

                    newOutputData = { type: 'SplitDataOutput', train: trainData, test: testData };
                    addLog('SUCCESS', 'Python으로 데이터 분할 완료 (sklearn.train_test_split 사용)');
                } catch (error: any) {
                    const errorMessage = error.message || String(error);
                    const elapsedTime = Date.now() - startTime;
                    
                    // Pyodide 에러 메시지 저장
                    pyodideErrorForNode = errorMessage;
                    
                    // 타임아웃이거나 Pyodide 실패 시 Node.js 백엔드로 전환
                    // "Failed to fetch"는 네트워크 오류이므로 Pyodide 오류로 간주하고 Node.js 백엔드로 전환
                    if (errorMessage.includes('타임아웃') || errorMessage.includes('timeout') || errorMessage.includes('Timeout') || errorMessage.includes('Failed to fetch') || errorMessage.includes('NetworkError')) {
                        addLog('WARN', `Pyodide 타임아웃/오류 발생 (${(elapsedTime / 1000).toFixed(1)}초 경과), Node.js 백엔드로 전환: ${errorMessage}`);
                        useNodeBackend = true;
                    } else {
                        addLog('WARN', `Pyodide 실행 실패 (${(elapsedTime / 1000).toFixed(1)}초 경과), Node.js 백엔드로 전환: ${errorMessage}`);
                        useNodeBackend = true;
                    }
                }
                
                // Node.js 백엔드로 전환
                if (useNodeBackend) {
                    try {
                        addLog('INFO', 'Node.js 백엔드를 통해 Python으로 데이터 분할 중...');
                        
                        // Node.js 백엔드 API 호출 (타임아웃: 30초)
                        const nodeBackendPromise = fetch('/api/split-data', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                data: inputRows,
                                train_size: parseFloat(train_size),
                                random_state: parseInt(random_state),
                                shuffle: shuffle === 'True' || shuffle === true,
                                stratify: stratify === 'True' || stratify === true,
                                stratify_column: stratify_column || null
                            })
                        });
                        
                        const nodeTimeoutPromise = new Promise<Response>((_, reject) => 
                            setTimeout(() => reject(new Error('Node.js 백엔드 타임아웃 (30초 초과)')), 30000)
                        );
                        
                        const response = await Promise.race([nodeBackendPromise, nodeTimeoutPromise]);

                        if (response.ok) {
                            const { trainIndices, testIndices } = await response.json();
                            
                            // Node.js 백엔드에서 받은 인덱스를 사용하여 데이터 분할
                            const trainRows = trainIndices.map((i: number) => inputRows[i]);
                            const testRows = testIndices.map((i: number) => inputRows[i]);

                            const totalTrainCount = Math.floor(inputData.totalRowCount * parseFloat(train_size));
                            const totalTestCount = inputData.totalRowCount - totalTrainCount;

                            const trainData: DataPreview = { 
                                type: 'DataPreview', 
                                columns: inputData.columns, 
                                totalRowCount: totalTrainCount, 
                                rows: trainRows 
                            };
                            const testData: DataPreview = { 
                                type: 'DataPreview', 
                                columns: inputData.columns, 
                                totalRowCount: totalTestCount, 
                                rows: testRows 
                            };

                            newOutputData = { type: 'SplitDataOutput', train: trainData, test: testData };
                            addLog('SUCCESS', 'Node.js 백엔드로 데이터 분할 완료 (sklearn.train_test_split 사용)');
                        } else {
                            const errorText = await response.text();
                            throw new Error(`Node.js 백엔드 응답 오류: ${response.status} - ${errorText}`);
                        }
                    } catch (nodeError: any) {
                        // Node.js 백엔드도 실패하면 에러 발생
                        const nodeErrorMessage = nodeError.message || String(nodeError);
                        
                        // Pyodide 에러 메시지 (이전 catch 블록에서 저장된 에러)
                        const pyodideErrorMsg = typeof pyodideErrorForNode !== 'undefined' ? pyodideErrorForNode : '알 수 없는 Pyodide 오류';
                        
                        // Node.js 백엔드 에러 메시지
                        let nodeErrorMsg = '';
                        if (nodeErrorMessage.includes('Failed to fetch') || nodeErrorMessage.includes('NetworkError') || nodeErrorMessage.includes('ERR_CONNECTION_REFUSED')) {
                            nodeErrorMsg = 'Express 서버(포트 3001)를 찾을 수 없습니다. 터미널에서 "pnpm run server" 또는 "pnpm run dev:full" 명령어로 Express 서버를 실행하세요.';
                        } else if (nodeErrorMessage.includes('타임아웃')) {
                            nodeErrorMsg = `Express 서버 타임아웃: ${nodeErrorMessage}`;
                        } else {
                            nodeErrorMsg = `Express 서버 오류: ${nodeErrorMessage}`;
                        }
                        
                        throw new Error(`데이터 분할 실패: Pyodide와 Express 서버 모두 실패했습니다.\n\nPyodide 오류: ${pyodideErrorMsg}\n\nExpress 서버 오류: ${nodeErrorMsg}\n\n해결 방법:\n1. Express 서버 실행: "pnpm run server" 또는 "pnpm run dev:full"\n2. Python이 설치되어 있고 sklearn, pandas가 설치되어 있는지 확인: "pip install scikit-learn pandas"`);
                    }
                }
            } else if (module.type === ModuleType.Statistics) {
                const inputData = getSingleInputData(module.id) as DataPreview;
                if (!inputData || !inputData.rows) {
                    throw new Error("Input data not available or is of the wrong type.");
                }

                // Pyodide를 사용하여 Python으로 통계 계산
                try {
                    addLog('INFO', 'Pyodide를 사용하여 Python으로 통계 계산 중...');
                    
                    const pyodideModule = await import('./utils/pyodideRunner');
                    const { calculateStatisticsPython } = pyodideModule;
                    
                    const result = await calculateStatisticsPython(
                        inputData.rows,
                        inputData.columns,
                        60000 // 타임아웃: 60초
                    );
                    
                    newOutputData = { 
                        type: 'StatisticsOutput', 
                        stats: result.stats, 
                        correlation: result.correlation, 
                        columns: inputData.columns 
                    };
                    addLog('SUCCESS', 'Python으로 통계 계산 완료');
                } catch (error: any) {
                    const errorMessage = error.message || String(error);
                    addLog('ERROR', `Python 통계 계산 실패: ${errorMessage}`);
                    throw new Error(`통계 계산 실패: ${errorMessage}`);
                }
            } else if (module.type === ModuleType.TrainModel) {
                const modelInputConnection = connections.find(c => c.to.moduleId === module.id && c.to.portName === 'model_in');
                const dataInputConnection = connections.find(c => c.to.moduleId === module.id && c.to.portName === 'data_in');
                
                if (!modelInputConnection || !dataInputConnection) {
                    throw new Error("Both 'model_in' and 'data_in' ports must be connected.");
                }

                const modelSourceModule = currentModules.find(m => m.id === modelInputConnection.from.moduleId);
                if (!modelSourceModule) throw new Error("Model source module not found.");

                const dataSourceModule = currentModules.find(m => m.id === dataInputConnection.from.moduleId);
                if (!dataSourceModule || !dataSourceModule.outputData) throw new Error("Data source module has no output.");

                let inputData: DataPreview | null = null;
                if (dataSourceModule.outputData.type === 'DataPreview') {
                    inputData = dataSourceModule.outputData;
                } else if (dataSourceModule.outputData.type === 'SplitDataOutput') {
                    const portName = dataInputConnection.from.portName;
                    if (portName === 'train_data_out') {
                        inputData = dataSourceModule.outputData.train;
                    } else if (portName === 'test_data_out') {
                        inputData = dataSourceModule.outputData.test;
                    }
                }
                
                if (!inputData) throw new Error("Input data not available or is of the wrong type.");
                
                const { feature_columns, label_column } = module.parameters;
                if (!feature_columns || feature_columns.length === 0 || !label_column) {
                    throw new Error("Feature and label columns are not configured.");
                }
                
                const ordered_feature_columns = inputData.columns
                    .map(c => c.name)
                    .filter(name => feature_columns.includes(name));

                if (ordered_feature_columns.length === 0) {
                    throw new Error("No valid feature columns found in the data.");
                }

                let trainedModelOutput: TrainedModelOutput;
                let intercept = 0;
                const coefficients: Record<string, number> = {};
                const metrics: Record<string, number> = {};

                const modelIsClassification = isClassification(modelSourceModule.type, modelSourceModule.parameters.model_purpose);
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
                
                if (!ordered_feature_columns || ordered_feature_columns.length === 0) {
                    throw new Error("No feature columns specified.");
                }
                
                for (let rowIdx = 0; rowIdx < rows.length; rowIdx++) {
                    const row = rows[rowIdx];
                    if (!row) {
                        continue; // Skip null/undefined rows
                    }
                    
                    const featureRow: number[] = [];
                    let hasValidFeatures = true;
                    
                    for (let colIdx = 0; colIdx < ordered_feature_columns.length; colIdx++) {
                        const col = ordered_feature_columns[colIdx];
                        if (!col) {
                            hasValidFeatures = false;
                            break;
                        }
                        const value = row[col];
                        if (typeof value === 'number' && !isNaN(value) && value !== null && value !== undefined) {
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
                    if (typeof labelValue === 'number' && !isNaN(labelValue) && labelValue !== null && labelValue !== undefined) {
                        X.push(featureRow);
                        y.push(labelValue);
                    }
                }

                if (X.length === 0) {
                    throw new Error(`No valid data rows found after filtering. Checked ${rows.length} rows. Ensure feature columns (${ordered_feature_columns.join(', ')}) and label column (${label_column}) contain valid numeric values.`);
                }
                
                if (X.length < ordered_feature_columns.length) {
                    throw new Error(`Insufficient data: need at least ${ordered_feature_columns.length} samples for ${ordered_feature_columns.length} features, but only found ${X.length} valid rows.`);
                }

                if (modelIsRegression) {
                    // Pyodide를 사용하여 Python으로 Linear Regression 훈련
                    if (modelSourceModule.type === ModuleType.LinearRegression) {
                        const fitIntercept = modelSourceModule.parameters.fit_intercept === 'True';
                        const modelType = modelSourceModule.parameters.model_type || 'LinearRegression';
                        const alpha = parseFloat(modelSourceModule.parameters.alpha) || 1.0;
                        const l1_ratio = parseFloat(modelSourceModule.parameters.l1_ratio) || 0.5;
                        
                        if (X.length < ordered_feature_columns.length) {
                            throw new Error(`Insufficient data: need at least ${ordered_feature_columns.length} samples for ${ordered_feature_columns.length} features, but got ${X.length}.`);
                        }
                        
                        try {
                            addLog('INFO', `Pyodide를 사용하여 Python으로 ${modelType} 모델 훈련 중...`);
                            
                            const pyodideModule = await import('./utils/pyodideRunner');
                            const { fitLinearRegressionPython } = pyodideModule;
                            
                            const fitResult = await fitLinearRegressionPython(
                                X,
                                y,
                                modelType,
                                fitIntercept,
                                alpha,
                                l1_ratio,
                                ordered_feature_columns, // feature columns 전달
                                60000 // 타임아웃: 60초
                            );
                            
                            if (!fitResult.coefficients || fitResult.coefficients.length !== ordered_feature_columns.length) {
                                throw new Error(`Coefficient count mismatch: expected ${ordered_feature_columns.length}, got ${fitResult.coefficients?.length || 0}.`);
                            }
                            
                            intercept = fitResult.intercept;
                            ordered_feature_columns.forEach((col, idx) => {
                                if (fitResult.coefficients[idx] !== undefined) {
                                    coefficients[col] = fitResult.coefficients[idx];
                                } else {
                                    throw new Error(`Missing coefficient for feature ${col} at index ${idx}.`);
                                }
                            });

                            // Python에서 계산된 메트릭 사용
                            const r2Value = typeof fitResult.metrics['R-squared'] === 'number' 
                                ? fitResult.metrics['R-squared'] 
                                : parseFloat(fitResult.metrics['R-squared']);
                            const mseValue = typeof fitResult.metrics['Mean Squared Error'] === 'number'
                                ? fitResult.metrics['Mean Squared Error']
                                : parseFloat(fitResult.metrics['Mean Squared Error']);
                            const rmseValue = typeof fitResult.metrics['Root Mean Squared Error'] === 'number'
                                ? fitResult.metrics['Root Mean Squared Error']
                                : parseFloat(fitResult.metrics['Root Mean Squared Error']);
                            
                            metrics['R-squared'] = parseFloat(r2Value.toFixed(4));
                            metrics['Mean Squared Error'] = parseFloat(mseValue.toFixed(4));
                            metrics['Root Mean Squared Error'] = parseFloat(rmseValue.toFixed(4));
                            
                            addLog('SUCCESS', `Python으로 ${modelType} 모델 훈련 완료`);
                        } catch (error: any) {
                            const errorMessage = error.message || String(error);
                            addLog('ERROR', `Python LinearRegression 훈련 실패: ${errorMessage}`);
                            throw new Error(`모델 훈련 실패: ${errorMessage}`);
                        }
                    } else {
                        // For other regression models, use simulation for now
                        intercept = Math.random() * 10;
                        ordered_feature_columns.forEach(col => {
                            coefficients[col] = Math.random() * 5 - 2.5;
                        });
                        metrics['R-squared'] = 0.65 + Math.random() * 0.25;
                        metrics['Mean Squared Error'] = 150 - Math.random() * 100;
                        metrics['Root Mean Squared Error'] = Math.sqrt(metrics['Mean Squared Error']);
                    }

                } else if (modelIsClassification) {
                    // For classification models, use simulation for now
                    intercept = Math.random() - 0.5;
                    ordered_feature_columns.forEach(col => {
                        coefficients[col] = Math.random() * 2 - 1;
                    });
                    metrics['Accuracy'] = 0.75 + Math.random() * 0.2;
                    metrics['Precision'] = 0.7 + Math.random() * 0.25;
                    metrics['Recall'] = 0.7 + Math.random() * 0.25;
                    metrics['F1-Score'] = 2 * (metrics['Precision'] * metrics['Recall']) / (metrics['Precision'] + metrics['Recall']);
                } else {
                    throw new Error(`Training simulation for model type '${modelSourceModule.type}' is not implemented, or its 'model_purpose' parameter is not set correctly.`);
                }
                
                trainedModelOutput = {
                    type: 'TrainedModelOutput',
                    modelType: modelSourceModule.type,
                    modelPurpose: modelIsClassification ? 'classification' : 'regression',
                    coefficients,
                    intercept,
                    metrics,
                    featureColumns: ordered_feature_columns,
                    labelColumn: label_column,
                };
                
                newOutputData = trainedModelOutput;

            } else if (module.type === ModuleType.ScoreModel) {
                const modelInputConnection = connections.find(c => c.to.moduleId === module.id && c.to.portName === 'model_in');
                const dataInputConnection = connections.find(c => c.to.moduleId === module.id && c.to.portName === 'data_in');
                
                if (!modelInputConnection || !dataInputConnection) {
                    throw new Error("Both 'model_in' and 'data_in' ports must be connected.");
                }

                const trainedModelSourceModule = currentModules.find(m => m.id === modelInputConnection.from.moduleId);
                if (!trainedModelSourceModule || !trainedModelSourceModule.outputData || trainedModelSourceModule.outputData.type !== 'TrainedModelOutput') {
                    throw new Error("A successfully trained model must be connected to 'model_in'.");
                }

                const dataSourceModule = currentModules.find(m => m.id === dataInputConnection.from.moduleId);
                if (!dataSourceModule || !dataSourceModule.outputData) throw new Error("Data source module has no output.");
                
                let inputData: DataPreview | null = null;
                if (dataSourceModule.outputData.type === 'DataPreview') {
                    inputData = dataSourceModule.outputData;
                } else if (dataSourceModule.outputData.type === 'SplitDataOutput') {
                    const portName = dataInputConnection.from.portName;
                    if (portName === 'train_data_out') {
                        inputData = dataSourceModule.outputData.train;
                    } else if (portName === 'test_data_out') {
                        inputData = dataSourceModule.outputData.test;
                    }
                }
                
                if (!inputData) throw new Error("Input data for scoring not available or is of the wrong type.");
                
                const trainedModel = trainedModelSourceModule.outputData;
                const modelIsClassification = isClassification(trainedModel.modelType, trainedModel.modelPurpose);
                const labelColumn = trainedModel.labelColumn;

                // Pyodide를 사용하여 Python으로 예측 수행
                try {
                    addLog('INFO', 'Pyodide를 사용하여 Python으로 모델 예측 수행 중...');
                    
                    const pyodideModule = await import('./utils/pyodideRunner');
                    const { scoreModelPython } = pyodideModule;
                    
                    const result = await scoreModelPython(
                        inputData.rows || [],
                        trainedModel.featureColumns,
                        trainedModel.coefficients,
                        trainedModel.intercept,
                        labelColumn,
                        modelIsClassification ? 'classification' : 'regression',
                        60000 // 타임아웃: 60초
                    );
                    
                    newOutputData = {
                        type: 'DataPreview',
                        columns: result.columns,
                        totalRowCount: inputData.totalRowCount,
                        rows: result.rows
                    };
                    
                    addLog('SUCCESS', 'Python으로 모델 예측 완료');
                } catch (error: any) {
                    const errorMessage = error.message || String(error);
                    addLog('ERROR', `Python ScoreModel 실패: ${errorMessage}`);
                    throw new Error(`모델 예측 실패: ${errorMessage}`);
                }
            } else if (module.type === ModuleType.EvaluateModel) {
                const inputData = getSingleInputData(module.id) as DataPreview;
                if (!inputData) throw new Error("Input data for evaluation not available.");

                const { label_column, prediction_column, model_type } = module.parameters;
                if (!label_column || !prediction_column) {
                    throw new Error("Label and prediction columns must be configured for evaluation.");
                }

                const rows = inputData.rows || [];
                if (rows.length === 0) throw new Error("No rows in input data to evaluate.");

                // Pyodide를 사용하여 Python으로 평가 메트릭 계산
                try {
                    addLog('INFO', 'Pyodide를 사용하여 Python으로 모델 평가 수행 중...');
                    
                    const pyodideModule = await import('./utils/pyodideRunner');
                    const { evaluateModelPython } = pyodideModule;
                    
                    const metrics = await evaluateModelPython(
                        rows,
                        label_column,
                        prediction_column,
                        model_type === 'regression' ? 'regression' : 'classification',
                        60000 // 타임아웃: 60초
                    );
                    
                    addLog('SUCCESS', 'Python으로 모델 평가 완료');
                    
                    newOutputData = {
                        type: 'EvaluationOutput',
                        modelType: model_type,
                        metrics,
                        columns: inputData.columns
                    };
                } catch (error: any) {
                    const errorMessage = error.message || String(error);
                    addLog('ERROR', `Python EvaluateModel 실패: ${errorMessage}`);
                    throw new Error(`모델 평가 실패: ${errorMessage}`);
                }

            } else if (module.type === ModuleType.StatModels) {
                 newOutputData = {
                    type: 'ModelDefinitionOutput',
                    modelFamily: 'statsmodels',
                    modelType: module.parameters.model,
                    parameters: {},
                };
            } else if (module.type === ModuleType.ResultModel) {
                const modelInputConnection = connections.find(c => c.to.moduleId === module.id && c.to.portName === 'model_in');
                const dataInputConnection = connections.find(c => c.to.moduleId === module.id && c.to.portName === 'data_in');
                if (!modelInputConnection || !dataInputConnection) throw new Error("Both 'model_in' and 'data_in' ports must be connected.");
                
                const modelSourceModule = currentModules.find(m => m.id === modelInputConnection.from.moduleId);
                if (!modelSourceModule || modelSourceModule.outputData?.type !== 'ModelDefinitionOutput') throw new Error("A Stat Models module must be connected.");

                const modelDefinition = modelSourceModule.outputData;
                if(modelDefinition.modelFamily !== 'statsmodels') throw new Error("Connected model is not a statsmodels type.");

                const dataSourceModule = currentModules.find(m => m.id === dataInputConnection.from.moduleId);
                if (!dataSourceModule || !dataSourceModule.outputData) throw new Error("Data source module has no output.");

                let inputData: DataPreview | null = null;
                if (dataSourceModule.outputData.type === 'DataPreview') inputData = dataSourceModule.outputData;
                else if (dataSourceModule.outputData.type === 'SplitDataOutput') {
                    const portName = dataInputConnection.from.portName;
                    inputData = portName === 'train_data_out' ? dataSourceModule.outputData.train : dataSourceModule.outputData.test;
                }
                if (!inputData) throw new Error("Input data not available.");

                const { feature_columns, label_column } = module.parameters;
                if (!feature_columns || feature_columns.length === 0 || !label_column) throw new Error("Feature and label columns must be configured.");
                
                const ordered_feature_columns = inputData.columns.map(c => c.name).filter(name => feature_columns.includes(name));
                const all_cols_for_summary = ['const', ...ordered_feature_columns];
                const summaryCoefficients: StatsModelsResultOutput['summary']['coefficients'] = {};

                all_cols_for_summary.forEach(col => {
                    const coef = (Math.random() - 0.5) * 2;
                    const stdErr = Math.random() * 0.1 + 0.01;
                    const t_or_z = coef / stdErr;
                    const p_value = Math.random() * 0.1;
                    const ci_lower = coef - 1.96 * stdErr;
                    const ci_upper = coef + 1.96 * stdErr;
                    summaryCoefficients[col] = {
                        coef, 'std err': stdErr, t: t_or_z, z: t_or_z, 'P>|t|': p_value, 'P>|z|': p_value, '[0.025': ci_lower, '0.975]': ci_upper
                    };
                });
                
                let metrics: StatsModelsResultOutput['summary']['metrics'] = {};
                if (modelDefinition.modelType === 'OLS') {
                    metrics = { 'R-squared': (Math.random() * 0.3 + 0.6).toFixed(3), 'Adj. R-squared': (Math.random() * 0.3 + 0.55).toFixed(3), 'F-statistic': (Math.random() * 50 + 10).toFixed(3), 'Prob (F-statistic)': (Math.random() * 0.01).toExponential(2), 'Log-Likelihood': (-Math.random() * 100 - 50).toFixed(3), 'AIC': (Math.random() * 200 + 120).toFixed(3), 'BIC': (Math.random() * 200 + 140).toFixed(3) };
                } else { // Logit, Poisson, NegativeBinomial, Gamma, Tweedie
                    metrics = { 'Pseudo R-squ.': (Math.random() * 0.3 + 0.1).toFixed(4), 'Log-Likelihood': (-Math.random() * 100 - 200).toFixed(3), 'LL-Null': (-Math.random() * 50 - 250).toFixed(3), 'LLR p-value': (Math.random() * 0.001).toExponential(2) };
                }

                newOutputData = {
                    type: 'StatsModelsResultOutput',
                    modelType: modelDefinition.modelType,
                    summary: { coefficients: summaryCoefficients, metrics },
                    featureColumns: ordered_feature_columns,
                    labelColumn: label_column,
                };

            } else if (module.type === ModuleType.PredictModel) {
                const modelInputConnection = connections.find(c => c.to.moduleId === module.id && c.to.portName === 'model_in');
                const dataInputConnection = connections.find(c => c.to.moduleId === module.id && c.to.portName === 'data_in');
                
                if (!modelInputConnection || !dataInputConnection) {
                    throw new Error("Both 'model_in' and 'data_in' ports must be connected.");
                }

                const modelSourceModule = currentModules.find(m => m.id === modelInputConnection.from.moduleId);
                if (!modelSourceModule || !modelSourceModule.outputData || modelSourceModule.outputData.type !== 'StatsModelsResultOutput') {
                    throw new Error("A successful Result Model module must be connected to 'model_in'.");
                }
                const modelOutput = modelSourceModule.outputData;

                const dataSourceModule = currentModules.find(m => m.id === dataInputConnection.from.moduleId);
                if (!dataSourceModule || !dataSourceModule.outputData) throw new Error("Data source module has no output.");
                
                let inputData: DataPreview | null = null;
                if (dataSourceModule.outputData.type === 'DataPreview') {
                    inputData = dataSourceModule.outputData;
                } else if (dataSourceModule.outputData.type === 'SplitDataOutput') {
                    const portName = dataInputConnection.from.portName;
                    if (portName === 'train_data_out') {
                        inputData = dataSourceModule.outputData.train;
                    } else if (portName === 'test_data_out') {
                        inputData = dataSourceModule.outputData.test;
                    }
                }
                
                if (!inputData) throw new Error("Input data for prediction not available or is of the wrong type.");
                
                const labelColumn = modelOutput.labelColumn;
                const predictColName = `${labelColumn}_Predict`;

                const newColumns: ColumnInfo[] = [...inputData.columns];
                if (!newColumns.some(c => c.name === predictColName)) {
                    newColumns.push({ name: predictColName, type: 'number' });
                }
                const inputRows = inputData.rows || [];

                const newRows = inputRows.map(row => {
                    let linearPredictor = modelOutput.summary.coefficients['const']?.coef ?? 0;
                    
                    for (const feature of modelOutput.featureColumns) {
                        const featureValue = row[feature] as number;
                        const coefficient = modelOutput.summary.coefficients[feature]?.coef;
                        if (typeof featureValue === 'number' && typeof coefficient === 'number') {
                            linearPredictor += featureValue * coefficient;
                        }
                    }

                    let prediction: number;
                    switch(modelOutput.modelType) {
                        case 'OLS':
                            prediction = linearPredictor;
                            break;
                        case 'Logit':
                            prediction = sigmoid(linearPredictor);
                            break;
                        case 'Poisson':
                        case 'NegativeBinomial':
                        case 'Gamma':
                        case 'Tweedie':
                            prediction = Math.exp(linearPredictor);
                            break;
                        default:
                            prediction = NaN;
                    }

                    return {
                        ...row,
                        [predictColName]: parseFloat(prediction.toFixed(4)),
                    };
                });

                newOutputData = {
                    type: 'DataPreview',
                    columns: newColumns,
                    totalRowCount: inputData.totalRowCount,
                    rows: newRows
                };
            } else if (module.type === ModuleType.KMeans || module.type === ModuleType.HierarchicalClustering) {
                const inputData = getSingleInputData(module.id) as DataPreview;
                if (!inputData) throw new Error("Input data not available.");

                const { n_clusters, feature_columns } = module.parameters;
                if (!feature_columns || !Array.isArray(feature_columns) || feature_columns.length === 0) throw new Error("Feature columns must be selected for clustering.");

                const newRows = (inputData.rows || []).map(row => ({
                    ...row,
                    cluster: Math.floor(Math.random() * n_clusters)
                }));
                const newColumns = [...inputData.columns, { name: 'cluster', type: 'number' }];
                
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
                    newOutputData = { type: 'KMeansOutput', clusterAssignments, centroids, model: {} } as KMeansOutput;
                } else {
                    newOutputData = { type: 'HierarchicalClusteringOutput', clusterAssignments } as HierarchicalClusteringOutput;
                }
            } else if (module.type === ModuleType.DBSCAN) {
                const inputData = getSingleInputData(module.id) as DataPreview;
                if (!inputData) throw new Error("Input data not available.");
                const { feature_columns } = module.parameters;
                if (!feature_columns || !Array.isArray(feature_columns) || feature_columns.length === 0) throw new Error("Feature columns must be selected.");

                const n_clusters = Math.floor(Math.random() * 4) + 2; // Random clusters 2-5
                let n_noise = 0;
                
                const newRows = (inputData.rows || []).map(row => {
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
                
                const newColumns = [...inputData.columns, { name: 'cluster', type: 'number' }];
                const clusterAssignments: DataPreview = { ...inputData, columns: newColumns, rows: newRows };
                newOutputData = { type: 'DBSCANOutput', clusterAssignments, n_clusters, n_noise } as DBSCANOutput;

            } else if (module.type === ModuleType.PrincipalComponentAnalysis) {
                const inputData = getSingleInputData(module.id) as DataPreview;
                if (!inputData) throw new Error("Input data not available.");
                const { n_components, feature_columns } = module.parameters;
                let columnsToProcess = feature_columns;
                if (!columnsToProcess || !Array.isArray(columnsToProcess) || columnsToProcess.length === 0) {
                     columnsToProcess = inputData.columns.filter(c => c.type === 'number').map(c => c.name);
                }
                if (columnsToProcess.length < n_components) throw new Error("Number of components cannot be greater than number of features.");

                const newColumns: ColumnInfo[] = Array.from({ length: n_components }, (_, i) => ({
                    name: `PC${i + 1}`,
                    type: 'number'
                }));

                const newRows = (inputData.rows || []).map(row => {
                    const newRow: Record<string, number> = {};
                    for (let i = 0; i < n_components; i++) {
                        newRow[`PC${i + 1}`] = Math.random() * 10 - 5;
                    }
                    return newRow;
                });
                
                // Mock explained variance
                let remainingVariance = 1;
                const explainedVarianceRatio = Array.from({ length: n_components }, (_, i) => {
                    const explained = Math.random() * (remainingVariance / 2) + (i === 0 ? 0.4 : 0);
                    remainingVariance -= explained;
                    return explained;
                });
                
                const transformedData: DataPreview = { type: 'DataPreview', columns: newColumns, totalRowCount: inputData.totalRowCount, rows: newRows };
                newOutputData = { type: 'PCAOutput', transformedData, explainedVarianceRatio } as PCAOutput;

            } else if (module.type === ModuleType.FitLossDistribution) {
                const inputData = getSingleInputData(module.id) as DataPreview;
                if (!inputData) throw new Error("Input data not available.");
                
                const { loss_column, distribution_type } = module.parameters;
                if (!loss_column) throw new Error("Loss column parameter is not set.");
                if (!inputData.columns.some(c => c.name === loss_column && c.type === 'number')) {
                    throw new Error(`Selected loss column '${loss_column}' is not a numeric column in the input data.`);
                }
                
                // Mock fitting process
                let params: Record<string, number> = {};
                if (distribution_type === 'Pareto') {
                    params = { alpha: 1.5 + Math.random() * 0.5, x_m: 100000 + Math.random() * 50000 };
                } else if (distribution_type === 'Lognormal') {
                    params = { mu: 12 + Math.random(), sigma: 1.2 + Math.random() * 0.5 };
                }
                
                newOutputData = {
                    type: 'FittedDistributionOutput',
                    distributionType: distribution_type,
                    parameters: params,
                    lossColumn: loss_column,
                };
            } else if (module.type === ModuleType.GenerateExposureCurve) {
                const distInputConnection = connections.find(c => c.to.moduleId === module.id && c.to.portName === 'dist_in');
                const dataInputConnection = connections.find(c => c.to.moduleId === module.id && c.to.portName === 'data_in');
                if (!distInputConnection || !dataInputConnection) throw new Error("Both 'dist_in' and 'data_in' ports must be connected.");

                const distSourceModule = currentModules.find(m => m.id === distInputConnection.from.moduleId);
                if (!distSourceModule || distSourceModule.outputData?.type !== 'FittedDistributionOutput') throw new Error("A fitted distribution must be connected.");
                const distOutput = distSourceModule.outputData;

                const dataSourceModule = currentModules.find(m => m.id === dataInputConnection.from.moduleId);
                let inputData: DataPreview | null = null;
                 if (dataSourceModule?.outputData?.type === 'DataPreview') inputData = dataSourceModule.outputData;
                if (!inputData) throw new Error("Input data not available from 'data_in'.");
                
                const lossColumn = distOutput.lossColumn;
                const totalExpectedLoss = (inputData.rows || []).reduce((sum, row) => sum + (Number(row[lossColumn]) || 0), 0);

                // Mock curve generation
                const curve: ExposureCurveOutput['curve'] = [];
                for (let i = 0; i <= 20; i++) {
                    const retention = i * (totalExpectedLoss / 10);
                    // Simple mock decay function
                    const loss_pct = Math.exp(-i * 0.25);
                    curve.push({ retention, loss_pct });
                }
                
                newOutputData = { type: 'ExposureCurveOutput', curve, totalExpectedLoss };

            } else if (module.type === ModuleType.PriceXoLLayer) {
                const curveInputConnection = connections.find(c => c.to.moduleId === module.id && c.to.portName === 'curve_in');
                if (!curveInputConnection) throw new Error("'curve_in' port must be connected.");
                
                const curveSourceModule = currentModules.find(m => m.id === curveInputConnection.from.moduleId);
                if (!curveSourceModule || curveSourceModule.outputData?.type !== 'ExposureCurveOutput') throw new Error("An exposure curve must be connected.");
                const curveOutput = curveSourceModule.outputData;

                const { retention, limit, loading_factor } = module.parameters;
                if (typeof retention !== 'number' || typeof limit !== 'number' || typeof loading_factor !== 'number') {
                    throw new Error("Retention, Limit, and Loading Factor must be configured as numbers.");
                }

                // Mock interpolation
                const interpolate = (x: number) => Math.exp(-x / (curveOutput.totalExpectedLoss / 2));
                
                const pct_at_retention = interpolate(retention);
                const pct_at_limit = interpolate(retention + limit);
                
                const layer_loss_pct = pct_at_retention - pct_at_limit;
                const expectedLayerLoss = curveOutput.totalExpectedLoss * layer_loss_pct;
                const rateOnLinePct = limit > 0 ? (expectedLayerLoss / limit) * 100 : 0;
                const premium = expectedLayerLoss * loading_factor;

                newOutputData = { type: 'XoLPriceOutput', retention, limit, expectedLayerLoss, rateOnLinePct, premium };
            } else if (module.type === ModuleType.ApplyThreshold) {
                const inputData = getSingleInputData(module.id) as DataPreview;
                if (!inputData) throw new Error("Input data not available.");
                const { threshold, loss_column } = module.parameters;
                if (!loss_column || typeof threshold !== 'number') throw new Error("Threshold and Loss Column must be set.");
                
                const newRows = (inputData.rows || []).filter(row => (row[loss_column] as number) >= threshold);
                newOutputData = { ...inputData, rows: newRows, totalRowCount: newRows.length };

            } else if (module.type === ModuleType.DefineXolContract) {
                const { deductible, limit, reinstatements, aggDeductible, expenseRatio } = module.parameters;
                newOutputData = { type: 'XolContractOutput', deductible, limit, reinstatements, aggDeductible, expenseRatio };
                
            } else if (module.type === ModuleType.CalculateCededLoss) {
                const dataConnection = connections.find(c => c.to.moduleId === module.id && c.to.portName === 'data_in');
                const contractConnection = connections.find(c => c.to.moduleId === module.id && c.to.portName === 'contract_in');
                if (!dataConnection || !contractConnection) throw new Error("Both data and contract inputs must be connected.");

                const dataSource = currentModules.find(m => m.id === dataConnection.from.moduleId);
                if (!dataSource || dataSource.outputData?.type !== 'DataPreview') throw new Error("Input data is not valid.");
                
                const contractSource = currentModules.find(m => m.id === contractConnection.from.moduleId);
                if (!contractSource || contractSource.outputData?.type !== 'XolContractOutput') throw new Error("Input contract is not valid.");
                
                const inputData = dataSource.outputData;
                const contract = contractSource.outputData;
                const { loss_column } = module.parameters;
                
                const newRows = (inputData.rows || []).map(row => {
                    const loss = row[loss_column] as number;
                    const ceded_loss = Math.min(contract.limit, Math.max(0, loss - contract.deductible));
                    return { ...row, ceded_loss: ceded_loss };
                });
                
                const newColumns = [...inputData.columns];
                if (!newColumns.some(c => c.name === 'ceded_loss')) {
                    newColumns.push({ name: 'ceded_loss', type: 'number' });
                }
                
                newOutputData = { ...inputData, columns: newColumns, rows: newRows };

            } else if (module.type === ModuleType.PriceXolContract) {
                const dataConnection = connections.find(c => c.to.moduleId === module.id && c.to.portName === 'data_in');
                const contractConnection = connections.find(c => c.to.moduleId === module.id && c.to.portName === 'contract_in');
                if (!dataConnection || !contractConnection) throw new Error("Both data and contract inputs must be connected.");

                const dataSource = currentModules.find(m => m.id === dataConnection.from.moduleId);
                if (!dataSource || dataSource.outputData?.type !== 'DataPreview') throw new Error("Input data is not valid.");
                
                const contractSource = currentModules.find(m => m.id === contractConnection.from.moduleId);
                if (!contractSource || contractSource.outputData?.type !== 'XolContractOutput') throw new Error("Input contract is not valid.");
                
                const inputData = dataSource.outputData;
                const contract = contractSource.outputData;
                const { year_column, ceded_loss_column, volatility_loading } = module.parameters;

                const yearlyLosses: Record<string, number> = {};
                (inputData.rows || []).forEach(row => {
                    const year = row[year_column];
                    const cededLoss = row[ceded_loss_column] as number;
                    if (year && typeof cededLoss === 'number') {
                        yearlyLosses[year] = (yearlyLosses[year] || 0) + cededLoss;
                    }
                });
                
                const yearlyLossValues = Object.values(yearlyLosses);
                if (yearlyLossValues.length === 0) throw new Error("No yearly losses to analyze. Check input data and column names.");

                const sum = yearlyLossValues.reduce((a, b) => a + b, 0);
                const mean = sum / yearlyLossValues.length;
                const stdDev = Math.sqrt(yearlyLossValues.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b, 0) / (yearlyLossValues.length -1 || 1));

                const expectedLoss = mean;
                const volatilityMargin = stdDev * (volatility_loading / 100);
                const purePremium = expectedLoss + volatilityMargin;
                const expenseLoading = purePremium / (1 - contract.expenseRatio) - purePremium;
                const finalPremium = purePremium + expenseLoading;

                newOutputData = { type: 'FinalXolPriceOutput', expectedLoss, stdDev, volatilityMargin, purePremium, expenseLoading, finalPremium };
            } else {
                const inputConnection = connections.find(c => c.to.moduleId === module.id);
                if (inputConnection) {
                    const sourceModule = currentModules.find(sm => sm.id === inputConnection.from.moduleId);
                    if (sourceModule?.outputData?.type === 'DataPreview') {
                       newOutputData = sourceModule.outputData;
                    } else if (sourceModule?.status !== ModuleStatus.Success) {
                        throw new Error(`Upstream module [${sourceModule?.name}] did not run successfully.`);
                    }
                }
            }
            newStatus = ModuleStatus.Success;
            logLevel = 'SUCCESS';
            logMessage = `Module [${moduleName}] executed successfully.`;
        } catch(error: any) {
            newStatus = ModuleStatus.Error;
            logLevel = 'ERROR';
            logMessage = `Module [${moduleName}] failed: ${error.message}`;
        }
        
        const finalModuleState = { ...module, status: newStatus, outputData: newOutputData };
        
        // Update the mutable array for the current run
        const moduleIndex = currentModules.findIndex(m => m.id === moduleId);
        if (moduleIndex !== -1) {
            currentModules[moduleIndex] = finalModuleState;
        }
        
        // If TrainModel succeeded, mark the connected model definition module as Success
        // If TrainModel is set to Pending, also mark the connected model definition module as Pending
        let modelDefinitionModuleId: string | null = null;
        let shouldUpdateModelDefinition = false;
        let modelDefinitionNewStatus: ModuleStatus | null = null;
        
        if (module.type === ModuleType.TrainModel) {
            const modelInputConnection = connections.find(c => c.to.moduleId === moduleId && c.to.portName === 'model_in');
            if (modelInputConnection) {
                modelDefinitionModuleId = modelInputConnection.from.moduleId;
                if (newStatus === ModuleStatus.Success) {
                    shouldUpdateModelDefinition = true;
                    modelDefinitionNewStatus = ModuleStatus.Success;
                } else if (newStatus === ModuleStatus.Pending || newStatus === ModuleStatus.Error) {
                    // When TrainModel becomes Pending or Error, mark connected model definition module as Pending
                    shouldUpdateModelDefinition = true;
                    modelDefinitionNewStatus = ModuleStatus.Pending;
                }
            }
        }
        
        // Update React's state for the UI
        setModules(prev => prev.map(m => {
            if (m.id === moduleId) {
                return finalModuleState;
            }
            // Update connected model definition module status when TrainModel status changes
            if (shouldUpdateModelDefinition && modelDefinitionModuleId && m.id === modelDefinitionModuleId && MODEL_DEFINITION_TYPES.includes(m.type) && modelDefinitionNewStatus) {
                return { ...m, status: modelDefinitionNewStatus, outputData: modelDefinitionNewStatus === ModuleStatus.Pending ? undefined : m.outputData };
            }
            return m;
        }));
        addLog(logLevel, logMessage);

        if (newStatus === ModuleStatus.Error) {
            break;
        }
    }
  };

  const handleRunAll = () => {
    const rootNodes = modules.filter(m => !connections.some(c => c.to.moduleId === m.id));
    if (rootNodes.length > 0) {
      addLog('INFO', `Project Run All started with ${rootNodes.length} root node(s)...`);
      setModules(prev => prev.map(m => ({ ...m, status: ModuleStatus.Pending, outputData: undefined })));
      // Run all modules starting from all root nodes
      // Pass the first root node ID, but runAll=true will traverse all root nodes
      runSimulation(rootNodes[0].id, true);
    } else if (modules.length > 0) {
        addLog('WARN', 'Circular dependency or no root nodes found. Starting from all modules.');
        setModules(prev => prev.map(m => ({ ...m, status: ModuleStatus.Pending, outputData: undefined })));
        // When no root nodes, runAll will traverse all modules
        runSimulation(modules[0].id, true);
    } else {
      addLog('WARN', 'No modules on canvas to run.');
    }
  };
  
  const adjustScale = (delta: number) => {
      setScale(prev => Math.max(0.2, Math.min(2, prev + delta)));
  }

  const selectedModule = modules.find(m => m.id === selectedModuleIds[selectedModuleIds.length - 1]) || null;

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
        const activeElement = document.activeElement;
        const isEditingText = activeElement && (activeElement.tagName === 'INPUT' || activeElement.tagName === 'TEXTAREA' || (activeElement as HTMLElement).isContentEditable);
        if (isEditingText) return;

        if (e.ctrlKey || e.metaKey) {
            if (e.key === 'a' || e.key === 'A') {
                e.preventDefault();
                // Ctrl+A: 모든 모듈 선택
                const allModuleIds = modules.map(m => m.id);
                setSelectedModuleIds(allModuleIds);
                addLog('INFO', `모든 모듈 선택됨 (${allModuleIds.length}개)`);
            } else if (e.key === 'z') {
                e.preventDefault();
                undo();
            } else if (e.key === 'y') {
                e.preventDefault();
                redo();
            } else if (e.key === 'c') {
                if (selectedModuleIds.length > 0) {
                    e.preventDefault();
                    pasteOffset.current = 0;
                    const selectedModules = modules.filter(m => selectedModuleIds.includes(m.id));
                    const selectedIdsSet = new Set(selectedModuleIds);
                    const internalConnections = connections.filter(
                        c => selectedIdsSet.has(c.from.moduleId) && selectedIdsSet.has(c.to.moduleId)
                    );
                    setClipboard({
                        modules: JSON.parse(JSON.stringify(selectedModules)),
                        connections: JSON.parse(JSON.stringify(internalConnections)),
                    });
                    addLog('INFO', `${selectedModuleIds.length} module(s) copied to clipboard.`);
                }
            } else if (e.key === 'v') {
                e.preventDefault();
                if (clipboard) {
                    pasteOffset.current += 30;
                    const idMap: Record<string, string> = {};
                    const newModules = clipboard.modules.map(mod => {
                        const newId = `${mod.type}-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`;
                        idMap[mod.id] = newId;
                        return {
                            ...mod,
                            id: newId,
                            position: { x: mod.position.x + pasteOffset.current, y: mod.position.y + pasteOffset.current },
                            status: ModuleStatus.Pending,
                            outputData: undefined,
                        };
                    });
                    const newConnections = clipboard.connections.map(conn => ({
                        ...conn,
                        id: `conn-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
                        from: { ...conn.from, moduleId: idMap[conn.from.moduleId] },
                        to: { ...conn.to, moduleId: idMap[conn.to.moduleId] },
                    }));

                    setModules(prev => [...prev, ...newModules]);
                    setConnections(prev => [...prev, ...newConnections]);
                    setSelectedModuleIds(newModules.map(m => m.id));
                    addLog('INFO', `${newModules.length} module(s) pasted from clipboard.`);
                }
            }
        } else if (selectedModuleIds.length > 0) {
            if (e.key === 'Delete' || e.key === 'Backspace') {
                e.preventDefault();
                deleteModules([...selectedModuleIds]);
            }
        }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedModuleIds, undo, redo, setModules, setConnections, setSelectedModuleIds, modules, connections, clipboard, deleteModules, addLog]);

    useEffect(() => {
        const handleWindowMouseMove = (e: globalThis.MouseEvent) => {
            if (isDraggingControlPanel.current && canvasContainerRef.current) {
                const containerRect = canvasContainerRef.current.getBoundingClientRect();
                setControlPanelPos({
                    x: e.clientX - containerRect.left - controlPanelDragOffset.current.x,
                    y: e.clientY - containerRect.top - controlPanelDragOffset.current.y,
                });
            }
        };

        const handleWindowMouseUp = () => {
            isDraggingControlPanel.current = false;
        };

        window.addEventListener('mousemove', handleWindowMouseMove);
        window.addEventListener('mouseup', handleWindowMouseUp);

        return () => {
            window.removeEventListener('mousemove', handleWindowMouseMove);
            window.removeEventListener('mouseup', handleWindowMouseUp);
        };
    }, []);

    const handleControlPanelMouseDown = (e: React.MouseEvent) => {
        e.preventDefault();
        e.stopPropagation(); // Prevent canvas panning
        isDraggingControlPanel.current = true;
        
        const rect = e.currentTarget.getBoundingClientRect();
        controlPanelDragOffset.current = {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
        
        // If it's the first drag (currently centered via CSS), set explicit coordinates relative to container
        if (!controlPanelPos && canvasContainerRef.current) {
             const containerRect = canvasContainerRef.current.getBoundingClientRect();
            setControlPanelPos({ 
                x: rect.left - containerRect.left, 
                y: rect.top - containerRect.top 
            });
        }
    };

  return (
    <div className="bg-gray-900 text-white h-screen w-screen flex flex-col overflow-hidden">
        {isAiGenerating && (
            <div className="fixed inset-0 bg-black bg-opacity-70 flex flex-col items-center justify-center z-50">
                <div role="status">
                    <svg aria-hidden="true" className="w-12 h-12 text-gray-200 animate-spin fill-blue-600" viewBox="0 0 100 101" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z" fill="currentColor"/>
                        <path d="M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0492C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z" fill="currentFill"/>
                    </svg>
                    <span className="sr-only">Loading...</span>
                </div>
                <p className="mt-4 text-lg font-semibold text-white">AI가 최적의 파이프라인을 설계하고 있습니다...</p>
            </div>
        )}

        <header className="flex items-center justify-between px-4 py-2 bg-gray-900 border-b border-gray-700 flex-shrink-0 z-20">
            <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                    <button onClick={() => setIsLeftPanelVisible(v => !v)} className="p-1" aria-label="Toggle modules panel">
                        <Bars3Icon className="h-6 w-6"/>
                    </button>
                    <LogoIcon className="h-6 w-6 text-blue-400 hidden md:inline-block" />
                    <h1 className="text-xl font-bold text-blue-300 tracking-wide hidden md:inline-block">Insure Auto Flow</h1>
                </div>
                <div className="hidden md:flex items-center gap-2">
                    <span className="text-gray-600">|</span>
                    {isEditingProjectName ? (
                        <input
                            value={projectName}
                            onChange={e => setProjectName(e.target.value)}
                            onBlur={() => setIsEditingProjectName(false)}
                            onKeyDown={e => {
                                if (e.key === 'Enter' || e.key === 'Escape') {
                                    setIsEditingProjectName(false);
                                }
                            }}
                            className="bg-gray-800 text-lg font-semibold text-white px-2 py-1 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                            autoFocus
                        />
                    ) : (
                        <h2
                            onClick={() => setIsEditingProjectName(true)}
                            className="text-lg font-semibold text-gray-300 hover:bg-gray-700 px-2 py-1 rounded-md cursor-pointer"
                            title="Click to edit project name"
                        >
                            {projectName}
                        </h2>
                    )}
                </div>
            </div>
             <div className="flex-1 px-2 md:px-8 flex flex-col items-center justify-center gap-2">
                <button
                    onClick={() => setIsGoalModalOpen(true)}
                    className="flex items-center gap-2 px-2 py-1 text-[10px] bg-purple-600 hover:bg-purple-700 rounded-md font-semibold transition-colors"
                    title="Generate pipeline from a goal"
                >
                    <SparklesIcon className="h-3 w-3" />
                    AI로 파이프라인 생성하기
                </button>
                 <button
                    onClick={() => setIsDataModalOpen(true)}
                    className="flex items-center gap-2 px-2 py-1 text-[10px] bg-indigo-600 hover:bg-indigo-700 rounded-md font-semibold transition-colors"
                    title="Generate pipeline from data"
                >
                    <SparklesIcon className="h-3 w-3" />
                    AI로 데이터 분석 실행하기
                </button>
            </div>
            <div className="flex items-center gap-2">
                 <div className="flex items-center gap-1 md:gap-2 overflow-x-auto scrollbar-hide">
                     <button
                        onClick={undo}
                        disabled={!canUndo}
                        className="p-1 md:p-1.5 text-gray-300 hover:bg-gray-700 rounded-md disabled:text-gray-600 disabled:cursor-not-allowed transition-colors flex-shrink-0"
                        title="Undo (Ctrl+Z)"
                    >
                        <ArrowUturnLeftIcon className="h-4 w-4 md:h-5 md:w-5" />
                    </button>
                    <button
                        onClick={redo}
                        disabled={!canRedo}
                        className="p-1 md:p-1.5 text-gray-300 hover:bg-gray-700 rounded-md disabled:text-gray-600 disabled:cursor-not-allowed transition-colors flex-shrink-0"
                        title="Redo (Ctrl+Y)"
                    >
                        <ArrowUturnRightIcon className="h-4 w-4 md:h-5 md:w-5" />
                    </button>
                    <button onClick={handleSetFolder} className="flex items-center gap-1 md:gap-2 px-2 md:px-3 py-1 md:py-1.5 text-[10px] md:text-xs bg-gray-700 hover:bg-gray-600 rounded-md font-semibold transition-colors flex-shrink-0" title="Set Save Folder">
                        <FolderOpenIcon className="h-3 w-3 md:h-4 md:w-4" />
                        <span className="hidden sm:inline">Set Folder</span>
                    </button>
                    <button onClick={handleLoadPipeline} className="flex items-center gap-1 md:gap-2 px-2 md:px-3 py-1 md:py-1.5 text-[10px] md:text-xs bg-gray-700 hover:bg-gray-600 rounded-md font-semibold transition-colors flex-shrink-0" title="Load Pipeline">
                        <FolderOpenIcon className="h-3 w-3 md:h-4 md:w-4" />
                        <span className="hidden sm:inline">Load</span>
                    </button>
                    <div className="relative flex-shrink-0" ref={sampleMenuRef}>
                        <button 
                            onClick={() => setIsSampleMenuOpen(!isSampleMenuOpen)} 
                            className="flex items-center gap-1 md:gap-2 px-2 md:px-3 py-1 md:py-1.5 text-[10px] md:text-xs bg-blue-600 hover:bg-blue-700 rounded-md font-semibold transition-colors" 
                            title="Load Sample Model"
                        >
                            <SparklesIcon className="h-3 w-3 md:h-4 md:w-4" />
                            <span className="hidden sm:inline">Samples</span>
                        </button>
                        {isSampleMenuOpen && (
                            <div className="absolute top-full left-0 mt-1 bg-gray-800 border border-gray-700 rounded-md shadow-lg z-50 min-w-[200px]">
                                {SAMPLE_MODELS.map((sample: any) => (
                                    <button
                                        key={sample.name}
                                        onClick={() => handleLoadSample(sample.name)}
                                        className="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 first:rounded-t-md last:rounded-b-md transition-colors"
                                    >
                                        {sample.name}
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>
                     <button 
                        onClick={handleSavePipeline} 
                        disabled={!isDirty}
                        className={`flex items-center gap-1 md:gap-2 px-2 md:px-3 py-1 md:py-1.5 text-[10px] md:text-xs rounded-md font-semibold transition-colors flex-shrink-0 ${!isDirty ? 'bg-gray-600 cursor-not-allowed opacity-50' : 'bg-gray-700 hover:bg-gray-600'}`} 
                        title="Save Pipeline"
                    >
                        {saveButtonText === 'Save' ? <CodeBracketIcon className="h-3 w-3 md:h-4 md:w-4" /> : <CheckIcon className="h-3 w-3 md:h-4 md:w-4" />}
                        <span className="hidden sm:inline">{saveButtonText}</span>
                    </button>
                </div>
                 <button onClick={handleRunAll} className="flex items-center gap-2 px-3 py-1.5 text-sm bg-green-600 hover:bg-green-500 rounded-md font-bold text-white transition-colors">
                    <PlayIcon className="h-4 w-4" />
                    Run All
                </button>
                <button onClick={handleToggleRightPanel} className="p-1.5 text-gray-300 hover:bg-gray-700 rounded-md transition-colors" title="Toggle Properties Panel">
                    <CogIcon className="h-5 w-5" />
                </button>
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
                  suggestion={suggestion}
                  onAcceptSuggestion={acceptSuggestion}
                  onClearSuggestion={clearSuggestion}
                  onStartSuggestion={handleSuggestModule}
                  areUpstreamModulesReady={areUpstreamModulesReady}
                />
                 <div 
                    onMouseDown={handleControlPanelMouseDown}
                    className={`absolute bg-gray-800 rounded-lg p-1 flex items-center gap-1 shadow-lg cursor-move z-50 ${!controlPanelPos ? 'bottom-4 left-1/2 -translate-x-1/2' : ''}`}
                    style={controlPanelPos ? { left: controlPanelPos.x, top: controlPanelPos.y } : {}}
                >
                    <button onClick={(e) => {e.stopPropagation(); adjustScale(-0.1)}} className="p-1.5 hover:bg-gray-700 rounded-md" title="Zoom Out"><MinusIcon className="w-5 h-5"/></button>
                    <button onClick={(e) => {e.stopPropagation(); handleRearrangeModules()}} className="p-1.5 hover:bg-gray-700 rounded-md" title="Rearrange Modules"><Squares2X2Icon className="w-5 h-5"/></button>
                    <button onClick={(e) => {e.stopPropagation(); handleFitToView()}} className="p-1.5 hover:bg-gray-700 rounded-md" title="Fit to View"><ArrowsPointingOutIcon className="w-5 h-5"/></button>
                    <button onClick={(e) => {e.stopPropagation(); setScale(1); setPan({ x: 0, y: 0 }); }} className="px-2 py-1 text-sm font-semibold hover:bg-gray-700 rounded-md" title="Reset View">{Math.round(scale * 100)}%</button>
                    <button onClick={(e) => {e.stopPropagation(); adjustScale(0.1)}} className="p-1.5 hover:bg-gray-700 rounded-md" title="Zoom In"><PlusIcon className="w-5 h-5"/></button>
                </div>
            </main>

            {/* -- Unified Side Panels -- */}
            <div className={`absolute top-0 left-0 h-full z-10 transition-transform duration-300 ease-in-out ${isLeftPanelVisible ? 'translate-x-0' : '-translate-x-full'}`}>
                <Toolbox onModuleDoubleClick={handleModuleToolboxDoubleClick} />
            </div>

            <div className={`absolute top-0 right-0 h-full z-10 transition-transform duration-300 ease-in-out ${isRightPanelVisible ? 'translate-x-0' : 'translate-x-full'}`}>
              <div className="flex h-full" style={{ width: `${rightPanelWidth}px` }}>
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
            plan={aiPlan || ''}
        />

        {/* -- Modals -- */}
        {viewingDataForModule && (viewingDataForModule.outputData?.type === 'DataPreview' || viewingDataForModule.outputData?.type === 'KMeansOutput' || viewingDataForModule.outputData?.type === 'HierarchicalClusteringOutput' || viewingDataForModule.outputData?.type === 'DBSCANOutput' || viewingDataForModule.outputData?.type === 'PCAOutput') && (
            <DataPreviewModal 
                module={viewingDataForModule}
                projectName={projectName}
                onClose={handleCloseModal}
            />
        )}
        {viewingDataForModule && viewingDataForModule.outputData?.type === 'StatisticsOutput' && (
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
        {viewingEvaluation && (
            <EvaluationPreviewModal
                module={viewingEvaluation}
                onClose={handleCloseModal}
            />
        )}
    </div>
  );
};

export default App;