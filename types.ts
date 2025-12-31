// fix: Removed erroneous import of ModuleType from './App' to resolve circular dependency and declaration merge errors.
export enum ModuleType {
  LoadData = "LoadData",
  Statistics = "Statistics", // New module type
  SelectData = "SelectData",
  HandleMissingValues = "HandleMissingValues",
  TransformData = "TransformData",
  EncodeCategorical = "EncodeCategorical",
  NormalizeData = "NormalizeData",
  TransitionData = "TransitionData",
  ResampleData = "ResampleData",
  SplitData = "SplitData",

  // Supervised Learning Models
  LinearRegression = "LinearRegression",
  LogisticRegression = "LogisticRegression",
  PoissonRegression = "PoissonRegression",
  NegativeBinomialRegression = "NegativeBinomialRegression",
  DecisionTree = "DecisionTree",
  RandomForest = "RandomForest",
  NeuralNetwork = "NeuralNetwork",
  SVM = "SVM",
  LinearDiscriminantAnalysis = "LinearDiscriminantAnalysis",
  NaiveBayes = "NaiveBayes",
  KNN = "KNN",

  // Model Operations
  TrainModel = "TrainModel",
  ScoreModel = "ScoreModel",
  EvaluateModel = "EvaluateModel",

  // Unsupervised Learning
  KMeans = "KMeans",
  HierarchicalClustering = "HierarchicalClustering",
  DBSCAN = "DBSCAN",
  PrincipalComponentAnalysis = "PrincipalComponentAnalysis",

  // Traditional Analysis - Statsmodels Models
  OLSModel = "OLSModel",
  LogisticModel = "LogisticModel",
  PoissonModel = "PoissonModel",
  QuasiPoissonModel = "QuasiPoissonModel",
  NegativeBinomialModel = "NegativeBinomialModel",
  DiversionChecker = "DiversionChecker",
  EvaluateStat = "EvaluateStat",

  // Legacy/StatModels - Keeping for advanced models (Gamma, Tweedie)
  StatModels = "StatModels",
  ResultModel = "ResultModel",
  PredictModel = "PredictModel",

  // Reinsurance Modules
  FitLossDistribution = "FitLossDistribution",
  GenerateExposureCurve = "GenerateExposureCurve",
  PriceXoLLayer = "PriceXoLLayer",
  XolLoading = "XolLoading",
  ApplyThreshold = "ApplyThreshold",
  DefineXolContract = "DefineXolContract",
  CalculateCededLoss = "CalculateCededLoss",
  PriceXolContract = "PriceXolContract",

  // Deprecating these
  LogisticTradition = "LogisticTradition",

  // Shape Types
  TextBox = "TextBox",
  GroupBox = "GroupBox",
}

export enum ModuleStatus {
  Pending = "Pending",
  Running = "Running",
  Success = "Success",
  Error = "Error",
}

export interface Port {
  name: string;
  type:
    | "data"
    | "model"
    | "evaluation"
    | "distribution"
    | "curve"
    | "contract"
    | "handler";
}

export interface ColumnInfo {
  name: string;
  type: string;
}

export interface DataPreview {
  type: "DataPreview"; // Differentiator
  columns: ColumnInfo[];
  totalRowCount: number;
  rows?: Record<string, any>[];
}

// Types for the new Statistics module output
export interface DescriptiveStats {
  [columnName: string]: {
    count: number;
    mean: number;
    std: number;
    min: number;
    "25%": number;
    "50%": number; // median
    "75%": number;
    max: number;
    variance: number;
    nulls: number;
    mode: number | string;
    skewness: number;
    kurtosis: number;
  };
}

export interface CorrelationMatrix {
  [column1: string]: {
    [column2: string]: number;
  };
}

export interface StatisticsOutput {
  type: "StatisticsOutput"; // Differentiator
  stats: DescriptiveStats;
  correlation: CorrelationMatrix;
  columns: ColumnInfo[]; // Keep original column info
}

export interface SplitDataOutput {
  type: "SplitDataOutput";
  train: DataPreview;
  test: DataPreview;
}

export interface TuningCandidateScore {
  params: Record<string, number>;
  score: number;
}

export interface TuningSummary {
  enabled: boolean;
  strategy?: "grid";
  bestParams?: Record<string, number>;
  bestScore?: number;
  scoringMetric?: string;
  candidates?: TuningCandidateScore[];
}

export interface TrainedModelOutput {
  type: "TrainedModelOutput";
  modelType: ModuleType;
  modelPurpose?: "classification" | "regression";
  coefficients: Record<string, number>;
  intercept: number;
  metrics: Record<string, number>;
  featureColumns: string[];
  labelColumn: string;
  tuningSummary?: TuningSummary;
  statsModelsResult?: StatsModelsResultOutput; // statsmodels 결과 (포아송/음이항 회귀용)
  trainingData?: any[]; // Decision Tree plot_tree 생성을 위한 훈련 데이터 (선택적)
  modelParameters?: Record<string, any>; // 모델 파라미터 (Decision Tree 등)
}

export type StatsModelFamily =
  | "OLS"
  | "Logistic"
  | "Logit"
  | "Poisson"
  | "QuasiPoisson"
  | "NegativeBinomial"
  | "Gamma"
  | "Tweedie";

export interface ModelDefinitionOutput {
  type: "ModelDefinitionOutput";
  modelFamily: "statsmodels";
  modelType: StatsModelFamily;
  parameters: Record<string, any>;
}

export interface StatsModelsResultOutput {
  type: "StatsModelsResultOutput";
  modelType: StatsModelFamily;
  summary: {
    coefficients: Record<
      string,
      {
        coef: number;
        "std err": number;
        t?: number;
        z?: number;
        "P>|t|"?: number;
        "P>|z|"?: number;
        "[0.025": number;
        "0.975]": number;
      }
    >;
    metrics: Record<string, string | number>;
  };
  featureColumns: string[];
  labelColumn: string;
}

export interface ConfusionMatrix {
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export interface ThresholdMetric {
  threshold: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export interface EvaluationOutput {
  type: "EvaluationOutput";
  modelType: "classification" | "regression";
  metrics: Record<string, number | string>;
  confusionMatrix?: ConfusionMatrix;
  threshold?: number;
  thresholdMetrics?: ThresholdMetric[]; // 여러 threshold에 대한 precision/recall
}

export interface DiversionCheckerOutput {
  type: "DiversionCheckerOutput";
  phi: number;
  recommendation: "Poisson" | "QuasiPoisson" | "NegativeBinomial";
  poissonAic: number | null;
  negativeBinomialAic: number | null;
  aicComparison: string | null;
  cameronTrivediCoef: number;
  cameronTrivediPvalue: number;
  cameronTrivediConclusion: string;
  methodsUsed: string[];
  results: {
    phi: number;
    phi_interpretation: string;
    recommendation: string;
    poisson_aic: number | null;
    negative_binomial_aic: number | null;
    cameron_trivedi_coef: number;
    cameron_trivedi_pvalue: number;
    cameron_trivedi_conclusion: string;
  };
}

export interface EvaluateStatOutput {
  type: "EvaluateStatOutput";
  modelType: string;
  metrics: Record<string, number | string>;
  residuals?: number[];
  deviance?: number;
  pearsonChi2?: number;
  dispersion?: number;
  aic?: number;
  bic?: number;
  logLikelihood?: number;
}

// --- New Unsupervised Learning Outputs ---
export interface KMeansOutput {
  type: "KMeansOutput";
  clusterAssignments: DataPreview; // Data with an added 'cluster' column
  centroids: Record<string, number>[];
  model: any; // To hold inertia_ or other model properties
}

export interface HierarchicalClusteringOutput {
  type: "HierarchicalClusteringOutput";
  clusterAssignments: DataPreview; // Data with an added 'cluster' column
}

export interface DBSCANOutput {
  type: "DBSCANOutput";
  clusterAssignments: DataPreview; // Data with an added 'cluster' column
  n_clusters: number;
  n_noise: number;
}

export interface PCAOutput {
  type: "PCAOutput";
  transformedData: DataPreview;
  explainedVarianceRatio: number[];
}

// --- Reinsurance Module Outputs ---
export type LossDistributionType = "Pareto" | "Lognormal";

export interface FittedDistributionOutput {
  type: "FittedDistributionOutput";
  distributionType: LossDistributionType;
  parameters: Record<string, number>;
  lossColumn: string;
}

export interface ExposureCurveOutput {
  type: "ExposureCurveOutput";
  curve: { retention: number; loss_pct: number }[];
  totalExpectedLoss: number;
}

export interface XoLPriceOutput {
  type: "XoLPriceOutput";
  retention: number;
  limit: number;
  expectedLayerLoss: number;
  rateOnLinePct: number;
  premium: number;
}

export interface XolContractOutput {
  type: "XolContractOutput";
  deductible: number;
  limit: number;
  reinstatements: number;
  aggDeductible: number;
  expenseRatio: number;
}

export interface FinalXolPriceOutput {
  type: "FinalXolPriceOutput";
  expectedLoss: number;
  stdDev: number;
  volatilityMargin: number;
  purePremium: number;
  expenseLoading: number;
  finalPremium: number;
}

export interface MissingHandlerOutput {
  type: "MissingHandlerOutput";
  method: "remove_row" | "impute" | "knn";
  // For impute
  strategy?: "mean" | "median" | "mode";
  // For KNN
  n_neighbors?: number;
  metric?: string;
  // For all methods that are not row removal, we need the values computed from the training set
  imputation_values: Record<string, number | string>; // e.g. { 'Age': 29.5, 'Embarked': 'S' }
}

export interface EncoderOutput {
  type: "EncoderOutput";
  method: "label" | "one_hot" | "ordinal";
  mappings: Record<string, Record<string, number> | string[]>;
  columns_to_encode: string[];
  // one-hot params that are passed through
  drop?: "first" | "if_binary" | null;
  handle_unknown?: "error" | "ignore";
}

export interface NormalizerOutput {
  type: "NormalizerOutput";
  method: "MinMax" | "StandardScaler" | "RobustScaler";
  stats: Record<
    string,
    {
      min?: number;
      max?: number;
      mean?: number;
      stdDev?: number;
      median?: number;
      iqr?: number;
    }
  >;
}

export interface CanvasModule {
  id: string;
  name: string;
  type: ModuleType;
  position: { x: number; y: number };
  status: ModuleStatus;
  parameters: Record<string, any>;
  inputs: Port[];
  outputs: Port[];
  outputData?:
    | DataPreview
    | StatisticsOutput
    | SplitDataOutput
    | TrainedModelOutput
    | ModelDefinitionOutput
    | StatsModelsResultOutput
    | FittedDistributionOutput
    | ExposureCurveOutput
    | XoLPriceOutput
    | XolContractOutput
    | FinalXolPriceOutput
    | EvaluationOutput
    | DiversionCheckerOutput
    | EvaluateStatOutput
    | KMeansOutput
    | HierarchicalClusteringOutput
    | PCAOutput
    | DBSCANOutput
    | MissingHandlerOutput
    | EncoderOutput
    | NormalizerOutput;
  // Shape-specific properties
  shapeData?: {
    // For TextBox
    text?: string;
    width?: number;
    height?: number;
    fontSize?: number;
    // For GroupBox
    moduleIds?: string[]; // IDs of modules in this group
    bounds?: { x: number; y: number; width: number; height: number }; // Bounding box of the group
  };
}

export interface Connection {
  id: string;
  from: { moduleId: string; portName: string };
  to: { moduleId: string; portName: string };
}
