// 공유 가능한 Samples 목록
// 이 파일은 커밋/푸시에 포함되어 모든 사용자가 공유할 수 있습니다.

import { ModuleType } from "./types";

export interface SavedSample {
  name: string;
  modules: Array<{
    type: ModuleType;
    position: { x: number; y: number };
    name: string;
    parameters?: Record<string, any>;
  }>;
  connections: Array<{
    fromModuleIndex: number;
    fromPort: string;
    toModuleIndex: number;
    toPort: string;
  }>;
}

export const SAVED_SAMPLES: SavedSample[] = [
  {
    name: "Linear Regression-1",
    modules: [
      {
        type: ModuleType.LoadData,
        position: { x: 94, y: 85 },
        name: "Load Data",
        parameters: {
          source: "your-data-source.csv",
        },
      },
      {
        type: ModuleType.SelectData,
        position: { x: 100, y: 250 },
        name: "Select Data 1",
        parameters: {
          columnSelections: {},
        },
      },
      {
        type: ModuleType.SplitData,
        position: { x: 100, y: 400 },
        name: "Split Data",
        parameters: {
          train_size: 0.7,
          random_state: 43,
          shuffle: "True",
          stratify: "False",
          stratify_column: null,
        },
      },
      {
        type: ModuleType.LinearRegression,
        position: { x: 100, y: 550 },
        name: "Linear Regression",
        parameters: {
          model_type: "LinearRegression",
          alpha: 1,
          l1_ratio: 0.5,
          fit_intercept: "True",
          tuning_enabled: "False",
          tuning_strategy: "GridSearch",
          alpha_candidates: "0.01,0.1,1,10",
          l1_ratio_candidates: "0.2,0.5,0.8",
          cv_folds: 5,
          scoring_metric: "neg_mean_squared_error",
        },
      },
      {
        type: ModuleType.Statistics,
        position: { x: 397, y: 76 },
        name: "Statistics 1",
        parameters: {},
      },
      {
        type: ModuleType.TrainModel,
        position: { x: 350, y: 550 },
        name: "Train Model",
        parameters: {
          feature_columns: [],
          label_column: null,
        },
      },
      {
        type: ModuleType.ScoreModel,
        position: { x: 600, y: 550 },
        name: "Score Model",
        parameters: {},
      },
      {
        type: ModuleType.EvaluateModel,
        position: { x: 850, y: 550 },
        name: "Evaluate Model",
        parameters: {
          label_column: null,
          prediction_column: null,
          model_type: "regression",
          threshold: 0.5,
        },
      },
    ],
    connections: [
      {
        fromModuleIndex: 0,
        fromPort: "data_out",
        toModuleIndex: 1,
        toPort: "data_in",
      },
      {
        fromModuleIndex: 0,
        fromPort: "data_out",
        toModuleIndex: 4,
        toPort: "data_in",
      },
      {
        fromModuleIndex: 1,
        fromPort: "data_out",
        toModuleIndex: 2,
        toPort: "data_in",
      },
      {
        fromModuleIndex: 2,
        fromPort: "train_data_out",
        toModuleIndex: 5,
        toPort: "data_in",
      },
      {
        fromModuleIndex: 2,
        fromPort: "test_data_out",
        toModuleIndex: 6,
        toPort: "data_in",
      },
      {
        fromModuleIndex: 3,
        fromPort: "model_out",
        toModuleIndex: 5,
        toPort: "model_in",
      },
      {
        fromModuleIndex: 5,
        fromPort: "trained_model_out",
        toModuleIndex: 6,
        toPort: "model_in",
      },
      {
        fromModuleIndex: 6,
        fromPort: "scored_data_out",
        toModuleIndex: 7,
        toPort: "data_in",
      },
    ],
  },
];
