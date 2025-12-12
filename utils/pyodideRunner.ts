/**
 * Pyodide를 사용하여 브라우저에서 Python 코드를 실행하는 유틸리티
 *
 * Pyodide는 WebAssembly를 통해 브라우저에서 직접 Python을 실행할 수 있게 해줍니다.
 * 별도의 백엔드 서버가 필요 없습니다.
 */

let pyodide: any = null;
let isLoading = false;
let loadPromise: Promise<any> | null = null;
let loadStartTime: number = 0;

/**
 * 타임아웃을 가진 Promise 래퍼
 */
function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  errorMessage: string
): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(() => reject(new Error(errorMessage)), timeoutMs)
    ),
  ]);
}

/**
 * Pyodide를 로드합니다 (한 번만 로드)
 * 타임아웃: 30초
 */
export async function loadPyodide(timeoutMs: number = 30000): Promise<any> {
  if (pyodide) {
    return pyodide;
  }

  if (isLoading && loadPromise) {
    return loadPromise;
  }

  isLoading = true;
  loadStartTime = Date.now();
  loadPromise = (async () => {
    try {
      // @ts-ignore - Pyodide는 전역에서 로드됩니다
      const pyodideModule = await withTimeout(
        loadPyodideModule(),
        timeoutMs,
        `Pyodide 로딩 타임아웃 (${timeoutMs / 1000}초 초과)`
      );
      pyodide = pyodideModule;

      // 필요한 패키지 설치 (타임아웃: 90초)
      // imblearn은 scikit-learn에 포함되어 있지만 별도 설치가 필요할 수 있음
      await withTimeout(
        pyodide.loadPackage(["pandas", "scikit-learn", "numpy", "scipy"]),
        90000,
        "패키지 설치 타임아웃 (90초 초과)"
      );

      isLoading = false;
      loadStartTime = 0;
      return pyodide;
    } catch (error) {
      isLoading = false;
      loadPromise = null;
      loadStartTime = 0;
      throw error;
    }
  })();

  return loadPromise;
}

/**
 * Pyodide 모듈을 동적으로 로드합니다
 */
async function loadPyodideModule(): Promise<any> {
  // Pyodide가 이미 로드되어 있는지 확인
  if (typeof window !== "undefined" && (window as any).loadPyodide) {
    return (window as any).loadPyodide({
      indexURL: "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/",
    });
  }

  // Pyodide가 아직 로드되지 않았다면 에러
  throw new Error(
    "Pyodide is not loaded. Please ensure pyodide.js is included in index.html"
  );
}

/**
 * Python 코드를 실행하고 결과를 반환합니다
 */
export async function runPython(code: string): Promise<any> {
  const py = await loadPyodide();

  try {
    const result = py.runPython(code);
    return result;
  } catch (error: any) {
    throw new Error(`Python execution error: ${error.message}`);
  }
}

/**
 * Python 함수를 호출합니다
 */
export async function callPythonFunction(
  functionName: string,
  ...args: any[]
): Promise<any> {
  const py = await loadPyodide();

  try {
    const func = py.globals.get(functionName);
    if (!func) {
      throw new Error(`Function ${functionName} not found`);
    }

    const result = func(...args);
    return result;
  } catch (error: any) {
    throw new Error(`Python function call error: ${error.message}`);
  }
}

/**
 * 데이터를 Python 객체로 변환합니다
 */
export function toPython(data: any): string {
  return JSON.stringify(data);
}

/**
 * Python 객체를 JavaScript 객체로 변환합니다
 */
export function fromPython(pythonObj: any): any {
  if (pythonObj && typeof pythonObj.toJs === "function") {
    return pythonObj.toJs({ dict_converter: Object.fromEntries });
  }
  return pythonObj;
}

/**
 * SplitData를 Python으로 실행합니다
 * 타임아웃: 60초
 */
export async function splitDataPython(
  data: any[],
  trainSize: number,
  randomState: number,
  shuffle: boolean,
  stratify: boolean,
  stratifyColumn: string | null,
  timeoutMs: number = 60000
): Promise<{ trainIndices: number[]; testIndices: number[] }> {
  let py: any = null;
  try {
    // Pyodide 로드 (타임아웃: 30초)
    try {
      py = await withTimeout(
        loadPyodide(30000),
        30000,
        "Pyodide 로딩 타임아웃 (30초 초과)"
      );
    } catch (loadError: any) {
      const loadErrorMessage = loadError.message || String(loadError);
      if (
        loadErrorMessage.includes("Failed to fetch") ||
        loadErrorMessage.includes("NetworkError")
      ) {
        throw new Error(
          `Pyodide CDN 로드 실패: 네트워크 연결을 확인하거나 인터넷 연결이 필요합니다. ${loadErrorMessage}`
        );
      }
      throw new Error(`Pyodide 로드 실패: ${loadErrorMessage}`);
    }

    // 데이터를 Python에 전달
    py.globals.set("js_data", data);

    // stratify_column을 Python 코드에 전달하기 위한 처리
    // None이면 문자열 'None'으로, 아니면 문자열로 감싸서 전달
    const stratifyColStr = stratifyColumn ? `'${stratifyColumn}'` : "None";

    // JavaScript boolean을 Python boolean으로 변환
    const shufflePython = shuffle ? "True" : "False";
    const stratifyPython = stratify ? "True" : "False";

    // Python 코드 실행 (에러 처리 포함)
    // 결과를 전역 변수에 저장한 후 가져오는 방식 사용
    const code = `
import json
import traceback
import sys
from sklearn.model_selection import train_test_split
import pandas as pd

try:
    # sklearn의 train_test_split을 사용하여 데이터를 분할합니다.
    dataframe = pd.DataFrame(js_data.to_py())
    
    # DataFrame 인덱스를 명시적으로 0부터 시작하도록 리셋
    dataframe.index = range(len(dataframe))
    
    # Parameters from UI
    p_train_size = ${trainSize}
    p_random_state = ${randomState}
    p_shuffle = ${shufflePython}
    p_stratify = ${stratifyPython}
    p_stratify_column = ${stratifyColStr}
    
    # Stratify 배열 준비
    stratify_array = None
    if p_stratify and p_stratify_column and p_stratify_column != 'None':
        if p_stratify_column not in dataframe.columns:
            raise ValueError(f"Stratify column '{p_stratify_column}' not found in DataFrame")
        stratify_array = dataframe[p_stratify_column]
    
    # 데이터 분할
    train_data, test_data = train_test_split(
        dataframe,
        train_size=p_train_size,
        random_state=p_random_state,
        shuffle=p_shuffle,
        stratify=stratify_array
    )
    
    result = {
        'train_indices': train_data.index.tolist(),
        'test_indices': test_data.index.tolist(),
        'train_count': len(train_data),
        'test_count': len(test_data)
    }
    
    # 전역 변수에 저장
    js_result = result
except Exception as e:
    error_traceback = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
    error_result = {
        '__error__': True,
        'error_type': type(e).__name__,
        'error_message': str(e),
        'error_traceback': error_traceback
    }
    # 전역 변수에 저장
    js_result = error_result
`;

    // Python 코드 실행
    await withTimeout(
      Promise.resolve(py.runPython(code)),
      timeoutMs,
      `Python split_data 실행 타임아웃 (${timeoutMs / 1000}초 초과)`
    );

    // 전역 변수에서 결과 가져오기
    const resultPyObj = py.globals.get("js_result");

    // 결과 객체 검증
    if (!resultPyObj) {
      throw new Error(
        `Python split_data error: Python code returned None or undefined.`
      );
    }

    // Python 딕셔너리를 JavaScript 객체로 변환
    const result = fromPython(resultPyObj);

    // 에러가 발생한 경우 처리
    if (result && result.__error__) {
      throw new Error(
        `Python split_data error:\n${
          result.error_traceback || result.error_message
        }`
      );
    }

    // 결과 검증
    if (!result.train_indices || !result.test_indices) {
      throw new Error(
        `Python split_data error: Missing train_indices or test_indices in result.`
      );
    }

    // 정리
    py.globals.delete("js_data");
    py.globals.delete("js_result");
    // js_tuning_options는 Linear Regression에서만 사용되므로 존재할 때만 삭제
    if (py.globals.has("js_tuning_options")) {
      py.globals.delete("js_tuning_options");
    }

    return {
      trainIndices: result.train_indices,
      testIndices: result.test_indices,
    };
  } catch (error: any) {
    // 정리
    try {
      const py = pyodide;
      if (py) {
        py.globals.delete("js_data");
        py.globals.delete("js_result");
        // js_tuning_options는 Linear Regression에서만 사용되므로 존재할 때만 삭제
        if (py.globals.has("js_tuning_options")) {
          py.globals.delete("js_tuning_options");
        }
      }
    } catch {}

    const errorMessage = error.message || String(error);
    throw new Error(`Python split_data error: ${errorMessage}`);
  }
}

export interface LinearRegressionTuningOptions {
  enabled: boolean;
  strategy?: "GridSearch";
  alphaCandidates?: number[];
  l1RatioCandidates?: number[];
  cvFolds?: number;
  scoringMetric?: string;
}

interface LinearRegressionTuningPayload {
  enabled: boolean;
  strategy?: "grid";
  bestParams?: Record<string, number>;
  bestScore?: number;
  scoringMetric?: string;
  candidates?: { params: Record<string, number>; score: number }[];
}

export interface LogisticRegressionTuningOptions {
  enabled: boolean;
  strategy?: "GridSearch";
  cCandidates?: number[];
  l1RatioCandidates?: number[];
  cvFolds?: number;
  scoringMetric?: string;
}

interface LogisticRegressionTuningPayload {
  enabled: boolean;
  strategy?: "grid";
  bestParams?: Record<string, number>;
  bestScore?: number;
  scoringMetric?: string;
  candidates?: { params: Record<string, number>; score: number }[];
}

/**
 * LinearRegression을 Python으로 실행합니다
 * 타임아웃: 60초
 */
export async function fitLinearRegressionPython(
  X: number[][],
  y: number[],
  modelType: string = "LinearRegression",
  fitIntercept: boolean = true,
  alpha: number = 1.0,
  l1Ratio: number = 0.5,
  featureColumns?: string[],
  timeoutMs: number = 60000,
  tuningOptions?: LinearRegressionTuningOptions
): Promise<{
  coefficients: number[];
  intercept: number;
  metrics: Record<string, number>;
  tuning?: LinearRegressionTuningPayload;
}> {
  try {
    // Pyodide 로드 (타임아웃: 30초)
    const py = await withTimeout(
      loadPyodide(30000),
      30000,
      "Pyodide 로딩 타임아웃 (30초 초과)"
    );

    // 데이터를 Python에 전달 (pandas DataFrame으로 변환하기 위해 전체 데이터 전달)
    // 실제 Python 코드와 동일하게 pandas DataFrame 사용
    const dataRows: any[] = [];
    for (let i = 0; i < X.length; i++) {
      const row: any = {};
      if (featureColumns) {
        featureColumns.forEach((col, idx) => {
          row[col] = X[i][idx];
        });
      } else {
        // featureColumns가 없으면 x0, x1, ... 형태로 사용
        X[i].forEach((val, idx) => {
          row[`x${idx}`] = val;
        });
      }
      row["y"] = y[i];
      dataRows.push(row);
    }

    py.globals.set("js_data", dataRows);
    py.globals.set(
      "js_feature_columns",
      featureColumns || X[0].map((_, idx) => `x${idx}`)
    );
    py.globals.set("js_label_column", "y");
    py.globals.set("js_tuning_options", tuningOptions ? tuningOptions : null);

    // Python 코드 실행 (에러 처리 포함)
    // 실제 Python 코드와 동일하게 pandas DataFrame 사용
    const code = `
import json
import numpy as np
import pandas as pd
import traceback
import sys
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

try:
    # 앱에서 보여지는 코드와 정확히 일치하도록 작성
    # 앱 코드: X_train = dataframe[p_feature_columns]
    #          y_train = dataframe[p_label_column]
    #          trained_model = model.fit(X_train, y_train)
    
    # 데이터 준비 - 앱 코드와 동일하게 dataframe 사용
    dataframe = pd.DataFrame(js_data.to_py())
    p_feature_columns = js_feature_columns.to_py()
    p_label_column = str(js_label_column)
    
    # 데이터 검증
    if dataframe.empty:
        raise ValueError("DataFrame is empty")
    if len(p_feature_columns) == 0:
        raise ValueError("No feature columns specified")
    if p_label_column not in dataframe.columns:
        raise ValueError(f"Label column '{p_label_column}' not found in DataFrame")
    
    # 앱 코드와 정확히 일치: X_train = dataframe[p_feature_columns]
    #                        y_train = dataframe[p_label_column]
    X_train = dataframe[p_feature_columns]
    y_train = dataframe[p_label_column]
    
    # 데이터 검증
    if X_train.empty:
        raise ValueError("X_train is empty")
    if y_train.empty:
        raise ValueError("y_train is empty")
    if len(X_train) != len(y_train):
        raise ValueError(f"X_train and y_train must have same number of samples: X_train.shape[0]={len(X_train)}, y_train.shape[0]={len(y_train)}")
    if len(X_train) < 1:
        raise ValueError(f"Need at least 1 sample, got {len(X_train)}")
    
    # 모델 생성 - LinearRegression 모듈에서 생성된 것과 동일
    model_type = '${modelType}'
    p_fit_intercept = ${fitIntercept ? "True" : "False"}
    p_alpha = ${alpha}
    p_l1_ratio = ${l1Ratio}
    
    if model_type == 'LinearRegression':
        model = LinearRegression(fit_intercept=p_fit_intercept)
    elif model_type == 'Lasso':
        model = Lasso(alpha=p_alpha, fit_intercept=p_fit_intercept, random_state=42)
    elif model_type == 'Ridge':
        model = Ridge(alpha=p_alpha, fit_intercept=p_fit_intercept, random_state=42)
    elif model_type == 'ElasticNet':
        model = ElasticNet(alpha=p_alpha, l1_ratio=p_l1_ratio, fit_intercept=p_fit_intercept, random_state=42)
    else:
        model = LinearRegression(fit_intercept=p_fit_intercept)
    
    # 튜닝 옵션 처리
    tuning_options = None
    tuning_enabled = False
    if 'js_tuning_options' in globals() and js_tuning_options is not None:
        try:
            tuning_options = js_tuning_options.to_py()
            tuning_enabled = bool(tuning_options.get('enabled'))
        except Exception:
            tuning_options = None
            tuning_enabled = False

    best_params = {}
    best_score = None
    cv_candidates = []
    scoring_metric_value = 'neg_mean_squared_error'
    if tuning_options and tuning_options.get('scoringMetric'):
        scoring_metric_value = tuning_options.get('scoringMetric')

    should_tune = tuning_enabled and tuning_options is not None and model_type in ('Lasso', 'Ridge', 'ElasticNet')

    if should_tune:
        alpha_candidates = tuning_options.get('alphaCandidates') or [p_alpha]
        alpha_candidates = [float(a) for a in alpha_candidates if a is not None]
        param_grid = {}
        if alpha_candidates:
            param_grid['alpha'] = alpha_candidates
        if model_type == 'ElasticNet':
            l1_candidates = tuning_options.get('l1RatioCandidates') or [p_l1_ratio]
            l1_candidates = [float(a) for a in l1_candidates if a is not None]
            if l1_candidates:
                param_grid['l1_ratio'] = l1_candidates
        if not param_grid:
            param_grid = {'alpha': [float(p_alpha)]}
        cv_folds = int(tuning_options.get('cvFolds', 5))
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=cv_folds,
            scoring=scoring_metric_value,
            n_jobs=None
        )
        grid_search.fit(X_train, y_train)
        trained_model = grid_search.best_estimator_
        best_params = {k: float(v) for k, v in grid_search.best_params_.items()}
        best_score = float(grid_search.best_score_)
        cv_candidates = [
            {'params': params, 'score': float(score)}
            for params, score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score'])
        ][:10]
    else:
        trained_model = model.fit(X_train, y_train)
        if model_type in ('Lasso', 'Ridge', 'ElasticNet'):
            best_params = {'alpha': float(p_alpha)}
            if model_type == 'ElasticNet':
                best_params['l1_ratio'] = float(p_l1_ratio)
    
    # 예측 및 평가 - trained_model 사용 (앱 코드와 일치)
    y_pred = trained_model.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_train, y_pred)
    
    # 결과 준비 - coefficients를 p_feature_columns 순서대로 반환
    if p_fit_intercept:
        intercept = float(trained_model.intercept_)
        # coefficients를 p_feature_columns 순서대로 매핑
        coefficients_list = trained_model.coef_.tolist()
    else:
        intercept = 0.0
        coefficients_list = trained_model.coef_.tolist()
    
    # p_feature_columns 순서대로 coefficients 딕셔너리 생성
    coefficients_dict = {}
    for idx, col in enumerate(p_feature_columns):
        if idx < len(coefficients_list):
            coefficients_dict[col] = float(coefficients_list[idx])
    
    result = {
        'coefficients': coefficients_list,  # 배열 형태로도 반환 (기존 호환성)
        'coefficients_dict': coefficients_dict,  # 딕셔너리 형태로도 반환
        'intercept': intercept,
        'metrics': {
            'R-squared': float(r2),
            'Mean Squared Error': float(mse),
            'Root Mean Squared Error': float(rmse)
        },
        'tuning': {
            'enabled': bool(should_tune),
            'strategy': 'grid' if should_tune else None,
            'bestParams': best_params,
            'bestScore': float(best_score) if best_score is not None else None,
            'scoringMetric': scoring_metric_value if should_tune else None,
            'candidates': cv_candidates
        },
        'feature_columns': p_feature_columns  # 순서 확인용
    }
    
    # 전역 변수에 저장
    js_result = result
except Exception as e:
    error_traceback = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
    error_result = {
        '__error__': True,
        'error_type': type(e).__name__,
        'error_message': str(e),
        'error_traceback': error_traceback
    }
    # 전역 변수에 저장
    js_result = error_result
`;

    // Python 코드 실행
    await withTimeout(
      Promise.resolve(py.runPython(code)),
      timeoutMs,
      "Python LinearRegression 실행 타임아웃 (60초 초과)"
    );

    // 전역 변수에서 결과 가져오기
    const resultPyObj = py.globals.get("js_result");

    // 결과 객체 검증
    if (!resultPyObj) {
      // 디버깅을 위해 Python 상태 확인
      try {
        const debugInfo = py.runPython(`
import sys
debug_info = {
    'last_type': str(type(sys.last_value)) if hasattr(sys, 'last_value') and sys.last_value else None,
    'last_value': str(sys.last_value) if hasattr(sys, 'last_value') and sys.last_value else None
}
debug_info
`);
        const debug = fromPython(debugInfo);
        throw new Error(
          `Python LinearRegression error: Python code returned None or undefined. Debug info: ${JSON.stringify(
            debug
          )}`
        );
      } catch (debugError) {
        throw new Error(
          `Python LinearRegression error: Python code returned None or undefined. Check Python code execution.`
        );
      }
    }

    // Python 딕셔너리를 JavaScript 객체로 변환
    const result = fromPython(resultPyObj);

    // 결과 검증
    if (!result || typeof result !== "object") {
      throw new Error(
        `Python LinearRegression error: Invalid result returned from Python code. Got: ${typeof result}, value: ${JSON.stringify(
          result
        )}`
      );
    }

    // 에러가 발생한 경우 처리
    if (result.__error__) {
      throw new Error(
        `Python LinearRegression error:\n${
          result.error_traceback || result.error_message || "Unknown error"
        }`
      );
    }

    // 필수 속성 검증
    if (!result.coefficients || !Array.isArray(result.coefficients)) {
      throw new Error(
        `Python LinearRegression error: Missing or invalid 'coefficients' in result. Got: ${JSON.stringify(
          result
        )}`
      );
    }
    if (
      typeof result.intercept !== "number" &&
      result.intercept !== null &&
      result.intercept !== undefined
    ) {
      throw new Error(
        `Python LinearRegression error: Missing or invalid 'intercept' in result. Got: ${typeof result.intercept}`
      );
    }
    if (!result.metrics || typeof result.metrics !== "object") {
      throw new Error(
        `Python LinearRegression error: Missing or invalid 'metrics' in result. Got: ${typeof result.metrics}`
      );
    }

    // 정리
    py.globals.delete("js_data");
    py.globals.delete("js_feature_columns");
    py.globals.delete("js_label_column");
    py.globals.delete("js_result");

    return {
      coefficients: result.coefficients,
      intercept: result.intercept ?? 0.0,
      metrics: result.metrics,
      tuning: result.tuning,
    };
  } catch (error: any) {
    // 정리
    try {
      const py = pyodide;
      if (py) {
        py.globals.delete("js_data");
        py.globals.delete("js_feature_columns");
        py.globals.delete("js_label_column");
        py.globals.delete("js_result");
      }
    } catch {}

    // 에러 메시지 추출
    let errorMessage = error.message || String(error);

    // Pyodide PythonError의 경우 더 자세한 정보 추출 시도
    if (
      error.name === "PythonError" ||
      error.toString().includes("Traceback")
    ) {
      try {
        const py = pyodide;
        if (py) {
          try {
            // Python의 sys.last_value에서 에러 정보 가져오기
            const lastError = py.runPython(`
import sys
import traceback
if hasattr(sys, 'last_value') and sys.last_value is not None:
    error_str = ''.join(traceback.format_exception(type(sys.last_value), sys.last_value, sys.last_traceback))
    error_str
else:
    ''
`);
            if (lastError && String(lastError).trim()) {
              errorMessage = String(lastError);
            }
          } catch (tracebackError) {
            // traceback 추출 실패 시 원본 에러 사용
          }
        }
      } catch (e) {
        // 에러 정보 추출 실패 시 원본 메시지 사용
      }
    }

    // 전체 에러 메시지 포함
    const fullError = errorMessage.includes("Traceback")
      ? errorMessage
      : `${error.toString()}\n${errorMessage}`;
    throw new Error(`Python LinearRegression error:\n${fullError}`);
  }
}

/**
 * LogisticRegression을 Python으로 실행합니다
 * 타임아웃: 60초
 */
export async function fitLogisticRegressionPython(
  X: number[][],
  y: number[],
  penalty: string = "l2",
  C: number = 1.0,
  solver: string = "lbfgs",
  maxIter: number = 100,
  featureColumns?: string[],
  timeoutMs: number = 60000,
  tuningOptions?: LogisticRegressionTuningOptions
): Promise<{
  coefficients: number[][];
  intercept: number[];
  metrics: Record<string, number>;
  tuning?: LogisticRegressionTuningPayload;
}> {
  try {
    // Pyodide 로드 (타임아웃: 30초)
    const py = await withTimeout(
      loadPyodide(30000),
      30000,
      "Pyodide 로딩 타임아웃 (30초 초과)"
    );

    // 데이터를 Python에 전달
    const dataRows: any[] = [];
    for (let i = 0; i < X.length; i++) {
      const row: any = {};
      if (featureColumns) {
        featureColumns.forEach((col, idx) => {
          row[col] = X[i][idx];
        });
      } else {
        X[i].forEach((val, idx) => {
          row[`x${idx}`] = val;
        });
      }
      row["y"] = y[i];
      dataRows.push(row);
    }

    py.globals.set("js_data", dataRows);
    py.globals.set(
      "js_feature_columns",
      featureColumns || X[0].map((_, idx) => `x${idx}`)
    );
    py.globals.set("js_label_column", "y");
    py.globals.set("js_tuning_options", tuningOptions ? tuningOptions : null);

    // Python 코드 실행
    const code = `
import json
import numpy as np
import pandas as pd
import traceback
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

try:
    # 데이터 준비
    dataframe = pd.DataFrame(js_data.to_py())
    p_feature_columns = js_feature_columns.to_py()
    p_label_column = str(js_label_column)
    
    # 데이터 검증
    if dataframe.empty:
        raise ValueError("DataFrame is empty")
    if len(p_feature_columns) == 0:
        raise ValueError("No feature columns specified")
    if p_label_column not in dataframe.columns:
        raise ValueError(f"Label column '{p_label_column}' not found in DataFrame")
    
    X_train = dataframe[p_feature_columns]
    y_train = dataframe[p_label_column]
    
    # 데이터 검증
    if X_train.empty:
        raise ValueError("X_train is empty")
    if y_train.empty:
        raise ValueError("y_train is empty")
    if len(X_train) != len(y_train):
        raise ValueError(f"X_train and y_train must have same number of samples: X_train.shape[0]={len(X_train)}, y_train.shape[0]={len(y_train)}")
    if len(X_train) < 1:
        raise ValueError(f"Need at least 1 sample, got {len(X_train)}")
    
    # 모델 생성
    p_penalty = '${penalty}'
    p_C = ${C}
    p_solver = '${solver}'
    p_max_iter = ${maxIter}
    
    # penalty와 solver 호환성 확인
    if p_penalty == 'l1' and p_solver not in ('liblinear', 'saga'):
        p_solver = 'liblinear'
    elif p_penalty == 'elasticnet' and p_solver != 'saga':
        p_solver = 'saga'
    elif p_penalty == 'none' and p_solver not in ('lbfgs', 'newton-cg', 'sag', 'saga'):
        p_solver = 'lbfgs'
    
    model = LogisticRegression(
        penalty=p_penalty if p_penalty != 'none' else None,
        C=p_C,
        solver=p_solver,
        max_iter=p_max_iter,
        random_state=42
    )
    
    # 튜닝 옵션 처리
    tuning_options = None
    tuning_enabled = False
    if 'js_tuning_options' in globals() and js_tuning_options is not None:
        try:
            tuning_options = js_tuning_options.to_py()
            tuning_enabled = bool(tuning_options.get('enabled'))
        except Exception:
            tuning_options = None
            tuning_enabled = False

    best_params = {}
    best_score = None
    cv_candidates = []
    scoring_metric_value = 'accuracy'
    if tuning_options and tuning_options.get('scoringMetric'):
        scoring_metric_value = tuning_options.get('scoringMetric')

    should_tune = tuning_enabled and tuning_options is not None

    if should_tune:
        c_candidates = tuning_options.get('cCandidates') or [p_C]
        c_candidates = [float(c) for c in c_candidates if c is not None]
        param_grid = {}
        if c_candidates:
            param_grid['C'] = c_candidates
        if p_penalty == 'elasticnet':
            l1_candidates = tuning_options.get('l1RatioCandidates') or [0.5]
            l1_candidates = [float(a) for a in l1_candidates if a is not None]
            if l1_candidates:
                param_grid['l1_ratio'] = l1_candidates
        if not param_grid:
            param_grid = {'C': [float(p_C)]}
        cv_folds = int(tuning_options.get('cvFolds', 5))
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=cv_folds,
            scoring=scoring_metric_value,
            n_jobs=None
        )
        grid_search.fit(X_train, y_train)
        trained_model = grid_search.best_estimator_
        best_params = {k: float(v) for k, v in grid_search.best_params_.items()}
        best_score = float(grid_search.best_score_)
        cv_candidates = [
            {'params': params, 'score': float(score)}
            for params, score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score'])
        ][:10]
    else:
        trained_model = model.fit(X_train, y_train)
        best_params = {'C': float(p_C)}
        if p_penalty == 'elasticnet':
            best_params['l1_ratio'] = 0.5
    
    # 예측 및 평가
    y_pred = trained_model.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_train, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_train, y_pred, average='weighted', zero_division=0)
    
    # ROC AUC (이진 분류인 경우만)
    roc_auc = None
    unique_labels = np.unique(y_train)
    if len(unique_labels) == 2:
        try:
            y_pred_proba = trained_model.predict_proba(X_train)[:, 1]
            roc_auc = roc_auc_score(y_train, y_pred_proba)
        except Exception:
            roc_auc = None
    
    # 결과 준비 - coefficients는 다중 클래스의 경우 2D 배열
    intercept = trained_model.intercept_.tolist()
    coefficients_list = trained_model.coef_.tolist()
    
    # p_feature_columns 순서대로 coefficients 딕셔너리 생성
    coefficients_dict = {}
    if len(coefficients_list) == 1:
        # 이진 분류
        for idx, col in enumerate(p_feature_columns):
            if idx < len(coefficients_list[0]):
                coefficients_dict[col] = float(coefficients_list[0][idx])
    else:
        # 다중 클래스
        for class_idx, coefs in enumerate(coefficients_list):
            for idx, col in enumerate(p_feature_columns):
                if idx < len(coefs):
                    key = f"{col}_class_{class_idx}"
                    coefficients_dict[key] = float(coefs[idx])
    
    metrics_dict = {
        'Accuracy': float(accuracy),
        'Precision': float(precision),
        'Recall': float(recall),
        'F1-Score': float(f1)
    }
    if roc_auc is not None:
        metrics_dict['ROC-AUC'] = float(roc_auc)
    
    result = {
        'coefficients': coefficients_list,
        'coefficients_dict': coefficients_dict,
        'intercept': intercept,
        'metrics': metrics_dict,
        'tuning': {
            'enabled': bool(should_tune),
            'strategy': 'grid' if should_tune else None,
            'bestParams': best_params,
            'bestScore': float(best_score) if best_score is not None else None,
            'scoringMetric': scoring_metric_value if should_tune else None,
            'candidates': cv_candidates
        },
        'feature_columns': p_feature_columns
    }
    
    # 전역 변수에 저장
    js_result = result
except Exception as e:
    error_traceback = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
    error_result = {
        '__error__': True,
        'error_type': type(e).__name__,
        'error_message': str(e),
        'error_traceback': error_traceback
    }
    # 전역 변수에 저장
    js_result = error_result
`;

    // Python 코드 실행
    await withTimeout(
      Promise.resolve(py.runPython(code)),
      timeoutMs,
      "Python LogisticRegression 실행 타임아웃 (60초 초과)"
    );

    // 전역 변수에서 결과 가져오기
    const resultPyObj = py.globals.get("js_result");

    // 결과 객체 검증
    if (!resultPyObj) {
      throw new Error(
        `Python LogisticRegression error: Python code returned None or undefined.`
      );
    }

    // Python 딕셔너리를 JavaScript 객체로 변환
    const result = fromPython(resultPyObj);

    // 에러가 발생한 경우 처리
    if (result.__error__) {
      throw new Error(
        `Python LogisticRegression error:\n${
          result.error_traceback || result.error_message || "Unknown error"
        }`
      );
    }

    // 필수 속성 검증
    if (!result.coefficients || !Array.isArray(result.coefficients)) {
      throw new Error(
        `Python LogisticRegression error: Missing or invalid 'coefficients' in result.`
      );
    }
    if (!result.intercept || !Array.isArray(result.intercept)) {
      throw new Error(
        `Python LogisticRegression error: Missing or invalid 'intercept' in result.`
      );
    }
    if (!result.metrics || typeof result.metrics !== "object") {
      throw new Error(
        `Python LogisticRegression error: Missing or invalid 'metrics' in result.`
      );
    }

    // 정리
    py.globals.delete("js_data");
    py.globals.delete("js_feature_columns");
    py.globals.delete("js_label_column");
    py.globals.delete("js_result");
    if (py.globals.has("js_tuning_options")) {
      py.globals.delete("js_tuning_options");
    }

    return {
      coefficients: result.coefficients,
      intercept: result.intercept,
      metrics: result.metrics,
      tuning: result.tuning,
    };
  } catch (error: any) {
    // 정리
    try {
      const py = pyodide;
      if (py) {
        py.globals.delete("js_data");
        py.globals.delete("js_feature_columns");
        py.globals.delete("js_label_column");
        py.globals.delete("js_result");
        if (py.globals.has("js_tuning_options")) {
          py.globals.delete("js_tuning_options");
        }
      }
    } catch {}

    const errorMessage = error.message || String(error);
    throw new Error(`Python LogisticRegression error:\n${errorMessage}`);
  }
}

/**
 * Count Regression (Poisson, Negative Binomial, Quasi-Poisson)을 statsmodels로 실행합니다
 * 타임아웃: 60초
 */
export async function fitCountRegressionStatsmodels(
  X: number[][],
  y: number[],
  distributionType: string,
  featureColumns: string[],
  maxIter: number = 100,
  disp: number = 1.0,
  timeoutMs: number = 60000
): Promise<{
  coefficients: Record<string, number>;
  intercept: number;
  metrics: Record<string, number>;
  summary: {
    coefficients: Record<
      string,
      {
        coef: number;
        "std err": number;
        z: number;
        "P>|z|": number;
        "[0.025": number;
        "0.975]": number;
      }
    >;
    metrics: Record<string, number | string>;
  };
}> {
  try {
    const py = await withTimeout(
      loadPyodide(30000),
      30000,
      "Pyodide 로딩 타임아웃 (30초 초과)"
    );

    // statsmodels 패키지 로드
    await withTimeout(
      py.loadPackage(["statsmodels"]),
      60000,
      "statsmodels 패키지 설치 타임아웃 (60초 초과)"
    );

    py.globals.set("js_X", X);
    py.globals.set("js_y", y);
    py.globals.set("js_feature_columns", featureColumns);
    py.globals.set("js_distribution_type", distributionType);
    py.globals.set("js_max_iter", maxIter);
    py.globals.set("js_disp", disp);

    const code = `
import json
import pandas as pd
import numpy as np
import sys
sys.path.append('/')
from data_analysis_modules import fit_count_regression_statsmodels

# 데이터 준비
X_array = js_X.to_py()
y_array = js_y.to_py()
feature_columns = js_feature_columns.to_py()
distribution_type = js_distribution_type.to_py()
max_iter = int(js_max_iter)
disp = float(js_disp)

# DataFrame 생성
df = pd.DataFrame(X_array, columns=feature_columns)
df['label'] = y_array

# 모델 피팅
result = fit_count_regression_statsmodels(
    df=df,
    distribution_type=distribution_type,
    feature_columns=feature_columns,
    label_column='label',
    max_iter=max_iter,
    disp=disp
)

# 결과 추출
results_obj = result['results']
coefficients_dict = result['coefficients']
metrics_dict = result['metrics']

# 계수와 절편 추출
intercept = coefficients_dict.get('const', {}).get('coef', 0.0)
feature_coefficients = {}
for col in feature_columns:
    if col in coefficients_dict:
        feature_coefficients[col] = coefficients_dict[col]['coef']

# 통계량 준비
summary_coefficients = {}
for param_name, param_data in coefficients_dict.items():
    summary_coefficients[param_name] = {
        'coef': float(param_data['coef']),
        'std err': float(param_data['std err']),
        'z': float(param_data['z']),
        'P>|z|': float(param_data['P>|z|']),
        '[0.025': float(param_data['[0.025']),
        '0.975]': float(param_data['0.975]'])
    }

# 메트릭 준비 (None 값 제거)
summary_metrics = {}
for key, value in metrics_dict.items():
    if value is not None:
        if isinstance(value, (int, float)):
            summary_metrics[key] = float(value)
        else:
            summary_metrics[key] = str(value)

# 반환값 구성
result_dict = {
    'coefficients': feature_coefficients,
    'intercept': float(intercept),
    'metrics': summary_metrics,
    'summary': {
        'coefficients': summary_coefficients,
        'metrics': summary_metrics
    }
}

json.dumps(result_dict)
`;

    const resultJson = await withTimeout(
      py.runPython(code),
      timeoutMs,
      `Count Regression 실행 타임아웃 (${timeoutMs / 1000}초 초과)`
    );

    const result = JSON.parse(resultJson);

    return {
      coefficients: result.coefficients,
      intercept: result.intercept,
      metrics: result.metrics,
      summary: result.summary,
    };
  } catch (error: any) {
    const errorMessage = error.message || String(error);
    throw new Error(
      `Python Count Regression (statsmodels) error:\n${errorMessage}`
    );
  }
}

/**
 * Statsmodels를 사용하여 통계 모델을 실행합니다
 * 타임아웃: 120초
 */
export async function runStatsModel(
  X: number[][],
  y: number[],
  modelType: string,
  featureColumns: string[],
  timeoutMs: number = 120000,
  maxIter: number = 100,
  disp: number = 1.0
): Promise<{
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
}> {
  try {
    const py = await withTimeout(
      loadPyodide(30000),
      30000,
      "Pyodide 로딩 타임아웃 (30초 초과)"
    );

    // statsmodels 패키지 로드
    await withTimeout(
      py.loadPackage(["statsmodels"]),
      60000,
      "statsmodels 패키지 설치 타임아웃 (60초 초과)"
    );

    py.globals.set("js_X", X);
    py.globals.set("js_y", y);
    py.globals.set("js_feature_columns", featureColumns);
    py.globals.set("js_model_type", modelType);
    py.globals.set("js_max_iter", maxIter);
    py.globals.set("js_disp", disp);

    const code = `
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

# fit_count_regression_statsmodels 함수 정의
def fit_count_regression_statsmodels(df, distribution_type, feature_columns, label_column, max_iter=100, disp=1.0):
    print(f"{distribution_type} 회귀 모델 피팅 중...")
    
    X = df[feature_columns].copy()
    y = df[label_column].copy()
    
    # 결측치 제거
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    if len(X) == 0:
        raise ValueError("유효한 데이터가 없습니다. 결측치를 확인하세요.")
    
    X = sm.add_constant(X, prepend=True)
    
    try:
        if distribution_type == 'Poisson':
            model = sm.Poisson(y, X)
            results = model.fit(maxiter=max_iter)
        elif distribution_type == 'NegativeBinomial':
            model = sm.NegativeBinomial(y, X, loglike_method='nb2')
            results = model.fit(maxiter=max_iter, disp=disp)
        elif distribution_type == 'QuasiPoisson':
            model = sm.GLM(y, X, family=sm.families.Poisson())
            results = model.fit(maxiter=max_iter)
            mu = results.mu
            pearson_resid = (y - mu) / np.sqrt(mu)
            phi = np.sum(pearson_resid**2) / (len(y) - len(feature_columns) - 1)
            results.scale = phi
        else:
            raise ValueError(f"지원하지 않는 분포 타입: {distribution_type}")
        
        summary_text = str(results.summary())
        print(f"\\n--- {distribution_type} 회귀 모델 결과 ---")
        print(summary_text)
        
        metrics = {}
        metrics['Log Likelihood'] = results.llf if hasattr(results, 'llf') else None
        metrics['AIC'] = results.aic if hasattr(results, 'aic') else None
        metrics['BIC'] = results.bic if hasattr(results, 'bic') else None
        metrics['Deviance'] = results.deviance if hasattr(results, 'deviance') else None
        metrics['Pearson chi2'] = results.pearson_chi2 if hasattr(results, 'pearson_chi2') else None
        
        if distribution_type == 'NegativeBinomial':
            if hasattr(model, 'alpha'):
                metrics['Dispersion (alpha)'] = model.alpha
            elif hasattr(results, 'alpha'):
                metrics['Dispersion (alpha)'] = results.alpha
        
        if distribution_type == 'QuasiPoisson':
            if hasattr(results, 'scale'):
                metrics['Dispersion (phi)'] = results.scale
        
        coefficients = {}
        if hasattr(results, 'params'):
            params = results.params
            if hasattr(params, 'to_dict'):
                params_dict = params.to_dict()
            else:
                params_dict = {}
                for i, name in enumerate(results.model.exog_names):
                    if hasattr(params, 'iloc'):
                        val = params.iloc[i]
                    elif hasattr(params, '__getitem__'):
                        val = params[i]
                    else:
                        val = params.values[i] if hasattr(params, 'values') else 0.0
                    # 값이 메서드가 아닌지 확인하고 실제 값을 가져옴
                    if callable(val) and not isinstance(val, (int, float, np.number)):
                        try:
                            val = val() if hasattr(val, '__call__') else 0.0
                        except:
                            val = 0.0
                    params_dict[name] = val
            
            if hasattr(results, 'bse'):
                bse = results.bse
                if hasattr(bse, 'to_dict'):
                    bse_dict = bse.to_dict()
                else:
                    bse_dict = {}
                    for i, name in enumerate(results.model.exog_names):
                        if hasattr(bse, 'iloc'):
                            val = bse.iloc[i]
                        elif hasattr(bse, '__getitem__'):
                            val = bse[i]
                        else:
                            val = bse.values[i] if hasattr(bse, 'values') else 0.0
                        # 값이 메서드가 아닌지 확인하고 실제 값을 가져옴
                        if callable(val) and not isinstance(val, (int, float, np.number)):
                            try:
                                val = val() if hasattr(val, '__call__') else 0.0
                            except:
                                val = 0.0
                        bse_dict[name] = val
            else:
                bse_dict = {name: 0.0 for name in params_dict.keys()}
            
            if hasattr(results, 'tvalues'):
                tvalues = results.tvalues
            elif hasattr(results, 'zvalues'):
                tvalues = results.zvalues
            else:
                tvalues = None
            
            if hasattr(results, 'pvalues'):
                pvalues = results.pvalues
            else:
                pvalues = None
            
            conf_int = None
            if hasattr(results, 'conf_int'):
                conf_int = results.conf_int()
            
            # float 변환을 안전하게 처리하는 헬퍼 함수
            def safe_float(value, default=0.0):
                if value is None:
                    return default
                # callable이지만 숫자 타입이 아닌 경우 (메서드 객체)
                if callable(value) and not isinstance(value, (int, float, np.number)):
                    try:
                        value = value()
                    except:
                        return default
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return default
            
            for param_name in params_dict.keys():
                coef_value = params_dict[param_name]
                std_err = bse_dict.get(param_name, 0.0)
                z_value = tvalues[param_name] if tvalues is not None and param_name in tvalues.index else 0.0
                p_value = pvalues[param_name] if pvalues is not None and param_name in pvalues.index else 1.0
                
                conf_lower = conf_int.loc[param_name, 0] if conf_int is not None and param_name in conf_int.index else 0.0
                conf_upper = conf_int.loc[param_name, 1] if conf_int is not None and param_name in conf_int.index else 0.0
                
                coefficients[param_name] = {
                    'coef': safe_float(coef_value),
                    'std err': safe_float(std_err),
                    'z': safe_float(z_value),
                    'P>|z|': safe_float(p_value),
                    '[0.025': safe_float(conf_lower),
                    '0.975]': safe_float(conf_upper)
                }
        
        return {
            'results': results,
            'summary_text': summary_text,
            'metrics': metrics,
            'coefficients': coefficients,
            'distribution_type': distribution_type
        }
        
    except Exception as e:
        print(f"모델 피팅 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise

# run_stats_model 함수 정의
def run_stats_model(df, model_type, feature_columns, label_column, max_iter=100, disp=1.0):
    # Count regression 모델의 경우 fit_count_regression_statsmodels 사용
    if model_type in ['Poisson', 'NegativeBinomial', 'QuasiPoisson']:
        model_results = fit_count_regression_statsmodels(
            df, model_type, feature_columns, label_column, max_iter, disp
        )
        
        print("\\n=== 모델 통계량 ===")
        for key, value in model_results['metrics'].items():
            if value is not None:
                print(f"{key}: {value:.6f}")
        
        print("\\n=== 계수 정보 ===")
        for param_name, coef_info in model_results['coefficients'].items():
            print(f"{param_name}:")
            print(f"  계수: {coef_info['coef']:.6f}")
            print(f"  표준 오차: {coef_info['std err']:.6f}")
            print(f"  z-통계량: {coef_info['z']:.6f}")
            print(f"  p-value: {coef_info['P>|z|']:.6f}")
            print(f"  신뢰구간: [{coef_info['[0.025']:.6f}, {coef_info['0.975]']:.6f}]")
        
        return model_results['results']
    
    # 다른 모델의 경우 기존 방식 사용
    print(f"{model_type} 모델 피팅 중...")
    
    X = df[feature_columns]
    y = df[label_column]
    X = sm.add_constant(X, prepend=True)
    
    if model_type == 'OLS':
        model = sm.OLS(y, X)
    elif model_type == 'Logistic':
        model = sm.Logit(y, X)
    elif model_type == 'Gamma':
        model = sm.GLM(y, X, family=sm.families.Gamma())
    elif model_type == 'Tweedie':
        model = sm.GLM(y, X, family=sm.families.Tweedie(var_power=1.5))
    else:
        print(f"오류: 알 수 없는 모델 타입 '{model_type}'")
        return None
    
    try:
        results = model.fit()
        print(f"\\n--- {model_type} 모델 결과 ---")
        print(results.summary())
        return results
    except Exception as e:
        print(f"모델 피팅 중 오류 발생: {e}")
        return None

# 데이터 준비
X_array = js_X.to_py()
y_array = js_y.to_py()
feature_columns = js_feature_columns.to_py()
model_type = str(js_model_type)  # 이미 문자열이므로 to_py() 불필요
max_iter = int(js_max_iter)
disp = float(js_disp)

# DataFrame 생성
df = pd.DataFrame(X_array, columns=feature_columns)
df['label'] = y_array

# 모델 피팅
results_obj = run_stats_model(
    df=df,
    model_type=model_type,
    feature_columns=feature_columns,
    label_column='label',
    max_iter=max_iter,
    disp=disp
)

if results_obj is None:
    raise ValueError("모델 피팅 실패")

# 결과 추출
summary_text = str(results_obj.summary())

# 계수 정보 추출
coefficients_dict = {}
if hasattr(results_obj, 'params'):
    params = results_obj.params
    bse = results_obj.bse if hasattr(results_obj, 'bse') else None
    tvalues = results_obj.tvalues if hasattr(results_obj, 'tvalues') else None
    zvalues = results_obj.zvalues if hasattr(results_obj, 'zvalues') else None
    pvalues = results_obj.pvalues if hasattr(results_obj, 'pvalues') else None
    conf_int = results_obj.conf_int() if hasattr(results_obj, 'conf_int') else None
    
    # float 변환을 안전하게 처리하는 헬퍼 함수
    def safe_float(value, default=0.0):
        if value is None:
            return default
        # callable이지만 숫자 타입이 아닌 경우 (메서드 객체)
        if callable(value) and not isinstance(value, (int, float, np.number)):
            try:
                value = value()
            except:
                return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    
    for param_name in params.index:
        coef_val = params[param_name]
        coef_value = safe_float(coef_val)
        std_err_val = bse[param_name] if bse is not None else 0.0
        std_err = safe_float(std_err_val)
        z_val = zvalues[param_name] if zvalues is not None else (tvalues[param_name] if tvalues is not None else 0.0)
        z_value = safe_float(z_val)
        p_val = pvalues[param_name] if pvalues is not None else 1.0
        p_value = safe_float(p_val)
        conf_lower_val = conf_int.loc[param_name, 0] if conf_int is not None else 0.0
        conf_lower = safe_float(conf_lower_val)
        conf_upper_val = conf_int.loc[param_name, 1] if conf_int is not None else 0.0
        conf_upper = safe_float(conf_upper_val)
        
        coefficients_dict[param_name] = {
            'coef': coef_value,
            'std err': std_err,
            'z': z_value,
            't': z_value,
            'P>|z|': p_value,
            'P>|t|': p_value,
            '[0.025': conf_lower,
            '0.975]': conf_upper
        }

# 메트릭 추출
metrics_dict = {}
if hasattr(results_obj, 'llf'):
    llf_val = results_obj.llf
    metrics_dict['Log-Likelihood'] = safe_float(llf_val)
if hasattr(results_obj, 'aic'):
    aic_val = results_obj.aic
    metrics_dict['AIC'] = safe_float(aic_val)
if hasattr(results_obj, 'bic'):
    bic_val = results_obj.bic
    metrics_dict['BIC'] = safe_float(bic_val)
if hasattr(results_obj, 'rsquared'):
    rsq_val = results_obj.rsquared
    metrics_dict['R-squared'] = safe_float(rsq_val)
if hasattr(results_obj, 'rsquared_adj'):
    rsq_adj_val = results_obj.rsquared_adj
    metrics_dict['Adj. R-squared'] = safe_float(rsq_adj_val)
if hasattr(results_obj, 'fvalue'):
    fval = results_obj.fvalue
    metrics_dict['F-statistic'] = safe_float(fval)
if hasattr(results_obj, 'f_pvalue'):
    fpval = results_obj.f_pvalue
    metrics_dict['Prob (F-statistic)'] = safe_float(fpval)
if hasattr(results_obj, 'llnull'):
    llnull_val = results_obj.llnull
    metrics_dict['LL-Null'] = safe_float(llnull_val)
if hasattr(results_obj, 'llr'):
    llr_val = results_obj.llr
    metrics_dict['LLR'] = safe_float(llr_val)
if hasattr(results_obj, 'llr_pvalue'):
    llr_pval = results_obj.llr_pvalue
    metrics_dict['LLR p-value'] = safe_float(llr_pval)
if hasattr(results_obj, 'pseudo_rsquared'):
    pseudo_rsq = results_obj.pseudo_rsquared
    metrics_dict['Pseudo R-squ.'] = safe_float(pseudo_rsq)

# 반환값 구성
result_dict = {
    'summary': {
        'coefficients': coefficients_dict,
        'metrics': metrics_dict
    }
}

json.dumps(result_dict)
`;

    const resultJson = await withTimeout(
      py.runPython(code),
      timeoutMs,
      `Stats Model 실행 타임아웃 (${timeoutMs / 1000}초 초과)`
    );

    const result = JSON.parse(resultJson);

    return {
      summary: result.summary,
    };
  } catch (error: any) {
    const errorMessage = error.message || String(error);
    throw new Error(`Python Stats Model (statsmodels) error:\n${errorMessage}`);
  }
}

/**
 * DiversionChecker를 Python으로 실행합니다
 * 타임아웃: 120초
 */
export async function runDiversionChecker(
  X: number[][],
  y: number[],
  featureColumns: string[],
  labelColumn: string,
  maxIter: number = 100,
  timeoutMs: number = 120000
): Promise<{
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
    phiInterpretation: string;
    recommendation: string;
    poissonAic: number | null;
    negativeBinomialAic: number | null;
    cameronTrivediCoef: number;
    cameronTrivediPvalue: number;
    cameronTrivediConclusion: string;
  };
}> {
  try {
    const py = await withTimeout(
      loadPyodide(30000),
      30000,
      "Pyodide 로딩 타임아웃 (30초 초과)"
    );

    // statsmodels 패키지 로드
    await withTimeout(
      py.loadPackage(["statsmodels"]),
      60000,
      "statsmodels 패키지 설치 타임아웃 (60초 초과)"
    );

    // 데이터를 Python에 전달
    const dataRows: any[] = [];
    for (let i = 0; i < X.length; i++) {
      const row: any = {};
      featureColumns.forEach((col, idx) => {
        row[col] = X[i][idx];
      });
      row[labelColumn] = y[i];
      dataRows.push(row);
    }

    py.globals.set("js_data", dataRows);
    py.globals.set("js_feature_columns", featureColumns);
    py.globals.set("js_label_column", labelColumn);
    py.globals.set("js_max_iter", maxIter);

    const code = `
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

# fit_count_regression_statsmodels 함수 정의
def fit_count_regression_statsmodels(df, distribution_type, feature_columns, label_column, max_iter=100, disp=1.0):
    print(f"{distribution_type} 회귀 모델 피팅 중...")
    
    X = df[feature_columns].copy()
    y = df[label_column].copy()
    
    # 결측치 제거
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    if len(X) == 0:
        raise ValueError("유효한 데이터가 없습니다. 결측치를 확인하세요.")
    
    X = sm.add_constant(X, prepend=True)
    
    try:
        if distribution_type == 'Poisson':
            model = sm.Poisson(y, X)
            results = model.fit(maxiter=max_iter)
        elif distribution_type == 'NegativeBinomial':
            model = sm.NegativeBinomial(y, X, loglike_method='nb2')
            results = model.fit(maxiter=max_iter, disp=disp)
        elif distribution_type == 'QuasiPoisson':
            model = sm.GLM(y, X, family=sm.families.Poisson())
            results = model.fit(maxiter=max_iter)
            mu = results.mu
            pearson_resid = (y - mu) / np.sqrt(mu)
            phi = np.sum(pearson_resid**2) / (len(y) - len(feature_columns) - 1)
            results.scale = phi
        else:
            raise ValueError(f"지원하지 않는 분포 타입: {distribution_type}")
        
        summary_text = str(results.summary())
        print(f"\\n--- {distribution_type} 회귀 모델 결과 ---")
        print(summary_text)
        
        metrics = {}
        metrics['Log Likelihood'] = results.llf if hasattr(results, 'llf') else None
        metrics['AIC'] = results.aic if hasattr(results, 'aic') else None
        metrics['BIC'] = results.bic if hasattr(results, 'bic') else None
        metrics['Deviance'] = results.deviance if hasattr(results, 'deviance') else None
        metrics['Pearson chi2'] = results.pearson_chi2 if hasattr(results, 'pearson_chi2') else None
        
        if distribution_type == 'NegativeBinomial':
            if hasattr(model, 'alpha'):
                metrics['Dispersion (alpha)'] = model.alpha
            elif hasattr(results, 'alpha'):
                metrics['Dispersion (alpha)'] = results.alpha
        
        if distribution_type == 'QuasiPoisson':
            if hasattr(results, 'scale'):
                metrics['Dispersion (phi)'] = results.scale
        
        coefficients = {}
        if hasattr(results, 'params'):
            params = results.params
            if hasattr(params, 'to_dict'):
                params_dict = params.to_dict()
            else:
                params_dict = {name: params.iloc[i] if hasattr(params, 'iloc') else params[i] 
                               for i, name in enumerate(results.model.exog_names)}
            
            if hasattr(results, 'bse'):
                bse = results.bse
                if hasattr(bse, 'to_dict'):
                    bse_dict = bse.to_dict()
                else:
                    bse_dict = {name: bse.iloc[i] if hasattr(bse, 'iloc') else bse[i] 
                               for i, name in enumerate(results.model.exog_names)}
            else:
                bse_dict = {name: 0.0 for name in params_dict.keys()}
            
            if hasattr(results, 'tvalues'):
                tvalues = results.tvalues
            elif hasattr(results, 'zvalues'):
                tvalues = results.zvalues
            else:
                tvalues = None
            
            if hasattr(results, 'pvalues'):
                pvalues = results.pvalues
            else:
                pvalues = None
            
            conf_int = None
            if hasattr(results, 'conf_int'):
                conf_int = results.conf_int()
            
            for param_name in params_dict.keys():
                coef_value = params_dict[param_name]
                std_err = bse_dict.get(param_name, 0.0)
                z_value = tvalues[param_name] if tvalues is not None and param_name in tvalues.index else 0.0
                p_value = pvalues[param_name] if pvalues is not None and param_name in pvalues.index else 1.0
                
                conf_lower = conf_int.loc[param_name, 0] if conf_int is not None and param_name in conf_int.index else 0.0
                conf_upper = conf_int.loc[param_name, 1] if conf_int is not None and param_name in conf_int.index else 0.0
                
                coefficients[param_name] = {
                    'coef': float(coef_value),
                    'std err': float(std_err),
                    'z': float(z_value),
                    'P>|z|': float(p_value),
                    '[0.025': float(conf_lower),
                    '0.975]': float(conf_upper)
                }
        
        return {
            'results': results,
            'summary_text': summary_text,
            'metrics': metrics,
            'coefficients': coefficients,
            'distribution_type': distribution_type
        }
        
    except Exception as e:
        print(f"모델 피팅 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise

def dispersion_checker(df, feature_columns, label_column, max_iter=100):
    print("=== 과대산포 검사 (Diversion Checker) ===\\n")
    
    # 1. 포아송 모델 적합
    print("1. 포아송 모델 적합 중...")
    poisson_result = fit_count_regression_statsmodels(
        df, 'Poisson', feature_columns, label_column, max_iter, 1.0
    )
    poisson_results = poisson_result['results']
    
    # 2. Dispersion φ 계산
    print("\\n2. Dispersion φ 계산 중...")
    y = df[label_column].copy()
    mask = ~(df[feature_columns].isnull().any(axis=1) | y.isnull())
    y = y[mask]
    # PoissonResults에서 mu는 fittedvalues 속성을 사용
    if hasattr(poisson_results, 'fittedvalues'):
        mu = poisson_results.fittedvalues
    elif hasattr(poisson_results, 'mu'):
        mu = poisson_results.mu
    else:
        # 예측값을 사용
        X = df[feature_columns].copy()
        X = X[mask]
        X = sm.add_constant(X, prepend=True)
        mu = poisson_results.predict(X)
    pearson_resid = (y - mu) / np.sqrt(mu)
    phi = np.sum(pearson_resid**2) / (len(y) - len(feature_columns) - 1)
    
    print(f"Dispersion φ = {phi:.6f}")
    
    # 3. 모델 추천
    print("\\n3. 모델 추천:")
    if phi < 1.2:
        recommendation = "Poisson"
        print(f"φ < 1.2 → Poisson 모델 추천")
    elif 1.2 <= phi < 2:
        recommendation = "QuasiPoisson"
        print(f"1.2 ≤ φ < 2 → Quasi-Poisson 모델 추천")
    else:
        recommendation = "NegativeBinomial"
        print(f"φ ≥ 2 → Negative Binomial 모델 추천")
    
    # 4. 포아송 vs 음이항 AIC 비교
    print("\\n4. 포아송 vs 음이항 AIC 비교 (보조 기준):")
    poisson_aic = poisson_result['metrics'].get('AIC', None)
    print(f"Poisson AIC: {poisson_aic:.6f}" if poisson_aic else "Poisson AIC: N/A")
    
    print("음이항 모델 적합 중...")
    nb_result = fit_count_regression_statsmodels(
        df, 'NegativeBinomial', feature_columns, label_column, max_iter, 1.0
    )
    nb_aic = nb_result['metrics'].get('AIC', None)
    print(f"Negative Binomial AIC: {nb_aic:.6f}" if nb_aic else "Negative Binomial AIC: N/A")
    
    aic_comparison = None
    if poisson_aic is not None and nb_aic is not None:
        if nb_aic < poisson_aic:
            aic_comparison = "Negative Binomial이 더 낮은 AIC를 가짐 (더 나은 적합도)"
        else:
            aic_comparison = "Poisson이 더 낮은 AIC를 가짐 (더 나은 적합도)"
        print(f"AIC 비교: {aic_comparison}")
    
    # 5. Cameron–Trivedi test
    print("\\n5. Cameron–Trivedi test (최종 확인):")
    # mu는 이미 위에서 계산됨
    X_test = df[feature_columns].copy()
    X_test = X_test[mask]
    X_test = sm.add_constant(X_test, prepend=True)
    
    test_stat = (y - mu)**2 - y
    ct_model = sm.OLS(test_stat, X_test)
    ct_results = ct_model.fit()
    
    const_coef = ct_results.params.get('const', ct_results.params.iloc[0] if len(ct_results.params) > 0 else 0)
    const_pvalue = ct_results.pvalues.get('const', ct_results.pvalues.iloc[0] if len(ct_results.pvalues) > 0 else 1.0)
    
    print(f"Cameron–Trivedi test 통계량 (상수항 계수): {const_coef:.6f}")
    print(f"Cameron–Trivedi test p-value: {const_pvalue:.6f}")
    
    if const_pvalue < 0.05:
        ct_conclusion = "과대산포가 통계적으로 유의함 (p < 0.05)"
        print(f"결론: {ct_conclusion}")
    else:
        ct_conclusion = "과대산포가 통계적으로 유의하지 않음 (p ≥ 0.05)"
        print(f"결론: {ct_conclusion}")
    
    # 최종 추천
    print("\\n=== 최종 추천 ===")
    print(f"추천 모델: {recommendation}")
    if aic_comparison:
        print(f"AIC 비교: {aic_comparison}")
    print(f"Cameron–Trivedi test: {ct_conclusion}")
    
    return {
        'phi': float(phi),
        'recommendation': recommendation,
        'poisson_aic': float(poisson_aic) if poisson_aic is not None else None,
        'negative_binomial_aic': float(nb_aic) if nb_aic is not None else None,
        'aic_comparison': aic_comparison,
        'cameron_trivedi_coef': float(const_coef),
        'cameron_trivedi_pvalue': float(const_pvalue),
        'cameron_trivedi_conclusion': ct_conclusion,
        'methods_used': [
            '1. 포아송 모델 적합',
            '2. Dispersion φ 계산',
            '3. φ 기준 모델 추천',
            '4. 포아송 vs 음이항 AIC 비교',
            '5. Cameron–Trivedi test'
        ],
        'results': {
            'phi': float(phi),
            'phi_interpretation': f"φ = {phi:.6f}",
            'recommendation': recommendation,
            'poisson_aic': float(poisson_aic) if poisson_aic is not None else None,
            'negative_binomial_aic': float(nb_aic) if nb_aic is not None else None,
            'cameron_trivedi_coef': float(const_coef),
            'cameron_trivedi_pvalue': float(const_pvalue),
            'cameron_trivedi_conclusion': ct_conclusion
        }
    }

# 데이터 준비
dataframe = pd.DataFrame(js_data.to_py())
p_feature_columns = js_feature_columns.to_py()
p_label_column = str(js_label_column)
p_max_iter = int(js_max_iter)

# Execution
result = dispersion_checker(dataframe, p_feature_columns, p_label_column, p_max_iter)
print("\\n=== 분석 완료 ===")

json.dumps(result)
`;

    const resultJson = await withTimeout(
      py.runPython(code),
      timeoutMs,
      `DiversionChecker 실행 타임아웃 (${timeoutMs / 1000}초 초과)`
    );

    const result = JSON.parse(resultJson);

    return {
      phi: result.phi,
      recommendation: result.recommendation,
      poissonAic: result.poisson_aic,
      negativeBinomialAic: result.negative_binomial_aic,
      aicComparison: result.aic_comparison,
      cameronTrivediCoef: result.cameron_trivedi_coef,
      cameronTrivediPvalue: result.cameron_trivedi_pvalue,
      cameronTrivediConclusion: result.cameron_trivedi_conclusion,
      methodsUsed: result.methods_used,
      results: result.results,
    };
  } catch (error: any) {
    const errorMessage = error.message || String(error);
    throw new Error(`Python DiversionChecker error:\n${errorMessage}`);
  }
}

/**
 * Statistics를 Python으로 실행합니다
 * 타임아웃: 60초
 */
export async function calculateStatisticsPython(
  data: any[],
  columns: Array<{ name: string; type: string }>,
  timeoutMs: number = 60000
): Promise<{
  stats: Record<string, any>;
  correlation: Record<string, Record<string, number>>;
}> {
  try {
    // Pyodide 로드 (타임아웃: 30초)
    const py = await withTimeout(
      loadPyodide(30000),
      30000,
      "Pyodide 로딩 타임아웃 (30초 초과)"
    );

    // 데이터를 Python에 전달
    py.globals.set("js_data", data);
    py.globals.set("js_columns", columns);

    // Python 코드 실행
    const code = `
import json
import pandas as pd
import numpy as np

# 데이터 준비
df = pd.DataFrame(js_data.to_py())

# 기술 통계량 계산
stats = {}
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        desc = df[col].describe()
        nulls = df[col].isnull().sum()
        mode_val = df[col].mode()
        mode = float(mode_val.iloc[0]) if len(mode_val) > 0 else None
        
        stats[col] = {
            'count': int(desc.get('count', 0)),
            'mean': float(desc.get('mean', 0)),
            'std': float(desc.get('std', 0)),
            'min': float(desc.get('min', 0)),
            '25%': float(desc.get('25%', 0)),
            '50%': float(desc.get('50%', 0)),
            '75%': float(desc.get('75%', 0)),
            'max': float(desc.get('max', 0)),
            'nulls': int(nulls),
            'mode': mode if mode is not None else 'N/A',
            'variance': float(df[col].var()),
            'skewness': float(df[col].skew()) if len(df[col].dropna()) > 0 else 0.0,
            'kurtosis': float(df[col].kurtosis()) if len(df[col].dropna()) > 0 else 0.0
        }
    else:
        nulls = df[col].isnull().sum()
        mode_val = df[col].mode()
        mode = str(mode_val.iloc[0]) if len(mode_val) > 0 else 'N/A'
        
        stats[col] = {
            'count': len(df[col]),
            'mean': None,
            'std': None,
            'min': None,
            '25%': None,
            '50%': None,
            '75%': None,
            'max': None,
            'nulls': int(nulls),
            'mode': mode,
            'variance': None,
            'skewness': None,
            'kurtosis': None
        }

# 상관관계 계산
numeric_df = df.select_dtypes(include=[np.number])
correlation = {}
if len(numeric_df.columns) > 0:
    corr_matrix = numeric_df.corr()
    for col1 in corr_matrix.columns:
        correlation[col1] = {}
        for col2 in corr_matrix.columns:
            correlation[col1][col2] = float(corr_matrix.loc[col1, col2])
else:
    correlation = {}

result = {
    'stats': stats,
    'correlation': correlation
}

result
`;

    const resultPyObj = await withTimeout(
      Promise.resolve(py.runPython(code)),
      timeoutMs,
      "Python Statistics 실행 타임아웃 (60초 초과)"
    );

    // Python 딕셔너리를 JavaScript 객체로 변환
    const result = fromPython(resultPyObj);

    // 정리
    py.globals.delete("js_data");
    py.globals.delete("js_columns");

    return {
      stats: result.stats,
      correlation: result.correlation,
    };
  } catch (error: any) {
    // 정리
    try {
      const py = pyodide;
      if (py) {
        py.globals.delete("js_data");
        py.globals.delete("js_columns");
      }
    } catch {}

    const errorMessage = error.message || String(error);
    throw new Error(`Python Statistics error: ${errorMessage}`);
  }
}

/**
 * ScoreModel을 Python으로 실행합니다 (예측 수행)
 * 타임아웃: 60초
 */
export async function scoreModelPython(
  data: any[],
  featureColumns: string[],
  coefficients: Record<string, number>,
  intercept: number,
  labelColumn: string,
  modelPurpose: "classification" | "regression",
  timeoutMs: number = 60000
): Promise<{ rows: any[]; columns: Array<{ name: string; type: string }> }> {
  try {
    // Pyodide 로드 (타임아웃: 30초)
    const py = await withTimeout(
      loadPyodide(30000),
      30000,
      "Pyodide 로딩 타임아웃 (30초 초과)"
    );

    // 데이터를 Python에 전달
    py.globals.set("js_data", data);
    py.globals.set("js_feature_columns", featureColumns);
    py.globals.set("js_coefficients", coefficients);
    py.globals.set("js_intercept", intercept);
    py.globals.set("js_label_column", labelColumn);
    py.globals.set("js_model_purpose", modelPurpose);

    // Python 코드 실행
    const code = `
import json
import pandas as pd
import numpy as np

# 데이터 준비
df = pd.DataFrame(js_data.to_py())
feature_columns = js_feature_columns.to_py()
coefficients_dict = js_coefficients.to_py()
intercept = float(js_intercept)
label_column = str(js_label_column)
model_purpose = str(js_model_purpose)

# 특성 컬럼 순서대로 coefficients 배열 생성
coefficients_list = [coefficients_dict.get(col, 0.0) for col in feature_columns]

# 예측 수행
X = df[feature_columns].values
predictions = intercept + np.dot(X, coefficients_list)

# 결과 데이터프레임 생성
result_df = df.copy()
predict_col_name = "Predict"
result_df[predict_col_name] = predictions

# 분류 모델인 경우 확률 계산
if model_purpose == 'classification':
    # sigmoid 함수: 1 / (1 + exp(-x))
    probabilities_1 = 1.0 / (1.0 + np.exp(-predictions))
    probabilities_0 = 1 - probabilities_1
    final_predictions = (probabilities_1 > 0.5).astype(int)
    
    result_df[predict_col_name] = final_predictions
    result_df[f"{label_column}_Predict_Proba_0"] = probabilities_0
    result_df[f"{label_column}_Predict_Proba_1"] = probabilities_1

# 결과를 딕셔너리 리스트로 변환
result_rows = result_df.to_dict('records')
result_columns = [{'name': col, 'type': 'number' if pd.api.types.is_numeric_dtype(result_df[col]) else 'string'} for col in result_df.columns]

result = {
    'rows': result_rows,
    'columns': result_columns
}

result
`;

    const resultPyObj = await withTimeout(
      Promise.resolve(py.runPython(code)),
      timeoutMs,
      "Python ScoreModel 실행 타임아웃 (60초 초과)"
    );

    // Python 딕셔너리를 JavaScript 객체로 변환
    const result = fromPython(resultPyObj);

    // 정리
    py.globals.delete("js_data");
    py.globals.delete("js_feature_columns");
    py.globals.delete("js_coefficients");
    py.globals.delete("js_intercept");
    py.globals.delete("js_label_column");
    py.globals.delete("js_model_purpose");

    return {
      rows: result.rows,
      columns: result.columns,
    };
  } catch (error: any) {
    // 정리
    try {
      const py = pyodide;
      if (py) {
        py.globals.delete("js_data");
        py.globals.delete("js_feature_columns");
        py.globals.delete("js_coefficients");
        py.globals.delete("js_intercept");
        py.globals.delete("js_label_column");
        py.globals.delete("js_model_purpose");
      }
    } catch {}

    const errorMessage = error.message || String(error);
    throw new Error(`Python ScoreModel error: ${errorMessage}`);
  }
}

/**
 * EvaluateModel을 Python으로 실행합니다 (평가 메트릭 계산)
 * 타임아웃: 60초
 */
export async function evaluateModelPython(
  data: any[],
  labelColumn: string,
  predictionColumn: string,
  modelType: "classification" | "regression",
  threshold: number = 0.5,
  timeoutMs: number = 60000,
  calculateThresholdMetrics: boolean = false // 여러 threshold에 대한 precision/recall 계산 여부
): Promise<
  Record<string, number | string> & {
    thresholdMetrics?: Array<{
      threshold: number;
      accuracy: number;
      precision: number;
      recall: number;
      f1Score: number;
      tp: number;
      fp: number;
      tn: number;
      fn: number;
    }>;
  }
> {
  try {
    // Pyodide 로드 (타임아웃: 30초)
    const py = await withTimeout(
      loadPyodide(30000),
      30000,
      "Pyodide 로딩 타임아웃 (30초 초과)"
    );

    // 데이터를 Python에 전달
    py.globals.set("js_data", data);
    py.globals.set("js_label_column", labelColumn);
    py.globals.set("js_prediction_column", predictionColumn);
    py.globals.set("js_model_type", modelType);
    py.globals.set("js_threshold", threshold);
    py.globals.set("js_calculate_threshold_metrics", calculateThresholdMetrics);

    // Python 코드 실행
    const code = `
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score, confusion_matrix

# 데이터 준비
df = pd.DataFrame(js_data.to_py())
label_column = str(js_label_column)
prediction_column = str(js_prediction_column)
model_type = str(js_model_type)
threshold = float(js_threshold)
calculate_threshold_metrics = bool(js_calculate_threshold_metrics)

# 실제값 추출
y_true = df[label_column].values

metrics = {}
threshold_metrics_list = []

if model_type == 'classification':
    # 분류 모델: prediction_column이 확률값인 경우 threshold로 이진 분류 수행
    y_pred_proba = df[prediction_column].values
    
    # 확률값을 threshold 기반으로 이진 분류로 변환
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # 분류 메트릭 계산
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    recall = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    
    # 혼동 행렬
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tp = int(cm[1, 1])
        fp = int(cm[0, 1])
        tn = int(cm[0, 0])
        fn = int(cm[1, 0])
    else:
        tp = fp = tn = fn = 0
    
    metrics['Threshold'] = threshold
    metrics['Accuracy'] = accuracy
    metrics['Precision'] = precision
    metrics['Recall'] = recall
    metrics['F1-Score'] = f1
    metrics['Confusion Matrix'] = f"TP:{tp}, FP:{fp}, TN:{tn}, FN:{fn}"
    metrics['TP'] = tp
    metrics['FP'] = fp
    metrics['TN'] = tn
    metrics['FN'] = fn
    
    # 여러 threshold에 대한 모든 통계량 계산 (0부터 1까지 0.01 단위)
    if calculate_threshold_metrics:
        threshold_list = np.arange(0, 1.01, 0.01)
        for th in threshold_list:
            y_pred_th = (y_pred_proba >= th).astype(int)
            try:
                acc = float(accuracy_score(y_true, y_pred_th))
                prec = float(precision_score(y_true, y_pred_th, average='weighted', zero_division=0))
                rec = float(recall_score(y_true, y_pred_th, average='weighted', zero_division=0))
                f1 = float(f1_score(y_true, y_pred_th, average='weighted', zero_division=0))
                
                # 혼동 행렬
                cm_th = confusion_matrix(y_true, y_pred_th)
                if cm_th.shape == (2, 2):
                    tp_th = int(cm_th[1, 1])
                    fp_th = int(cm_th[0, 1])
                    tn_th = int(cm_th[0, 0])
                    fn_th = int(cm_th[1, 0])
                else:
                    tp_th = fp_th = tn_th = fn_th = 0
                
                threshold_metrics_list.append({
                    'threshold': float(th),
                    'accuracy': acc,
                    'precision': prec,
                    'recall': rec,
                    'f1Score': f1,
                    'tp': tp_th,
                    'fp': fp_th,
                    'tn': tn_th,
                    'fn': fn_th
                })
            except:
                # 에러 발생 시 스킵
                pass
        
        # threshold_metrics를 JSON 문자열로 변환하여 전달
        import json as json_module
        metrics['_threshold_metrics_json'] = json_module.dumps(threshold_metrics_list)
else:
    # 회귀 모델: prediction_column이 직접 예측값
    y_pred = df[prediction_column].values
    
    # 회귀 메트릭
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    
    metrics['Mean Squared Error (MSE)'] = mse
    metrics['Root Mean Squared Error (RMSE)'] = rmse
    metrics['Mean Absolute Error (MAE)'] = mae
    metrics['R-squared'] = r2

metrics
`;

    const resultPyObj = await withTimeout(
      Promise.resolve(py.runPython(code)),
      timeoutMs,
      "Python EvaluateModel 실행 타임아웃 (60초 초과)"
    );

    // Python 딕셔너리를 JavaScript 객체로 변환
    const metrics = fromPython(resultPyObj);

    // threshold_metrics가 있으면 파싱
    let thresholdMetrics:
      | Array<{
          threshold: number;
          accuracy: number;
          precision: number;
          recall: number;
          f1Score: number;
          tp: number;
          fp: number;
          tn: number;
          fn: number;
        }>
      | undefined = undefined;
    if (metrics["_threshold_metrics_json"]) {
      try {
        thresholdMetrics = JSON.parse(
          metrics["_threshold_metrics_json"] as string
        );
        delete metrics["_threshold_metrics_json"];
      } catch (e) {
        // 파싱 실패 시 무시
      }
    }

    // 정리
    py.globals.delete("js_data");
    py.globals.delete("js_label_column");
    py.globals.delete("js_prediction_column");
    py.globals.delete("js_model_type");
    py.globals.delete("js_threshold");
    py.globals.delete("js_calculate_threshold_metrics");

    return { ...metrics, thresholdMetrics };
  } catch (error: any) {
    // 정리
    try {
      const py = pyodide;
      if (py) {
        py.globals.delete("js_data");
        py.globals.delete("js_label_column");
        py.globals.delete("js_prediction_column");
        py.globals.delete("js_model_type");
        py.globals.delete("js_threshold");
      }
    } catch {}

    const errorMessage = error.message || String(error);
    throw new Error(`Python EvaluateModel error: ${errorMessage}`);
  }
}

/**
 * PredictModel을 Python으로 실행합니다 (statsmodels 모델 예측)
 * 타임아웃: 120초
 */
export async function predictWithStatsmodel(
  data: any[],
  featureColumns: string[],
  coefficients: Record<string, { coef: number }>,
  modelType: string,
  timeoutMs: number = 120000
): Promise<{ rows: any[]; columns: Array<{ name: string; type: string }> }> {
  try {
    // Pyodide 로드 (타임아웃: 30초)
    const py = await withTimeout(
      loadPyodide(30000),
      30000,
      "Pyodide 로딩 타임아웃 (30초 초과)"
    );

    // statsmodels 패키지 로드
    await withTimeout(
      py.loadPackage(["statsmodels"]),
      60000,
      "statsmodels 패키지 설치 타임아웃 (60초 초과)"
    );

    // 데이터를 Python에 전달
    py.globals.set("js_data", data);
    py.globals.set("js_feature_columns", featureColumns);
    py.globals.set("js_coefficients", coefficients);
    py.globals.set("js_model_type", modelType);

    // Python 코드 실행
    const code = `
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

# 데이터 준비
df = pd.DataFrame(js_data.to_py())
feature_columns = js_feature_columns.to_py()
coefficients_dict = js_coefficients.to_py()
model_type = str(js_model_type)

# 특성 컬럼만 선택
X = df[feature_columns].copy()

# 상수항 추가 (모델 피팅 시와 동일한 방식)
X = sm.add_constant(X, prepend=True, has_constant='add')

# 계수 정보를 사용하여 모델 재구성
# statsmodels의 predict를 정확히 재현하기 위해 모델을 다시 피팅하는 대신
# 선형 예측자를 직접 계산하고 link function을 적용

# 계수 딕셔너리에서 실제 키 이름 확인 (statsmodels가 생성한 이름 사용)
# 상수항 이름 확인 ('const' 또는 다른 이름)
const_name = None
for key in coefficients_dict.keys():
    if key.lower() in ['const', 'intercept']:
        const_name = key
        break
if const_name is None:
    # 첫 번째 키가 상수항일 가능성이 높음
    const_name = list(coefficients_dict.keys())[0] if coefficients_dict else 'const'

# exog_names 생성: 상수항 + feature_columns (계수 딕셔너리의 키 순서 사용)
# 하지만 feature_columns 순서를 우선하고, 계수 딕셔너리에 있는 것만 사용
exog_names = [const_name] + feature_columns

# 계수 배열 생성 (exog_names 순서에 맞춰)
coef_array = []
for name in exog_names:
    if name in coefficients_dict:
        coef_value = coefficients_dict[name].get('coef', 0.0) if isinstance(coefficients_dict[name], dict) else coefficients_dict[name]
        coef_array.append(float(coef_value))
    else:
        coef_array.append(0.0)
coef_array = np.array(coef_array)

# X를 exog_names 순서에 맞춰 정렬
X_aligned = X.reindex(columns=exog_names).fillna(0)

# 선형 예측자 계산
linear_predictor = np.dot(X_aligned.values, coef_array)

# 모델 타입에 따라 link function 적용
if model_type == 'OLS':
    predictions = linear_predictor
elif model_type == 'Logistic' or model_type == 'Logit':
    # Logit: exp(x) / (1 + exp(x))
    predictions = 1.0 / (1.0 + np.exp(-linear_predictor))
elif model_type in ['Poisson', 'NegativeBinomial', 'QuasiPoisson']:
    # Count regression: exp(x)
    predictions = np.exp(linear_predictor)
elif model_type == 'Gamma':
    # Gamma GLM: exp(x) (log link)
    predictions = np.exp(linear_predictor)
elif model_type == 'Tweedie':
    # Tweedie GLM: exp(x) (log link)
    predictions = np.exp(linear_predictor)
else:
    # 기본값: 선형 예측자
    predictions = linear_predictor

# 결과 데이터프레임 생성
result_df = df.copy()
result_df['Predict'] = predictions

# 정수 예측값 컬럼 추가 (Logistic, Poisson, NegativeBinomial, QuasiPoisson 모델)
if model_type in ['Logistic', 'Logit', 'Poisson', 'NegativeBinomial', 'QuasiPoisson']:
    # 가장 가까운 정수로 반올림
    predictions_int = np.round(predictions).astype(int)
    # 음수는 0으로 제한 (count 모델의 경우)
    if model_type in ['Poisson', 'NegativeBinomial', 'QuasiPoisson']:
        predictions_int = np.maximum(predictions_int, 0)
    # Logistic의 경우 0 또는 1로 제한
    elif model_type in ['Logistic', 'Logit']:
        predictions_int = np.clip(predictions_int, 0, 1)
    result_df['y_Pred'] = predictions_int

# 결과를 딕셔너리 리스트로 변환
result_rows = result_df.to_dict('records')
result_columns = [{'name': col, 'type': 'number' if pd.api.types.is_numeric_dtype(result_df[col]) else 'string'} for col in result_df.columns]

result = {
    'rows': result_rows,
    'columns': result_columns
}

result
`;

    const resultPyObj = await withTimeout(
      Promise.resolve(py.runPython(code)),
      timeoutMs,
      "Python PredictModel 실행 타임아웃 (120초 초과)"
    );

    // Python 딕셔너리를 JavaScript 객체로 변환
    const result = fromPython(resultPyObj);

    // 정리
    py.globals.delete("js_data");
    py.globals.delete("js_feature_columns");
    py.globals.delete("js_coefficients");
    py.globals.delete("js_model_type");

    return result;
  } catch (error: any) {
    const errorMessage = error.message || String(error);
    throw new Error(`Python PredictModel error: ${errorMessage}`);
  }
}

/**
 * EvaluateStats를 Python으로 실행합니다 (GLM 통계량 평가)
 * 타임아웃: 120초
 */
export async function evaluateStatsPython(
  data: any[],
  labelColumn: string,
  predictionColumn: string,
  modelType: string,
  timeoutMs: number = 120000
): Promise<{
  metrics: Record<string, number | string>;
  residuals?: number[];
  deviance?: number;
  pearsonChi2?: number;
  dispersion?: number;
  aic?: number;
  bic?: number;
  logLikelihood?: number;
}> {
  try {
    // Pyodide 로드 (타임아웃: 30초)
    const py = await withTimeout(
      loadPyodide(30000),
      30000,
      "Pyodide 로딩 타임아웃 (30초 초과)"
    );

    // statsmodels 패키지 로드
    await withTimeout(
      py.loadPackage(["statsmodels", "scipy"]),
      60000,
      "statsmodels 패키지 설치 타임아웃 (60초 초과)"
    );

    // 데이터를 Python에 전달
    py.globals.set("js_data", data);
    py.globals.set("js_label_column", labelColumn);
    py.globals.set("js_prediction_column", predictionColumn);
    py.globals.set("js_model_type", modelType);

    // Python 코드 실행
    const code = `
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# 데이터 준비
df = pd.DataFrame(js_data.to_py())
label_column = str(js_label_column)
prediction_column = str(js_prediction_column)
model_type = str(js_model_type)

# 실제값과 예측값 추출
y_true = df[label_column].values
y_pred = df[prediction_column].values

# 기본 메트릭 계산
metrics = {}
residuals = None
deviance = None
pearson_chi2 = None
dispersion = None
aic = None
bic = None
log_likelihood = None

# 잔차 계산
residuals = (y_true - y_pred).tolist()

# 기본 회귀 메트릭
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = float(mean_squared_error(y_true, y_pred))
rmse = float(np.sqrt(mse))
mae = float(mean_absolute_error(y_true, y_pred))
r2 = float(r2_score(y_true, y_pred))

metrics['Mean Squared Error (MSE)'] = mse
metrics['Root Mean Squared Error (RMSE)'] = rmse
metrics['Mean Absolute Error (MAE)'] = mae
metrics['R-squared'] = r2

# 모델 타입에 따른 추가 통계량 (선택적)
# 모델 타입이 제공되면 해당 모델의 특수 통계량 계산
if model_type and model_type != '' and model_type != 'None':
    if model_type in ['Poisson', 'NegativeBinomial', 'QuasiPoisson']:
        # Count regression 모델 통계량
        # Deviance 계산
        if model_type == 'Poisson':
            # Poisson deviance: 2 * sum(y * log(y/mu) - (y - mu))
            mu = np.maximum(y_pred, 1e-10)  # 0 방지
            deviance_val = 2 * np.sum(y_true * np.log(np.maximum(y_true, 1e-10) / mu) - (y_true - mu))
            deviance = float(deviance_val)
            
            # Pearson chi2
            pearson_resid = (y_true - mu) / np.sqrt(mu)
            pearson_chi2_val = np.sum(pearson_resid ** 2)
            pearson_chi2 = float(pearson_chi2_val)
            
            # Dispersion (phi)
            n = len(y_true)
            p = 1  # 간단히 1로 가정 (실제로는 모델의 파라미터 수)
            dispersion_val = pearson_chi2_val / (n - p) if (n - p) > 0 else 1.0
            dispersion = float(dispersion_val)
            
            # Log-likelihood (Poisson)
            log_likelihood_val = np.sum(stats.poisson.logpmf(y_true, mu))
            log_likelihood = float(log_likelihood_val)
            
            # AIC, BIC (근사치)
            aic = float(-2 * log_likelihood_val + 2 * p)
            bic = float(-2 * log_likelihood_val + np.log(n) * p)
            
        elif model_type in ['NegativeBinomial', 'QuasiPoisson']:
            # Negative Binomial / Quasi-Poisson 통계량
            mu = np.maximum(y_pred, 1e-10)
            deviance_val = 2 * np.sum(y_true * np.log(np.maximum(y_true, 1e-10) / mu) - (y_true - mu))
            deviance = float(deviance_val)
            
            pearson_resid = (y_true - mu) / np.sqrt(mu)
            pearson_chi2_val = np.sum(pearson_resid ** 2)
            pearson_chi2 = float(pearson_chi2_val)
            
            n = len(y_true)
            p = 1
            dispersion_val = pearson_chi2_val / (n - p) if (n - p) > 0 else 1.0
            dispersion = float(dispersion_val)

    elif model_type in ['Logistic', 'Logit']:
        # Logistic regression 통계량
        # Deviance (binomial deviance)
        y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
        y_true_clipped = np.clip(y_true, 1e-10, 1 - 1e-10)
        deviance_val = -2 * np.sum(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        deviance = float(deviance_val)
        
        # Pearson chi2
        pearson_resid = (y_true - y_pred) / np.sqrt(y_pred * (1 - y_pred) + 1e-10)
        pearson_chi2_val = np.sum(pearson_resid ** 2)
        pearson_chi2 = float(pearson_chi2_val)
        
        # Log-likelihood
        log_likelihood_val = np.sum(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        log_likelihood = float(log_likelihood_val)
        
        n = len(y_true)
        p = 1
        aic = float(-2 * log_likelihood_val + 2 * p)
        bic = float(-2 * log_likelihood_val + np.log(n) * p)

    elif model_type == 'OLS':
        # OLS 통계량
        # Deviance (residual sum of squares)
        deviance_val = np.sum((y_true - y_pred) ** 2)
        deviance = float(deviance_val)
        
        # Log-likelihood (normal distribution)
        n = len(y_true)
        sigma2 = deviance_val / n if n > 0 else 1.0
        log_likelihood_val = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
        log_likelihood = float(log_likelihood_val)
        
        p = 1
        aic = float(-2 * log_likelihood_val + 2 * p)
        bic = float(-2 * log_likelihood_val + np.log(n) * p)

# 메트릭에 추가
if deviance is not None:
    metrics['Deviance'] = deviance
if pearson_chi2 is not None:
    metrics['Pearson chi2'] = pearson_chi2
if dispersion is not None:
    metrics['Dispersion (phi)'] = dispersion
if aic is not None:
    metrics['AIC'] = aic
if bic is not None:
    metrics['BIC'] = bic
if log_likelihood is not None:
    metrics['Log-Likelihood'] = log_likelihood

# 잔차 통계량
if residuals is not None:
    residuals_array = np.array(residuals)
    metrics['Mean Residual'] = float(np.mean(residuals_array))
    metrics['Std Residual'] = float(np.std(residuals_array))
    metrics['Min Residual'] = float(np.min(residuals_array))
    metrics['Max Residual'] = float(np.max(residuals_array))

result = {
    'metrics': metrics,
    'residuals': residuals,
    'deviance': deviance,
    'pearsonChi2': pearson_chi2,
    'dispersion': dispersion,
    'aic': aic,
    'bic': bic,
    'logLikelihood': log_likelihood
}

result
`;

    const resultPyObj = await withTimeout(
      Promise.resolve(py.runPython(code)),
      timeoutMs,
      "Python EvaluateStats 실행 타임아웃 (120초 초과)"
    );

    // Python 딕셔너리를 JavaScript 객체로 변환
    const result = fromPython(resultPyObj);

    // 정리
    py.globals.delete("js_data");
    py.globals.delete("js_label_column");
    py.globals.delete("js_prediction_column");
    py.globals.delete("js_model_type");

    return result;
  } catch (error: any) {
    const errorMessage = error.message || String(error);
    throw new Error(`Python EvaluateStats error: ${errorMessage}`);
  }
}

/**
 * HandleMissingValues를 Python으로 실행합니다 (결측치 처리 통계 계산)
 * 타임아웃: 60초
 */
export async function handleMissingValuesPython(
  data: any[],
  method: string,
  strategy: string,
  columns: string[] | null,
  n_neighbors: number,
  timeoutMs: number = 60000
): Promise<{ imputation_values: Record<string, number | string> }> {
  try {
    const py = await withTimeout(
      loadPyodide(30000),
      30000,
      "Pyodide 로딩 타임아웃 (30초 초과)"
    );

    py.globals.set("js_data", data);
    py.globals.set("js_method", method);
    py.globals.set("js_strategy", strategy);
    py.globals.set("js_columns", columns);
    py.globals.set("js_n_neighbors", n_neighbors);

    const code = `
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

df = pd.DataFrame(js_data.to_py())
method = str(js_method)
strategy = str(js_strategy)
columns = js_columns.to_py() if js_columns else None
n_neighbors = int(js_n_neighbors)

imputation_values = {}

if method == 'impute' or method == 'knn':
    cols_to_process = columns if columns else df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in cols_to_process:
        if col not in df.columns:
            continue
        
        if df[col].dtype in ['int64', 'float64']:
            non_null_values = df[col].dropna()
            if len(non_null_values) == 0:
                continue
            
            if method == 'knn' or strategy == 'mean':
                imputation_values[col] = float(non_null_values.mean())
            elif strategy == 'median':
                imputation_values[col] = float(non_null_values.median())
            elif strategy == 'mode':
                mode_val = non_null_values.mode()
                imputation_values[col] = float(mode_val.iloc[0]) if len(mode_val) > 0 else 0.0
            else:
                imputation_values[col] = float(non_null_values.mean())
        else:
            # 문자열/범주형 컬럼은 mode 사용
            non_null_values = df[col].dropna()
            if len(non_null_values) == 0:
                continue
            mode_val = non_null_values.mode()
            imputation_values[col] = str(mode_val.iloc[0]) if len(mode_val) > 0 else ''

result = {
    'imputation_values': imputation_values
}

result
`;

    const resultPyObj = await withTimeout(
      Promise.resolve(py.runPython(code)),
      timeoutMs,
      "Python HandleMissingValues 실행 타임아웃 (60초 초과)"
    );

    const result = fromPython(resultPyObj);

    py.globals.delete("js_data");
    py.globals.delete("js_method");
    py.globals.delete("js_strategy");
    py.globals.delete("js_columns");
    py.globals.delete("js_n_neighbors");

    return result;
  } catch (error: any) {
    try {
      const py = pyodide;
      if (py) {
        py.globals.delete("js_data");
        py.globals.delete("js_method");
        py.globals.delete("js_strategy");
        py.globals.delete("js_columns");
        py.globals.delete("js_n_neighbors");
      }
    } catch {}

    const errorMessage = error.message || String(error);
    throw new Error(`Python HandleMissingValues error: ${errorMessage}`);
  }
}

/**
 * NormalizeData를 Python으로 실행합니다 (정규화 통계 계산)
 * 타임아웃: 60초
 */
export async function normalizeDataPython(
  data: any[],
  method: string,
  columns: string[],
  timeoutMs: number = 60000
): Promise<{
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
}> {
  try {
    const py = await withTimeout(
      loadPyodide(30000),
      30000,
      "Pyodide 로딩 타임아웃 (30초 초과)"
    );

    py.globals.set("js_data", data);
    py.globals.set("js_method", method);
    py.globals.set("js_columns", columns);

    const code = `
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

df = pd.DataFrame(js_data.to_py())
method = str(js_method)
columns = js_columns.to_py()

stats = {}

for col in columns:
    if col not in df.columns:
        continue
    
    if df[col].dtype not in ['int64', 'float64']:
        continue
    
    values = df[col].dropna()
    if len(values) == 0:
        continue
    
    col_stats = {}
    
    if method == 'MinMax':
        col_stats['min'] = float(values.min())
        col_stats['max'] = float(values.max())
    elif method == 'StandardScaler':
        col_stats['mean'] = float(values.mean())
        col_stats['stdDev'] = float(values.std())
    elif method == 'RobustScaler':
        col_stats['median'] = float(values.median())
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        col_stats['iqr'] = float(q3 - q1)
    
    stats[col] = col_stats

result = {
    'stats': stats
}

result
`;

    const resultPyObj = await withTimeout(
      Promise.resolve(py.runPython(code)),
      timeoutMs,
      "Python NormalizeData 실행 타임아웃 (60초 초과)"
    );

    const result = fromPython(resultPyObj);

    py.globals.delete("js_data");
    py.globals.delete("js_method");
    py.globals.delete("js_columns");

    return result;
  } catch (error: any) {
    try {
      const py = pyodide;
      if (py) {
        py.globals.delete("js_data");
        py.globals.delete("js_method");
        py.globals.delete("js_columns");
      }
    } catch {}

    const errorMessage = error.message || String(error);
    throw new Error(`Python NormalizeData error: ${errorMessage}`);
  }
}

/**
 * TransitionData를 Python으로 실행합니다 (수학적 변환)
 * 타임아웃: 60초
 */
export async function transformDataPython(
  data: any[],
  transformations: Record<string, string>,
  timeoutMs: number = 60000
): Promise<{ rows: any[]; columns: Array<{ name: string; type: string }> }> {
  try {
    const py = await withTimeout(
      loadPyodide(30000),
      30000,
      "Pyodide 로딩 타임아웃 (30초 초과)"
    );

    py.globals.set("js_data", data);
    py.globals.set("js_transformations", transformations);

    const code = `
import pandas as pd
import numpy as np

df = pd.DataFrame(js_data.to_py())
transformations = js_transformations.to_py()

df_transformed = df.copy()

for col, method in transformations.items():
    if method == 'None' or col not in df_transformed.columns:
        continue
    
    if not pd.api.types.is_numeric_dtype(df_transformed[col]):
        continue
    
    new_col_name = f"{col}_{method.lower().replace(' ', '_').replace('-', '_')}"
    
    if method == 'Log':
        df_transformed[new_col_name] = np.log(df_transformed[col].apply(lambda x: x if x > 0 else np.nan))
        df_transformed[new_col_name].fillna(0, inplace=True)
    elif method == 'Square Root':
        df_transformed[new_col_name] = np.sqrt(df_transformed[col].apply(lambda x: x if x >= 0 else np.nan))
        df_transformed[new_col_name].fillna(0, inplace=True)
    elif method == 'Min-Log':
        min_val = df_transformed[col].min()
        df_transformed[new_col_name] = np.log((df_transformed[col] - min_val) + 1)
    elif method == 'Min-Square Root':
        min_val = df_transformed[col].min()
        df_transformed[new_col_name] = np.sqrt((df_transformed[col] - min_val) + 1)

result_rows = df_transformed.to_dict('records')
result_columns = [{'name': col, 'type': 'number' if pd.api.types.is_numeric_dtype(df_transformed[col]) else 'string'} for col in df_transformed.columns]

result = {
    'rows': result_rows,
    'columns': result_columns
}

result
`;

    const resultPyObj = await withTimeout(
      Promise.resolve(py.runPython(code)),
      timeoutMs,
      "Python TransitionData 실행 타임아웃 (60초 초과)"
    );

    const result = fromPython(resultPyObj);

    py.globals.delete("js_data");
    py.globals.delete("js_transformations");

    return result;
  } catch (error: any) {
    try {
      const py = pyodide;
      if (py) {
        py.globals.delete("js_data");
        py.globals.delete("js_transformations");
      }
    } catch {}

    const errorMessage = error.message || String(error);
    throw new Error(`Python TransitionData error: ${errorMessage}`);
  }
}

/**
 * EncodeCategorical를 Python으로 실행합니다 (인코딩 매핑 생성)
 * 타임아웃: 60초
 */
export async function encodeCategoricalPython(
  data: any[],
  method: string,
  columns: string[] | null,
  ordinal_mapping: Record<string, string[]> | null,
  drop: string,
  handle_unknown: string,
  timeoutMs: number = 60000
): Promise<{ mappings: Record<string, Record<string, number> | string[]> }> {
  try {
    const py = await withTimeout(
      loadPyodide(30000),
      30000,
      "Pyodide 로딩 타임아웃 (30초 초과)"
    );

    py.globals.set("js_data", data);
    py.globals.set("js_method", method);
    py.globals.set("js_columns", columns);
    py.globals.set("js_ordinal_mapping", ordinal_mapping);
    py.globals.set("js_drop", drop);
    py.globals.set("js_handle_unknown", handle_unknown);

    const code = `
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.DataFrame(js_data.to_py())
method = str(js_method)
columns = js_columns.to_py() if js_columns else None
ordinal_mapping = js_ordinal_mapping.to_py() if js_ordinal_mapping else None
drop = str(js_drop)
handle_unknown = str(js_handle_unknown)

if columns is None:
    columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

mappings = {}

if method == 'label':
    for col in columns:
        if col not in df.columns:
            continue
        unique_values = sorted(df[col].dropna().unique())
        col_mapping = {val: idx for idx, val in enumerate(unique_values)}
        mappings[col] = col_mapping
elif method == 'one_hot':
    for col in columns:
        if col not in df.columns:
            continue
        unique_values = sorted(df[col].dropna().unique())
        mappings[col] = unique_values
elif method == 'ordinal':
    for col in columns:
        if col not in df.columns:
            continue
        if ordinal_mapping and col in ordinal_mapping:
            mapping_list = ordinal_mapping[col]
            col_mapping = {val: idx for idx, val in enumerate(mapping_list)}
            mappings[col] = col_mapping
        else:
            unique_values = sorted(df[col].dropna().unique())
            col_mapping = {val: idx for idx, val in enumerate(unique_values)}
            mappings[col] = col_mapping

result = {
    'mappings': mappings
}

result
`;

    const resultPyObj = await withTimeout(
      Promise.resolve(py.runPython(code)),
      timeoutMs,
      "Python EncodeCategorical 실행 타임아웃 (60초 초과)"
    );

    const result = fromPython(resultPyObj);

    py.globals.delete("js_data");
    py.globals.delete("js_method");
    py.globals.delete("js_columns");
    py.globals.delete("js_ordinal_mapping");
    py.globals.delete("js_drop");
    py.globals.delete("js_handle_unknown");

    return result;
  } catch (error: any) {
    try {
      const py = pyodide;
      if (py) {
        py.globals.delete("js_data");
        py.globals.delete("js_method");
        py.globals.delete("js_columns");
        py.globals.delete("js_ordinal_mapping");
        py.globals.delete("js_drop");
        py.globals.delete("js_handle_unknown");
      }
    } catch {}

    const errorMessage = error.message || String(error);
    throw new Error(`Python EncodeCategorical error: ${errorMessage}`);
  }
}

/**
 * ResampleData를 Python으로 실행합니다 (리샘플링)
 * 타임아웃: 60초
 */
export async function resampleDataPython(
  data: any[],
  method: string,
  target_column: string,
  timeoutMs: number = 60000
): Promise<{ rows: any[]; columns: Array<{ name: string; type: string }> }> {
  try {
    const py = await withTimeout(
      loadPyodide(30000),
      30000,
      "Pyodide 로딩 타임아웃 (30초 초과)"
    );

    py.globals.set("js_data", data);
    py.globals.set("js_method", method);
    py.globals.set("js_target_column", target_column);

    const code = `
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

df = pd.DataFrame(js_data.to_py())
method = str(js_method)
target_column = str(js_target_column)

if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found in dataframe")

X = df.drop(columns=[target_column])
y = df[target_column]

if method == 'SMOTE':
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
elif method == 'NearMiss':
    near_miss = NearMiss(version=1)
    X_resampled, y_resampled = near_miss.fit_resample(X, y)
else:
    raise ValueError(f"Unknown resampling method: {method}")

df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled[target_column] = y_resampled

result_rows = df_resampled.to_dict('records')
result_columns = [{'name': col, 'type': 'number' if pd.api.types.is_numeric_dtype(df_resampled[col]) else 'string'} for col in df_resampled.columns]

result = {
    'rows': result_rows,
    'columns': result_columns
}

result
`;

    const resultPyObj = await withTimeout(
      Promise.resolve(py.runPython(code)),
      timeoutMs,
      "Python ResampleData 실행 타임아웃 (60초 초과)"
    );

    const result = fromPython(resultPyObj);

    py.globals.delete("js_data");
    py.globals.delete("js_method");
    py.globals.delete("js_target_column");

    return result;
  } catch (error: any) {
    try {
      const py = pyodide;
      if (py) {
        py.globals.delete("js_data");
        py.globals.delete("js_method");
        py.globals.delete("js_target_column");
      }
    } catch {}

    const errorMessage = error.message || String(error);
    throw new Error(`Python ResampleData error: ${errorMessage}`);
  }
}

/**
 * TransformData를 Python으로 실행합니다 (변환 적용)
 * 타임아웃: 60초
 */
export async function applyTransformPython(
  data: any[],
  handler: any,
  exclude_columns: string[],
  timeoutMs: number = 60000
): Promise<{ rows: any[]; columns: Array<{ name: string; type: string }> }> {
  try {
    const py = await withTimeout(
      loadPyodide(30000),
      30000,
      "Pyodide 로딩 타임아웃 (30초 초과)"
    );

    py.globals.set("js_data", data);
    py.globals.set("js_handler", handler);
    py.globals.set("js_exclude_columns", exclude_columns);

    const code = `
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder

df = pd.DataFrame(js_data.to_py())
handler = js_handler.to_py()
exclude_columns = js_exclude_columns.to_py() if js_exclude_columns else []

df_result = df.copy()

if handler.get('type') == 'MissingHandlerOutput':
    method = handler.get('method')
    imputation_values = handler.get('imputation_values', {})
    
    if method == 'remove_row':
        df_result = df_result.dropna()
    elif method == 'impute' or method == 'knn':
        for col_name, fill_value in imputation_values.items():
            if col_name in exclude_columns:
                continue
            if col_name in df_result.columns:
                df_result[col_name].fillna(fill_value, inplace=True)

elif handler.get('type') == 'NormalizerOutput':
    method = handler.get('method')
    stats = handler.get('stats', {})
    
    for col_name, col_stats in stats.items():
        if col_name in exclude_columns or col_name not in df_result.columns:
            continue
        
        if df_result[col_name].dtype not in ['int64', 'float64']:
            continue
        
        values = df_result[col_name].values
        
        if method == 'MinMax' and 'min' in col_stats and 'max' in col_stats:
            min_val = col_stats['min']
            max_val = col_stats['max']
            range_val = max_val - min_val
            if range_val > 0:
                df_result[col_name] = (values - min_val) / range_val
            else:
                df_result[col_name] = 0.5
        elif method == 'StandardScaler' and 'mean' in col_stats and 'stdDev' in col_stats:
            mean_val = col_stats['mean']
            std_val = col_stats['stdDev']
            if std_val > 0:
                df_result[col_name] = (values - mean_val) / std_val
            else:
                df_result[col_name] = 0.0
        elif method == 'RobustScaler' and 'median' in col_stats and 'iqr' in col_stats:
            median_val = col_stats['median']
            iqr_val = col_stats['iqr']
            if iqr_val > 0:
                df_result[col_name] = (values - median_val) / iqr_val
            else:
                df_result[col_name] = 0.0

elif handler.get('type') == 'EncoderOutput':
    method = handler.get('method')
    mappings = handler.get('mappings', {})
    drop = handler.get('drop', 'first')
    
    if method == 'label' or method == 'ordinal':
        for col_name, mapping in mappings.items():
            if col_name in exclude_columns or col_name not in df_result.columns:
                continue
            df_result[col_name] = df_result[col_name].map(mapping)
            df_result[col_name] = df_result[col_name].astype('float64')
    elif method == 'one_hot':
        for col_name, unique_values in mappings.items():
            if col_name in exclude_columns or col_name not in df_result.columns:
                continue
            
            values_to_create = list(unique_values)
            if drop == 'first' and len(values_to_create) > 0:
                values_to_create = values_to_create[1:]
            elif drop == 'if_binary' and len(values_to_create) == 2:
                values_to_create = values_to_create[1:]
            
            for value in values_to_create:
                new_col_name = f"{col_name}_{value}"
                df_result[new_col_name] = (df_result[col_name] == value).astype(int)
            
            df_result = df_result.drop(columns=[col_name])

result_rows = df_result.to_dict('records')
result_columns = [{'name': col, 'type': 'number' if pd.api.types.is_numeric_dtype(df_result[col]) else 'string'} for col in df_result.columns]

result = {
    'rows': result_rows,
    'columns': result_columns
}

result
`;

    const resultPyObj = await withTimeout(
      Promise.resolve(py.runPython(code)),
      timeoutMs,
      "Python TransformData 실행 타임아웃 (60초 초과)"
    );

    const result = fromPython(resultPyObj);

    py.globals.delete("js_data");
    py.globals.delete("js_handler");
    py.globals.delete("js_exclude_columns");

    return result;
  } catch (error: any) {
    try {
      const py = pyodide;
      if (py) {
        py.globals.delete("js_data");
        py.globals.delete("js_handler");
        py.globals.delete("js_exclude_columns");
      }
    } catch {}

    const errorMessage = error.message || String(error);
    throw new Error(`Python TransformData error: ${errorMessage}`);
  }
}

/**
 * Diversion Checker (과대산포 검사)를 실행합니다
 */
export async function dispersionCheckerPython(
  rows: Record<string, any>[],
  featureColumns: string[],
  labelColumn: string,
  maxIter: number = 100,
  timeoutMs: number = 120000
): Promise<{
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
}> {
  try {
    const py = await loadPyodide(30000);

    await withTimeout(
      py.loadPackage(["statsmodels", "pandas", "numpy"]),
      60000,
      "statsmodels 패키지 설치 타임아웃 (60초 초과)"
    );

    // data_analysis_modules.py 파일을 로드
    const response = await fetch("/data_analysis_modules.py");
    const pythonCode = await response.text();
    py.runPython(pythonCode);

    py.globals.set("js_rows", rows);
    py.globals.set("js_feature_columns", featureColumns);
    py.globals.set("js_label_column", labelColumn);
    py.globals.set("js_max_iter", maxIter);

    const code = `
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import traceback
import sys

try:
    # JavaScript에서 전달된 데이터를 DataFrame으로 변환
    rows = js_rows.to_py()
    feature_columns = js_feature_columns.to_py()
    label_column = str(js_label_column)
    max_iter = int(js_max_iter)
    
    if not rows or len(rows) == 0:
        raise ValueError("입력 데이터가 비어있습니다.")
    
    df = pd.DataFrame(rows)
    
    # NaN 값 처리 (0으로 대체)
    df = df.fillna(0)
    
    # dispersion_checker 함수 호출
    result = dispersion_checker(df, feature_columns, label_column, max_iter)
    
    # 결과를 JavaScript로 전달할 형식으로 변환
    js_result = {
        'phi': float(result['phi']),
        'recommendation': str(result['recommendation']),
        'poisson_aic': float(result['poisson_aic']) if result['poisson_aic'] is not None else None,
        'negative_binomial_aic': float(result['negative_binomial_aic']) if result['negative_binomial_aic'] is not None else None,
        'aic_comparison': str(result['aic_comparison']) if result['aic_comparison'] is not None else None,
        'cameron_trivedi_coef': float(result['cameron_trivedi_coef']),
        'cameron_trivedi_pvalue': float(result['cameron_trivedi_pvalue']),
        'cameron_trivedi_conclusion': str(result['cameron_trivedi_conclusion']),
        'methods_used': result['methods_used'],
        'results': result['results']
    }
except Exception as e:
    error_traceback = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
    error_result = {
        '__error__': True,
        'error_type': type(e).__name__,
        'error_message': str(e),
        'error_traceback': error_traceback
    }
    js_result = error_result
`;

    await withTimeout(
      Promise.resolve(py.runPython(code)),
      timeoutMs,
      `Diversion Checker 실행 타임아웃 (${timeoutMs / 1000}초 초과)`
    );

    const resultPyObj = py.globals.get("js_result");
    if (!resultPyObj) {
      throw new Error("Python 코드가 결과를 반환하지 않았습니다.");
    }

    const result = fromPython(resultPyObj);

    if (result.__error__) {
      throw new Error(
        `Python Diversion Checker error:\n${
          result.error_traceback || result.error_message || "Unknown error"
        }`
      );
    }

    if (!result.phi || !result.recommendation) {
      throw new Error("Python Diversion Checker error: Invalid result structure");
    }

    py.globals.delete("js_rows");
    py.globals.delete("js_feature_columns");
    py.globals.delete("js_label_column");
    py.globals.delete("js_max_iter");
    py.globals.delete("js_result");

    return {
      phi: result.phi,
      recommendation: result.recommendation as "Poisson" | "QuasiPoisson" | "NegativeBinomial",
      poissonAic: result.poisson_aic,
      negativeBinomialAic: result.negative_binomial_aic,
      aicComparison: result.aic_comparison,
      cameronTrivediCoef: result.cameron_trivedi_coef,
      cameronTrivediPvalue: result.cameron_trivedi_pvalue,
      cameronTrivediConclusion: result.cameron_trivedi_conclusion,
      methodsUsed: result.methods_used,
      results: result.results,
    };
  } catch (error: any) {
    try {
      if (pyodide) {
        pyodide.globals.delete("js_rows");
        pyodide.globals.delete("js_feature_columns");
        pyodide.globals.delete("js_label_column");
        pyodide.globals.delete("js_max_iter");
        pyodide.globals.delete("js_result");
      }
    } catch {}

    const errorMessage = error.message || String(error);
    throw new Error(`Python Diversion Checker error:\n${errorMessage}`);
  }
}
