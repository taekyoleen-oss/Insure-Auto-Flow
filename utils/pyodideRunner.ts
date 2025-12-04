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
function withTimeout<T>(promise: Promise<T>, timeoutMs: number, errorMessage: string): Promise<T> {
    return Promise.race([
        promise,
        new Promise<T>((_, reject) => 
            setTimeout(() => reject(new Error(errorMessage)), timeoutMs)
        )
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
                pyodide.loadPackage(['pandas', 'scikit-learn', 'numpy', 'scipy']),
                90000,
                '패키지 설치 타임아웃 (90초 초과)'
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
    if (typeof window !== 'undefined' && (window as any).loadPyodide) {
        return (window as any).loadPyodide({
            indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/'
        });
    }

    // Pyodide가 아직 로드되지 않았다면 에러
    throw new Error('Pyodide is not loaded. Please ensure pyodide.js is included in index.html');
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
export async function callPythonFunction(functionName: string, ...args: any[]): Promise<any> {
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
    if (pythonObj && typeof pythonObj.toJs === 'function') {
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
): Promise<{ trainIndices: number[], testIndices: number[] }> {
    let py: any = null;
    try {
        // Pyodide 로드 (타임아웃: 30초)
        try {
            py = await withTimeout(
                loadPyodide(30000),
                30000,
                'Pyodide 로딩 타임아웃 (30초 초과)'
            );
        } catch (loadError: any) {
            const loadErrorMessage = loadError.message || String(loadError);
            if (loadErrorMessage.includes('Failed to fetch') || loadErrorMessage.includes('NetworkError')) {
                throw new Error(`Pyodide CDN 로드 실패: 네트워크 연결을 확인하거나 인터넷 연결이 필요합니다. ${loadErrorMessage}`);
            }
            throw new Error(`Pyodide 로드 실패: ${loadErrorMessage}`);
        }
        
        // 데이터를 Python에 전달
        py.globals.set('js_data', data);
        
        // stratify_column을 Python 코드에 전달하기 위한 처리
        // None이면 문자열 'None'으로, 아니면 문자열로 감싸서 전달
        const stratifyColStr = stratifyColumn ? `'${stratifyColumn}'` : 'None';
        
        // JavaScript boolean을 Python boolean으로 변환
        const shufflePython = shuffle ? 'True' : 'False';
        const stratifyPython = stratify ? 'True' : 'False';
        
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
        const resultPyObj = py.globals.get('js_result');
        
        // 결과 객체 검증
        if (!resultPyObj) {
            throw new Error(`Python split_data error: Python code returned None or undefined.`);
        }
        
        // Python 딕셔너리를 JavaScript 객체로 변환
        const result = fromPython(resultPyObj);
        
        // 에러가 발생한 경우 처리
        if (result && result.__error__) {
            throw new Error(`Python split_data error:\n${result.error_traceback || result.error_message}`);
        }
        
        // 결과 검증
        if (!result.train_indices || !result.test_indices) {
            throw new Error(`Python split_data error: Missing train_indices or test_indices in result.`);
        }
        
        // 정리
        py.globals.delete('js_data');
        py.globals.delete('js_result');
        // js_tuning_options는 Linear Regression에서만 사용되므로 존재할 때만 삭제
        if (py.globals.has('js_tuning_options')) {
            py.globals.delete('js_tuning_options');
        }
        
        return {
            trainIndices: result.train_indices,
            testIndices: result.test_indices
        };
    } catch (error: any) {
        // 정리
        try {
            const py = pyodide;
            if (py) {
                py.globals.delete('js_data');
                py.globals.delete('js_result');
                // js_tuning_options는 Linear Regression에서만 사용되므로 존재할 때만 삭제
                if (py.globals.has('js_tuning_options')) {
                    py.globals.delete('js_tuning_options');
                }
            }
        } catch {}
        
        const errorMessage = error.message || String(error);
        throw new Error(`Python split_data error: ${errorMessage}`);
    }
}

export interface LinearRegressionTuningOptions {
    enabled: boolean;
    strategy?: 'GridSearch';
    alphaCandidates?: number[];
    l1RatioCandidates?: number[];
    cvFolds?: number;
    scoringMetric?: string;
}

interface LinearRegressionTuningPayload {
    enabled: boolean;
    strategy?: 'grid';
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
    modelType: string = 'LinearRegression',
    fitIntercept: boolean = true,
    alpha: number = 1.0,
    l1Ratio: number = 0.5,
    featureColumns?: string[],
    timeoutMs: number = 60000,
    tuningOptions?: LinearRegressionTuningOptions
): Promise<{ coefficients: number[], intercept: number, metrics: Record<string, number>, tuning?: LinearRegressionTuningPayload }> {
    try {
        // Pyodide 로드 (타임아웃: 30초)
        const py = await withTimeout(
            loadPyodide(30000),
            30000,
            'Pyodide 로딩 타임아웃 (30초 초과)'
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
            row['y'] = y[i];
            dataRows.push(row);
        }
        
        py.globals.set('js_data', dataRows);
        py.globals.set('js_feature_columns', featureColumns || X[0].map((_, idx) => `x${idx}`));
        py.globals.set('js_label_column', 'y');
        py.globals.set('js_tuning_options', tuningOptions ? tuningOptions : null);
        
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
    p_fit_intercept = ${fitIntercept ? 'True' : 'False'}
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
            'Python LinearRegression 실행 타임아웃 (60초 초과)'
        );
        
        // 전역 변수에서 결과 가져오기
        const resultPyObj = py.globals.get('js_result');
        
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
                throw new Error(`Python LinearRegression error: Python code returned None or undefined. Debug info: ${JSON.stringify(debug)}`);
            } catch (debugError) {
                throw new Error(`Python LinearRegression error: Python code returned None or undefined. Check Python code execution.`);
            }
        }
        
        // Python 딕셔너리를 JavaScript 객체로 변환
        const result = fromPython(resultPyObj);
        
        // 결과 검증
        if (!result || typeof result !== 'object') {
            throw new Error(`Python LinearRegression error: Invalid result returned from Python code. Got: ${typeof result}, value: ${JSON.stringify(result)}`);
        }
        
        // 에러가 발생한 경우 처리
        if (result.__error__) {
            throw new Error(`Python LinearRegression error:\n${result.error_traceback || result.error_message || 'Unknown error'}`);
        }
        
        // 필수 속성 검증
        if (!result.coefficients || !Array.isArray(result.coefficients)) {
            throw new Error(`Python LinearRegression error: Missing or invalid 'coefficients' in result. Got: ${JSON.stringify(result)}`);
        }
        if (typeof result.intercept !== 'number' && result.intercept !== null && result.intercept !== undefined) {
            throw new Error(`Python LinearRegression error: Missing or invalid 'intercept' in result. Got: ${typeof result.intercept}`);
        }
        if (!result.metrics || typeof result.metrics !== 'object') {
            throw new Error(`Python LinearRegression error: Missing or invalid 'metrics' in result. Got: ${typeof result.metrics}`);
        }
        
        // 정리
        py.globals.delete('js_data');
        py.globals.delete('js_feature_columns');
        py.globals.delete('js_label_column');
        py.globals.delete('js_result');
        
        return {
            coefficients: result.coefficients,
            intercept: result.intercept ?? 0.0,
            metrics: result.metrics,
            tuning: result.tuning
        };
    } catch (error: any) {
        // 정리
        try {
            const py = pyodide;
            if (py) {
                py.globals.delete('js_data');
                py.globals.delete('js_feature_columns');
                py.globals.delete('js_label_column');
                py.globals.delete('js_result');
            }
        } catch {}
        
        // 에러 메시지 추출
        let errorMessage = error.message || String(error);
        
        // Pyodide PythonError의 경우 더 자세한 정보 추출 시도
        if (error.name === 'PythonError' || error.toString().includes('Traceback')) {
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
        const fullError = errorMessage.includes('Traceback') 
            ? errorMessage 
            : `${error.toString()}\n${errorMessage}`;
        throw new Error(`Python LinearRegression error:\n${fullError}`);
    }
}

/**
 * Statistics를 Python으로 실행합니다
 * 타임아웃: 60초
 */
export async function calculateStatisticsPython(
    data: any[],
    columns: Array<{ name: string, type: string }>,
    timeoutMs: number = 60000
): Promise<{ stats: Record<string, any>, correlation: Record<string, Record<string, number>> }> {
    try {
        // Pyodide 로드 (타임아웃: 30초)
        const py = await withTimeout(
            loadPyodide(30000),
            30000,
            'Pyodide 로딩 타임아웃 (30초 초과)'
        );
        
        // 데이터를 Python에 전달
        py.globals.set('js_data', data);
        py.globals.set('js_columns', columns);
        
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
            'Python Statistics 실행 타임아웃 (60초 초과)'
        );
        
        // Python 딕셔너리를 JavaScript 객체로 변환
        const result = fromPython(resultPyObj);
        
        // 정리
        py.globals.delete('js_data');
        py.globals.delete('js_columns');
        
        return {
            stats: result.stats,
            correlation: result.correlation
        };
    } catch (error: any) {
        // 정리
        try {
            const py = pyodide;
            if (py) {
                py.globals.delete('js_data');
                py.globals.delete('js_columns');
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
    modelPurpose: 'classification' | 'regression',
    timeoutMs: number = 60000
): Promise<{ rows: any[], columns: Array<{ name: string, type: string }> }> {
    try {
        // Pyodide 로드 (타임아웃: 30초)
        const py = await withTimeout(
            loadPyodide(30000),
            30000,
            'Pyodide 로딩 타임아웃 (30초 초과)'
        );
        
        // 데이터를 Python에 전달
        py.globals.set('js_data', data);
        py.globals.set('js_feature_columns', featureColumns);
        py.globals.set('js_coefficients', coefficients);
        py.globals.set('js_intercept', intercept);
        py.globals.set('js_label_column', labelColumn);
        py.globals.set('js_model_purpose', modelPurpose);
        
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
            'Python ScoreModel 실행 타임아웃 (60초 초과)'
        );
        
        // Python 딕셔너리를 JavaScript 객체로 변환
        const result = fromPython(resultPyObj);
        
        // 정리
        py.globals.delete('js_data');
        py.globals.delete('js_feature_columns');
        py.globals.delete('js_coefficients');
        py.globals.delete('js_intercept');
        py.globals.delete('js_label_column');
        py.globals.delete('js_model_purpose');
        
        return {
            rows: result.rows,
            columns: result.columns
        };
    } catch (error: any) {
        // 정리
        try {
            const py = pyodide;
            if (py) {
                py.globals.delete('js_data');
                py.globals.delete('js_feature_columns');
                py.globals.delete('js_coefficients');
                py.globals.delete('js_intercept');
                py.globals.delete('js_label_column');
                py.globals.delete('js_model_purpose');
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
    modelType: 'classification' | 'regression',
    timeoutMs: number = 60000
): Promise<Record<string, number | string>> {
    try {
        // Pyodide 로드 (타임아웃: 30초)
        const py = await withTimeout(
            loadPyodide(30000),
            30000,
            'Pyodide 로딩 타임아웃 (30초 초과)'
        );
        
        // 데이터를 Python에 전달
        py.globals.set('js_data', data);
        py.globals.set('js_label_column', labelColumn);
        py.globals.set('js_prediction_column', predictionColumn);
        py.globals.set('js_model_type', modelType);
        
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

# 실제값과 예측값 추출
y_true = df[label_column].values
y_pred = df[prediction_column].values

metrics = {}

if model_type == 'classification':
    # 분류 메트릭
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
    
    metrics['Accuracy'] = accuracy
    metrics['Precision'] = precision
    metrics['Recall'] = recall
    metrics['F1-Score'] = f1
    metrics['Confusion Matrix'] = f"TP:{tp}, FP:{fp}, TN:{tn}, FN:{fn}"
else:
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
            'Python EvaluateModel 실행 타임아웃 (60초 초과)'
        );
        
        // Python 딕셔너리를 JavaScript 객체로 변환
        const metrics = fromPython(resultPyObj);
        
        // 정리
        py.globals.delete('js_data');
        py.globals.delete('js_label_column');
        py.globals.delete('js_prediction_column');
        py.globals.delete('js_model_type');
        
        return metrics;
    } catch (error: any) {
        // 정리
        try {
            const py = pyodide;
            if (py) {
                py.globals.delete('js_data');
                py.globals.delete('js_label_column');
                py.globals.delete('js_prediction_column');
                py.globals.delete('js_model_type');
            }
        } catch {}
        
        const errorMessage = error.message || String(error);
        throw new Error(`Python EvaluateModel error: ${errorMessage}`);
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
            'Pyodide 로딩 타임아웃 (30초 초과)'
        );
        
        py.globals.set('js_data', data);
        py.globals.set('js_method', method);
        py.globals.set('js_strategy', strategy);
        py.globals.set('js_columns', columns);
        py.globals.set('js_n_neighbors', n_neighbors);
        
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
            'Python HandleMissingValues 실행 타임아웃 (60초 초과)'
        );
        
        const result = fromPython(resultPyObj);
        
        py.globals.delete('js_data');
        py.globals.delete('js_method');
        py.globals.delete('js_strategy');
        py.globals.delete('js_columns');
        py.globals.delete('js_n_neighbors');
        
        return result;
    } catch (error: any) {
        try {
            const py = pyodide;
            if (py) {
                py.globals.delete('js_data');
                py.globals.delete('js_method');
                py.globals.delete('js_strategy');
                py.globals.delete('js_columns');
                py.globals.delete('js_n_neighbors');
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
): Promise<{ stats: Record<string, { min?: number; max?: number; mean?: number; stdDev?: number; median?: number; iqr?: number }> }> {
    try {
        const py = await withTimeout(
            loadPyodide(30000),
            30000,
            'Pyodide 로딩 타임아웃 (30초 초과)'
        );
        
        py.globals.set('js_data', data);
        py.globals.set('js_method', method);
        py.globals.set('js_columns', columns);
        
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
            'Python NormalizeData 실행 타임아웃 (60초 초과)'
        );
        
        const result = fromPython(resultPyObj);
        
        py.globals.delete('js_data');
        py.globals.delete('js_method');
        py.globals.delete('js_columns');
        
        return result;
    } catch (error: any) {
        try {
            const py = pyodide;
            if (py) {
                py.globals.delete('js_data');
                py.globals.delete('js_method');
                py.globals.delete('js_columns');
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
): Promise<{ rows: any[], columns: Array<{ name: string, type: string }> }> {
    try {
        const py = await withTimeout(
            loadPyodide(30000),
            30000,
            'Pyodide 로딩 타임아웃 (30초 초과)'
        );
        
        py.globals.set('js_data', data);
        py.globals.set('js_transformations', transformations);
        
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
            'Python TransitionData 실행 타임아웃 (60초 초과)'
        );
        
        const result = fromPython(resultPyObj);
        
        py.globals.delete('js_data');
        py.globals.delete('js_transformations');
        
        return result;
    } catch (error: any) {
        try {
            const py = pyodide;
            if (py) {
                py.globals.delete('js_data');
                py.globals.delete('js_transformations');
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
            'Pyodide 로딩 타임아웃 (30초 초과)'
        );
        
        py.globals.set('js_data', data);
        py.globals.set('js_method', method);
        py.globals.set('js_columns', columns);
        py.globals.set('js_ordinal_mapping', ordinal_mapping);
        py.globals.set('js_drop', drop);
        py.globals.set('js_handle_unknown', handle_unknown);
        
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
            'Python EncodeCategorical 실행 타임아웃 (60초 초과)'
        );
        
        const result = fromPython(resultPyObj);
        
        py.globals.delete('js_data');
        py.globals.delete('js_method');
        py.globals.delete('js_columns');
        py.globals.delete('js_ordinal_mapping');
        py.globals.delete('js_drop');
        py.globals.delete('js_handle_unknown');
        
        return result;
    } catch (error: any) {
        try {
            const py = pyodide;
            if (py) {
                py.globals.delete('js_data');
                py.globals.delete('js_method');
                py.globals.delete('js_columns');
                py.globals.delete('js_ordinal_mapping');
                py.globals.delete('js_drop');
                py.globals.delete('js_handle_unknown');
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
): Promise<{ rows: any[], columns: Array<{ name: string, type: string }> }> {
    try {
        const py = await withTimeout(
            loadPyodide(30000),
            30000,
            'Pyodide 로딩 타임아웃 (30초 초과)'
        );
        
        py.globals.set('js_data', data);
        py.globals.set('js_method', method);
        py.globals.set('js_target_column', target_column);
        
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
            'Python ResampleData 실행 타임아웃 (60초 초과)'
        );
        
        const result = fromPython(resultPyObj);
        
        py.globals.delete('js_data');
        py.globals.delete('js_method');
        py.globals.delete('js_target_column');
        
        return result;
    } catch (error: any) {
        try {
            const py = pyodide;
            if (py) {
                py.globals.delete('js_data');
                py.globals.delete('js_method');
                py.globals.delete('js_target_column');
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
): Promise<{ rows: any[], columns: Array<{ name: string, type: string }> }> {
    try {
        const py = await withTimeout(
            loadPyodide(30000),
            30000,
            'Pyodide 로딩 타임아웃 (30초 초과)'
        );
        
        py.globals.set('js_data', data);
        py.globals.set('js_handler', handler);
        py.globals.set('js_exclude_columns', exclude_columns);
        
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
            'Python TransformData 실행 타임아웃 (60초 초과)'
        );
        
        const result = fromPython(resultPyObj);
        
        py.globals.delete('js_data');
        py.globals.delete('js_handler');
        py.globals.delete('js_exclude_columns');
        
        return result;
    } catch (error: any) {
        try {
            const py = pyodide;
            if (py) {
                py.globals.delete('js_data');
                py.globals.delete('js_handler');
                py.globals.delete('js_exclude_columns');
            }
        } catch {}
        
        const errorMessage = error.message || String(error);
        throw new Error(`Python TransformData error: ${errorMessage}`);
    }
}

