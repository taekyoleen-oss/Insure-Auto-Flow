# Pyodide 우선 개발 가이드

## 개요

이 프로젝트는 **모든 분석 알고리즘을 Pyodide를 통해 Python으로 실행**하여 Python과의 정합성을 보장합니다. JavaScript는 결과값을 변경하지 않는 범위에서만 사용됩니다.

## 개발 원칙

### 1. Pyodide 우선 원칙
- **모든 수식, 통계 계산, 알고리즘은 Pyodide를 통해 Python으로 실행**
- JavaScript는 UI 렌더링, 데이터 구조 변환, 결과 표시에만 사용
- Python 결과값을 변경하거나 조정하는 JavaScript 코드는 금지

### 2. 폴백 전략
```
Pyodide (우선) → Node.js 백엔드 (필요시) → 에러 발생
```
- Pyodide 실패 시에만 Node.js 백엔드 사용
- JavaScript 폴백은 사용하지 않음 (Python 정합성 보장)

## 현재 Pyodide로 구현된 모듈

### 데이터 처리 모듈
1. **SplitData** - `sklearn.train_test_split` 사용
2. **Statistics** - `pandas`/`numpy` 사용
3. **HandleMissingValues** - `pandas`/`sklearn.impute` 사용
4. **NormalizeData** - `sklearn.preprocessing` 사용
5. **TransitionData** - `numpy` 수학 함수 사용
6. **EncodeCategorical** - `sklearn.preprocessing` 사용
7. **ResampleData** - `imblearn` 사용
8. **TransformData** - `pandas`/`sklearn.preprocessing` 사용

### 머신러닝 모듈
1. **TrainModel (LinearRegression)** - `sklearn.linear_model` 사용
2. **ScoreModel** - `numpy` 사용
3. **EvaluateModel** - `sklearn.metrics` 사용

## 새 모듈 개발 가이드

### 1. Pyodide 유틸리티 함수 추가

`utils/pyodideRunner.ts`에 새로운 함수를 추가합니다:

```typescript
/**
 * 새 모듈을 Python으로 실행합니다
 * 타임아웃: 60초
 */
export async function newModulePython(
    data: any[],
    param1: string,
    param2: number,
    timeoutMs: number = 60000
): Promise<{ result: any }> {
    try {
        // Pyodide 로드 (타임아웃: 30초)
        const py = await withTimeout(
            loadPyodide(30000),
            30000,
            'Pyodide 로딩 타임아웃 (30초 초과)'
        );
        
        // 데이터를 Python에 전달
        py.globals.set('js_data', data);
        py.globals.set('js_param1', param1);
        py.globals.set('js_param2', param2);
        
        // Python 코드 실행
        const code = `
import pandas as pd
import numpy as np
from sklearn.xxx import XXX

df = pd.DataFrame(js_data.to_py())
param1 = str(js_param1)
param2 = float(js_param2)

# Python 분석 코드 작성
# ...

result = {
    'result': ...
}

result
`;
        
        const resultPyObj = await withTimeout(
            Promise.resolve(py.runPython(code)),
            timeoutMs,
            'Python NewModule 실행 타임아웃 (60초 초과)'
        );
        
        // Python 딕셔너리를 JavaScript 객체로 변환
        const result = fromPython(resultPyObj);
        
        // 정리
        py.globals.delete('js_data');
        py.globals.delete('js_param1');
        py.globals.delete('js_param2');
        
        return result;
    } catch (error: any) {
        // 정리
        try {
            const py = pyodide;
            if (py) {
                py.globals.delete('js_data');
                py.globals.delete('js_param1');
                py.globals.delete('js_param2');
            }
        } catch {}
        
        const errorMessage = error.message || String(error);
        throw new Error(`Python NewModule error: ${errorMessage}`);
    }
}
```

### 2. App.tsx에서 모듈 구현

```typescript
} else if (module.type === ModuleType.NewModule) {
    const inputData = getSingleInputData(module.id) as DataPreview;
    if (!inputData) throw new Error("Input data not available.");

    const { param1, param2 } = module.parameters;
    
    // Pyodide를 사용하여 Python으로 분석 수행
    try {
        addLog('INFO', 'Pyodide를 사용하여 Python으로 새 모듈 실행 중...');
        
        const pyodideModule = await import('./utils/pyodideRunner');
        const { newModulePython } = pyodideModule;
        
        const result = await newModulePython(
            inputData.rows || [],
            param1,
            param2,
            60000 // 타임아웃: 60초
        );
        
        newOutputData = {
            type: 'NewModuleOutput',
            // result 사용
        };
        
        addLog('SUCCESS', 'Python으로 새 모듈 실행 완료');
    } catch (error: any) {
        const errorMessage = error.message || String(error);
        addLog('ERROR', `Python NewModule 실패: ${errorMessage}`);
        throw new Error(`새 모듈 실행 실패: ${errorMessage}`);
    }
}
```

### 3. 폴백 처리 (선택사항)

Pyodide 실패 시 Node.js 백엔드로 전환하려면:

```typescript
} catch (error: any) {
    const errorMessage = error.message || String(error);
    
    // 타임아웃이거나 Pyodide 실패 시 Node.js 백엔드로 전환
    if (errorMessage.includes('타임아웃') || errorMessage.includes('timeout')) {
        addLog('WARN', `Pyodide 타임아웃, Node.js 백엔드로 전환`);
        
        try {
            // Node.js 백엔드 API 호출
            const response = await fetch('/api/new-module', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ data: inputData.rows, param1, param2 })
            });
            
            if (response.ok) {
                const result = await response.json();
                newOutputData = { type: 'NewModuleOutput', ...result };
                addLog('SUCCESS', 'Node.js 백엔드로 새 모듈 실행 완료');
            } else {
                throw new Error(`Node.js 백엔드 응답 오류: ${response.status}`);
            }
        } catch (nodeError: any) {
            throw new Error(`새 모듈 실행 실패: Pyodide와 Node.js 백엔드 모두 실패했습니다.`);
        }
    } else {
        throw new Error(`새 모듈 실행 실패: ${errorMessage}`);
    }
}
```

## 중요 사항

### ✅ 해야 할 것
- 모든 수식/통계 계산은 Python으로 수행
- Pyodide를 우선 사용하고, 필요시 Node.js 백엔드 사용
- Python 결과값을 변경하지 않는 JavaScript만 사용
- 에러 처리 및 타임아웃 설정

### ❌ 하지 말아야 할 것
- JavaScript로 수식 계산 (mean, median, std, log, sqrt 등)
- Python 결과값을 조정하거나 변경하는 JavaScript 코드
- JavaScript 폴백 사용 (Python 정합성 깨짐)

## Python 코드 작성 가이드

### 데이터 전달
```python
# JavaScript에서 전달된 데이터
df = pd.DataFrame(js_data.to_py())
param = str(js_param)
```

### 결과 반환
```python
# 딕셔너리로 반환 (자동으로 JavaScript 객체로 변환됨)
result = {
    'key1': value1,
    'key2': value2
}
result  # 마지막 줄에 반환할 객체
```

### 타입 변환
```python
# 숫자는 float()로 변환
value = float(some_value)

# 리스트는 .tolist() 사용
arr = numpy_array.tolist()

# 문자열은 str() 사용
text = str(some_text)
```

## 패키지 추가

새로운 Python 패키지가 필요하면 `loadPyodide()` 함수에서 패키지 목록에 추가:

```typescript
await pyodide.loadPackage(['pandas', 'scikit-learn', 'numpy', 'scipy', 'new-package']);
```

## 참고

- `data_analysis_modules.py`: Python 구현 참고용
- `codeSnippets.ts`: Python 코드 템플릿
- `utils/pyodideRunner.ts`: Pyodide 유틸리티 함수들







































