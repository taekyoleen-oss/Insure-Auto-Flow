"""
앱의 JavaScript LinearRegression 구현을 Python으로 재현하여 비교
앱 결과: intercept=70.06515, R²=0.7204
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def gauss_jordan_invert(matrix):
    """
    JavaScript의 invert 함수와 동일한 Gauss-Jordan elimination 구현
    """
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be square to be inverted.")
    
    # Identity matrix 생성
    identity = np.eye(n).tolist()
    
    # Augmented matrix 생성
    augmented = [row + identity[i] for i, row in enumerate(matrix)]
    augmented = np.array(augmented, dtype=float)
    
    # Gauss-Jordan elimination
    for i in range(n):
        # Pivot 찾기
        pivot = i
        for j in range(i + 1, n):
            if abs(augmented[j, i]) > abs(augmented[pivot, i]):
                pivot = j
        
        # 행 교환
        augmented[[i, pivot]] = augmented[[pivot, i]]
        
        # Pivot이 0에 가까우면 singular
        div = augmented[i, i]
        if abs(div) < 1e-10:
            raise ValueError("Matrix is singular and cannot be inverted.")
        
        # Pivot 행을 pivot 값으로 나누기
        augmented[i] /= div
        
        # 다른 행에서 pivot 열 제거
        for j in range(n):
            if i != j:
                mult = augmented[j, i]
                augmented[j] -= mult * augmented[i]
    
    # 역행렬 부분만 반환
    return augmented[:, n:].tolist()

def matrix_multiply(a, b):
    """
    JavaScript의 multiply 함수와 동일한 행렬 곱셈 구현
    """
    a = np.array(a)
    b = np.array(b)
    result = a @ b
    return result.tolist()

def fit_linear_regression_js_style(X, y, fit_intercept=True):
    """
    앱의 JavaScript fitLinearRegression 함수를 Python으로 재현
    """
    n = len(X)
    if n == 0:
        raise ValueError("No data provided for fitting.")
    
    if len(X[0]) == 0:
        raise ValueError("Feature matrix is empty or invalid.")
    
    m = len(X[0])
    if len(y) != n:
        raise ValueError(f"Mismatch between feature rows ({n}) and target length ({len(y)}).")
    
    # Intercept를 위한 열 추가
    if fit_intercept:
        X_with_intercept = [[1.0] + row for row in X]
    else:
        X_with_intercept = [row[:] for row in X]
    
    num_features = len(X_with_intercept[0])
    
    # Compute X^T * X
    XtX = []
    for i in range(num_features):
        XtX.append([])
        for j in range(num_features):
            sum_val = 0.0
            for k in range(n):
                sum_val += X_with_intercept[k][i] * X_with_intercept[k][j]
            XtX[i].append(sum_val)
    
    # Compute X^T * y
    Xty = []
    for i in range(num_features):
        sum_val = 0.0
        for k in range(n):
            sum_val += X_with_intercept[k][i] * y[k]
        Xty.append(sum_val)
    
    # Compute (X^T * X)^(-1)
    try:
        XtX_inv = gauss_jordan_invert(XtX)
    except Exception as e:
        raise ValueError(f"Matrix is singular. Cannot fit linear regression. {e}")
    
    # Compute β = (X^T * X)^(-1) * X^T * y
    Xty_column = [[val] for val in Xty]
    beta_matrix = matrix_multiply(XtX_inv, Xty_column)
    
    beta = [row[0] for row in beta_matrix]
    
    if fit_intercept:
        if len(beta) < 1:
            raise ValueError("Beta array is too short for intercept model.")
        return {
            'intercept': beta[0],
            'coefficients': beta[1:]
        }
    else:
        return {
            'intercept': 0.0,
            'coefficients': beta
        }

def compare_implementations(csv_path: str):
    """
    JavaScript 구현과 Python sklearn 구현 비교
    """
    print("=" * 80)
    print("JavaScript 구현 vs Python sklearn 구현 비교")
    print("=" * 80)
    
    # 데이터 로드
    df = pd.read_csv(csv_path)
    df = df.dropna()
    
    feature_columns = [col for col in df.columns if col != 'y']
    X = df[feature_columns].values.tolist()
    y = df['y'].tolist()
    
    print(f"\n데이터 Shape: {len(X)} x {len(X[0])}")
    print(f"특성 개수: {len(feature_columns)}")
    
    # 앱 결과
    app_intercept = 70.06515
    app_r2 = 0.7204
    
    print("\n" + "=" * 80)
    print("1. Python sklearn LinearRegression")
    print("=" * 80)
    
    X_array = np.array(X)
    y_array = np.array(y)
    
    lr_sklearn = LinearRegression(fit_intercept=True)
    lr_sklearn.fit(X_array, y_array)
    y_pred_sklearn = lr_sklearn.predict(X_array)
    r2_sklearn = r2_score(y_array, y_pred_sklearn)
    
    print(f"Intercept: {lr_sklearn.intercept_:.6f}")
    print(f"R² Score: {r2_sklearn:.6f}")
    print(f"앱과 차이 - Intercept: {abs(lr_sklearn.intercept_ - app_intercept):.6f}")
    print(f"앱과 차이 - R²: {abs(r2_sklearn - app_r2):.6f}")
    
    print("\n" + "=" * 80)
    print("2. JavaScript 스타일 구현 (Python으로 재현)")
    print("=" * 80)
    
    try:
        result_js_style = fit_linear_regression_js_style(X, y, fit_intercept=True)
        
        # 예측값 계산
        y_pred_js = []
        for i in range(len(X)):
            pred = result_js_style['intercept']
            for j in range(len(X[i])):
                pred += result_js_style['coefficients'][j] * X[i][j]
            y_pred_js.append(pred)
        
        r2_js = r2_score(y_array, np.array(y_pred_js))
        
        print(f"Intercept: {result_js_style['intercept']:.6f}")
        print(f"R² Score: {r2_js:.6f}")
        print(f"앱과 차이 - Intercept: {abs(result_js_style['intercept'] - app_intercept):.6f}")
        print(f"앱과 차이 - R²: {abs(r2_js - app_r2):.6f}")
        
        print(f"\nsklearn과 차이 - Intercept: {abs(result_js_style['intercept'] - lr_sklearn.intercept_):.6f}")
        print(f"sklearn과 차이 - R²: {abs(r2_js - r2_sklearn):.6f}")
        
        # 계수 비교 (처음 5개)
        print(f"\n계수 비교 (처음 5개):")
        for i in range(min(5, len(feature_columns))):
            diff = abs(result_js_style['coefficients'][i] - lr_sklearn.coef_[i])
            print(f"  {feature_columns[i]}:")
            print(f"    JS 스타일: {result_js_style['coefficients'][i]:.6f}")
            print(f"    sklearn:   {lr_sklearn.coef_[i]:.6f}")
            print(f"    차이:      {diff:.6f}")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    # NumPy로 직접 계산 (참고용)
    print("\n" + "=" * 80)
    print("3. NumPy로 직접 계산 (참고용)")
    print("=" * 80)
    
    X_with_ones = np.column_stack([np.ones(len(X_array)), X_array])
    XtX_np = X_with_ones.T @ X_with_ones
    Xty_np = X_with_ones.T @ y_array
    beta_np = np.linalg.solve(XtX_np, Xty_np)
    
    intercept_np = beta_np[0]
    coefficients_np = beta_np[1:]
    
    y_pred_np = X_with_ones @ beta_np
    r2_np = r2_score(y_array, y_pred_np)
    
    print(f"Intercept: {intercept_np:.6f}")
    print(f"R² Score: {r2_np:.6f}")
    print(f"앱과 차이 - Intercept: {abs(intercept_np - app_intercept):.6f}")
    print(f"앱과 차이 - R²: {abs(r2_np - app_r2):.6f}")
    
    # 행렬 조건수 확인
    print("\n" + "=" * 80)
    print("4. 행렬 조건수 분석")
    print("=" * 80)
    
    X_with_ones = np.column_stack([np.ones(len(X_array)), X_array])
    XtX = X_with_ones.T @ X_with_ones
    cond_num = np.linalg.cond(XtX)
    
    print(f"X^T * X 조건수: {cond_num:.2e}")
    print(f"조건수가 크면 (>{1e12:.2e}) 수치적으로 불안정할 수 있습니다.")
    
    if cond_num > 1e12:
        print("경고: 행렬이 수치적으로 불안정합니다. 정규화를 고려하세요.")
    
    # 결론
    print("\n" + "=" * 80)
    print("결론")
    print("=" * 80)
    print("\n앱 결과:")
    print(f"  Intercept: {app_intercept:.6f}")
    print(f"  R² Score: {app_r2:.6f}")
    print("\n가능한 원인:")
    print("1. JavaScript의 부동소수점 연산 정밀도 차이")
    print("2. 행렬 역행렬 계산의 수치적 불안정성")
    print("3. 데이터 전처리 차이 (정규화, 스케일링 등)")
    print("4. 특성 선택 차이")

if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'compactiv.csv')
    compare_implementations(csv_path)









































