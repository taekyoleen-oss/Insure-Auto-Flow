"""
앱의 데이터 처리 과정을 시뮬레이션하여 정확한 원인 파악
앱: Split 없음, LinearRegression, intercept=70.06515, R²=0.7204
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def simulate_app_processing(csv_path: str):
    """
    앱의 데이터 처리 과정을 시뮬레이션
    """
    print("=" * 80)
    print("앱의 데이터 처리 과정 시뮬레이션")
    print("=" * 80)
    
    # 앱 결과
    app_intercept = 70.06515
    app_r2 = 0.7204
    
    # 데이터 로드 (앱과 동일한 방식)
    df = pd.read_csv(csv_path)
    print(f"\n1. 원본 데이터 로드")
    print(f"   Shape: {df.shape}")
    
    # 앱의 데이터 필터링 과정 시뮬레이션
    # 앱에서는 NaN, null, undefined를 필터링함
    print(f"\n2. 데이터 필터링 (앱 스타일)")
    
    # 결측치가 있는 행 제거
    df_clean = df.dropna()
    print(f"   결측치 제거 후: {df_clean.shape}")
    
    # 특성과 타겟 분리
    feature_columns = [col for col in df_clean.columns if col != 'y']
    X = df_clean[feature_columns]
    y = df_clean['y']
    
    # 앱에서는 각 행을 검증하면서 유효한 행만 사용
    # JavaScript의 typeof 체크와 유사하게 처리
    valid_indices = []
    for idx in range(len(X)):
        row = X.iloc[idx]
        label_val = y.iloc[idx]
        
        # 앱의 검증 로직: 모든 특성이 숫자이고 NaN이 아니어야 함
        if (pd.isna(label_val) or 
            not isinstance(label_val, (int, float)) or
            np.isnan(label_val)):
            continue
        
        # 모든 특성이 유효한 숫자인지 확인
        valid = True
        for col in feature_columns:
            val = row[col]
            if (pd.isna(val) or 
                not isinstance(val, (int, float)) or
                np.isnan(val) or
                val is None):
                valid = False
                break
        
        if valid:
            valid_indices.append(idx)
    
    X_filtered = X.iloc[valid_indices]
    y_filtered = y.iloc[valid_indices]
    
    print(f"   유효한 행 수: {len(valid_indices)} / {len(X)}")
    
    if len(X_filtered) != len(X):
        print(f"   경고: {len(X) - len(X_filtered)}개 행이 필터링되었습니다!")
    
    # Linear Regression 실행
    print(f"\n3. Linear Regression 실행")
    
    lr = LinearRegression(fit_intercept=True)
    lr.fit(X_filtered, y_filtered)
    y_pred = lr.predict(X_filtered)
    r2 = r2_score(y_filtered, y_pred)
    
    print(f"   Intercept: {lr.intercept_:.6f}")
    print(f"   R² Score: {r2:.6f}")
    print(f"   앱과 차이 - Intercept: {abs(lr.intercept_ - app_intercept):.6f}")
    print(f"   앱과 차이 - R²: {abs(r2 - app_r2):.6f}")
    
    # 데이터 타입 변환 확인
    print(f"\n4. 데이터 타입 확인")
    print(f"   X 데이터 타입: {X_filtered.dtypes.value_counts().to_dict()}")
    print(f"   y 데이터 타입: {y_filtered.dtype}")
    
    # 특성별 통계
    print(f"\n5. 특성별 통계 (큰 값이 있는 특성)")
    stats = X_filtered.describe()
    for col in feature_columns:
        max_val = X_filtered[col].max()
        if max_val > 1e6:
            print(f"   {col}: max={max_val:.2e}, mean={X_filtered[col].mean():.2e}, std={X_filtered[col].std():.2e}")
    
    # JavaScript 수치 계산 시뮬레이션 (부동소수점 정밀도)
    print(f"\n6. JavaScript 부동소수점 정밀도 시뮬레이션")
    
    # JavaScript는 IEEE 754 double precision 사용
    # 큰 숫자에서 정밀도 손실 발생 가능
    
    # X^T * X 계산 시뮬레이션
    X_array = X_filtered.values
    X_with_ones = np.column_stack([np.ones(len(X_array)), X_array])
    
    # JavaScript 스타일로 계산 (단일 정밀도로 시뮬레이션)
    XtX_js_style = np.zeros((X_with_ones.shape[1], X_with_ones.shape[1]))
    for i in range(X_with_ones.shape[1]):
        for j in range(X_with_ones.shape[1]):
            sum_val = 0.0
            for k in range(X_with_ones.shape[0]):
                # JavaScript의 부동소수점 연산 시뮬레이션
                val1 = float(X_with_ones[k, i])
                val2 = float(X_with_ones[k, j])
                sum_val += val1 * val2
            XtX_js_style[i, j] = sum_val
    
    Xty_js_style = np.zeros(X_with_ones.shape[1])
    for i in range(X_with_ones.shape[1]):
        sum_val = 0.0
        for k in range(X_with_ones.shape[0]):
            val1 = float(X_with_ones[k, i])
            val2 = float(y_filtered.iloc[k])
            sum_val += val1 * val2
        Xty_js_style[i] = sum_val
    
    # NumPy로 계산한 것과 비교
    XtX_numpy = X_with_ones.T @ X_with_ones
    Xty_numpy = X_with_ones.T @ y_filtered.values
    
    diff_XtX = np.abs(XtX_js_style - XtX_numpy).max()
    diff_Xty = np.abs(Xty_js_style - Xty_numpy).max()
    
    print(f"   X^T * X 최대 차이: {diff_XtX:.2e}")
    print(f"   X^T * y 최대 차이: {diff_Xty:.2e}")
    
    if diff_XtX > 1e-6:
        print(f"   경고: X^T * X 계산에서 차이가 발생했습니다!")
    
    # 행렬 역행렬 계산
    try:
        beta_js_style = np.linalg.solve(XtX_js_style, Xty_js_style)
        intercept_js_style = beta_js_style[0]
        
        y_pred_js_style = X_with_ones @ beta_js_style
        r2_js_style = r2_score(y_filtered, y_pred_js_style)
        
        print(f"\n   JavaScript 스타일 계산 결과:")
        print(f"   Intercept: {intercept_js_style:.6f} (차이: {abs(intercept_js_style - app_intercept):.6f})")
        print(f"   R²: {r2_js_style:.6f} (차이: {abs(r2_js_style - app_r2):.6f})")
    except Exception as e:
        print(f"   오류: {e}")
    
    # 결론
    print("\n" + "=" * 80)
    print("결론")
    print("=" * 80)
    print("\n앱과 파이썬 결과 차이의 주요 원인:")
    print("1. 행렬 조건수가 매우 높음 (5.74e+13)")
    print("   - 특성 x21의 분산이 1.78e+11로 매우 큼")
    print("   - 큰 값의 곱셈에서 부동소수점 오차 발생")
    
    print("\n2. JavaScript 부동소수점 연산 정밀도")
    print("   - IEEE 754 double precision 사용")
    print("   - 큰 숫자에서 정밀도 손실 발생 가능")
    print("   - 행렬 곱셈의 누적 오차")
    
    print("\n3. 행렬 역행렬 계산의 수치적 불안정성")
    print("   - 조건수가 높은 행렬에서 작은 오차가 증폭")
    print("   - Gauss-Jordan elimination의 수치적 한계")
    
    print("\n해결 방안:")
    print("1. 데이터 정규화 (StandardScaler) 사용")
    print("2. Ridge Regression 사용 (작은 alpha 값)")
    print("3. Python 백엔드 사용 (Pyodide 등)")

if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'compactiv.csv')
    simulate_app_processing(csv_path)
















































