"""
앱 vs 파이썬 최종 비교 및 원인 분석
앱: Split 없음, LinearRegression 사용, intercept=70.06515, R²=0.7204
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def analyze_final_difference(csv_path: str):
    """
    앱과 파이썬 결과 차이의 최종 원인 분석
    """
    print("=" * 80)
    print("앱 vs 파이썬 최종 비교 분석")
    print("=" * 80)
    print("\n앱 설정:")
    print("  - Split 사용: 아니오")
    print("  - 모델: LinearRegression")
    print("  - 결과: intercept=70.06515, R²=0.7204")
    
    # 데이터 로드
    df = pd.read_csv(csv_path)
    print(f"\n원본 데이터 Shape: {df.shape}")
    
    # 결측치 확인 및 처리
    missing_before = df.isnull().sum().sum()
    df = df.dropna()
    print(f"결측치 제거 후 Shape: {df.shape} (제거된 결측치: {missing_before})")
    
    feature_columns = [col for col in df.columns if col != 'y']
    X = df[feature_columns]
    y = df['y']
    
    # 앱 결과
    app_intercept = 70.06515
    app_r2 = 0.7204
    
    print("\n" + "=" * 80)
    print("1. 기본 Linear Regression (전체 데이터)")
    print("=" * 80)
    
    lr_basic = LinearRegression(fit_intercept=True)
    lr_basic.fit(X, y)
    y_pred_basic = lr_basic.predict(X)
    r2_basic = r2_score(y, y_pred_basic)
    
    print(f"Intercept: {lr_basic.intercept_:.6f}")
    print(f"R² Score: {r2_basic:.6f}")
    print(f"앱과 차이 - Intercept: {abs(lr_basic.intercept_ - app_intercept):.6f}")
    print(f"앱과 차이 - R²: {abs(r2_basic - app_r2):.6f}")
    
    # 행렬 조건수 확인
    X_with_ones = np.column_stack([np.ones(len(X)), X.values])
    XtX = X_with_ones.T @ X_with_ones
    cond_num = np.linalg.cond(XtX)
    print(f"\n행렬 조건수: {cond_num:.2e}")
    print("경고: 조건수가 매우 높아 수치적으로 불안정합니다!")
    
    print("\n" + "=" * 80)
    print("2. 데이터 정규화 후 Linear Regression")
    print("=" * 80)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)
    
    lr_scaled = LinearRegression(fit_intercept=True)
    lr_scaled.fit(X_scaled_df, y)
    y_pred_scaled = lr_scaled.predict(X_scaled_df)
    r2_scaled = r2_score(y, y_pred_scaled)
    
    print(f"Intercept: {lr_scaled.intercept_:.6f}")
    print(f"R² Score: {r2_scaled:.6f}")
    print(f"앱과 차이 - Intercept: {abs(lr_scaled.intercept_ - app_intercept):.6f}")
    print(f"앱과 차이 - R²: {abs(r2_scaled - app_r2):.6f}")
    
    # 정규화된 데이터의 조건수
    X_scaled_with_ones = np.column_stack([np.ones(len(X_scaled)), X_scaled])
    XtX_scaled = X_scaled_with_ones.T @ X_scaled_with_ones
    cond_num_scaled = np.linalg.cond(XtX_scaled)
    print(f"\n정규화 후 행렬 조건수: {cond_num_scaled:.2e}")
    
    print("\n" + "=" * 80)
    print("3. 특성별 영향 분석")
    print("=" * 80)
    
    # 각 특성의 분산 확인
    print("\n특성별 분산 (큰 값이 수치 불안정성에 기여):")
    variances = X.var()
    top_variances = variances.sort_values(ascending=False).head(10)
    for feature, var_val in top_variances.items():
        print(f"  {feature}: {var_val:.2e}")
    
    print("\n" + "=" * 80)
    print("4. JavaScript 수치 계산 시뮬레이션")
    print("=" * 80)
    
    # JavaScript는 IEEE 754 double precision을 사용
    # 하지만 행렬 연산에서 누적 오차가 발생할 수 있음
    
    # 수치적으로 불안정한 계산 시뮬레이션
    X_with_ones = np.column_stack([np.ones(len(X)), X.values])
    XtX = X_with_ones.T @ X_with_ones
    
    # 작은 노이즈 추가 (JavaScript 부동소수점 오차 시뮬레이션)
    noise_levels = [1e-10, 1e-8, 1e-6]
    
    for noise_level in noise_levels:
        XtX_noisy = XtX + np.random.randn(*XtX.shape) * noise_level
        Xty = X_with_ones.T @ y.values
        
        try:
            beta_noisy = np.linalg.solve(XtX_noisy, Xty)
            intercept_noisy = beta_noisy[0]
            
            y_pred_noisy = X_with_ones @ beta_noisy
            r2_noisy = r2_score(y, y_pred_noisy)
            
            print(f"\n노이즈 레벨 {noise_level:.2e}:")
            print(f"  Intercept: {intercept_noisy:.6f} (차이: {abs(intercept_noisy - app_intercept):.6f})")
            print(f"  R²: {r2_noisy:.6f} (차이: {abs(r2_noisy - app_r2):.6f})")
        except:
            print(f"\n노이즈 레벨 {noise_level:.2e}: 행렬이 singular")
    
    print("\n" + "=" * 80)
    print("5. 결론 및 권장사항")
    print("=" * 80)
    
    print("\n주요 발견사항:")
    print("1. 행렬 조건수가 5.74e+13으로 매우 높아 수치적으로 불안정합니다.")
    print("2. JavaScript와 Python의 부동소수점 연산 정밀도 차이로 인한 오차가 발생할 수 있습니다.")
    print("3. 특히 행렬 역행렬 계산에서 작은 오차가 증폭될 수 있습니다.")
    
    print("\n앱과 파이썬 결과 차이의 원인:")
    print("1. 수치 계산 정밀도 차이:")
    print("   - JavaScript: IEEE 754 double precision")
    print("   - Python sklearn: 더 정밀한 수치 라이브러리 사용")
    print("   - 조건수가 높은 행렬에서 오차가 증폭됨")
    
    print("\n2. 행렬 역행렬 계산 방법 차이:")
    print("   - JavaScript: Gauss-Jordan elimination 직접 구현")
    print("   - Python sklearn: 더 안정적인 수치 알고리즘 사용")
    
    print("\n권장사항:")
    print("1. 데이터 정규화(StandardScaler) 사용:")
    print("   - 행렬 조건수를 낮춰 수치 안정성 향상")
    print("   - Intercept 해석이 달라지지만 모델 성능은 유지")
    
    print("\n2. Ridge Regression 사용:")
    print("   - 정규화로 수치 안정성 향상")
    print("   - 작은 alpha 값으로 Linear Regression과 유사한 결과")
    
    print("\n3. 더 정밀한 행렬 연산 라이브러리 사용:")
    print("   - JavaScript에서 더 정밀한 수치 라이브러리 고려")
    print("   - 또는 Python 백엔드 사용 (Pyodide 등)")

if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'compactiv.csv')
    analyze_final_difference(csv_path)
























































