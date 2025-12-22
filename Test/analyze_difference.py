"""
앱과 파이썬 결과 차이 분석
앱 결과: intercept=70.06515, R²=0.7204
파이썬 결과: intercept=65.498498 (Linear Regression), intercept=65.995145 (Lasso), R²=0.728924 (Linear Regression), R²=0.724256 (Lasso)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def analyze_differences(csv_path: str):
    """
    앱과 파이썬 결과 차이를 분석합니다.
    """
    print("=" * 80)
    print("앱 vs 파이썬 결과 차이 분석")
    print("=" * 80)
    
    # 데이터 로드
    df = pd.read_csv(csv_path)
    print(f"\n전체 데이터 Shape: {df.shape}")
    
    # 결측치 제거
    df = df.dropna()
    print(f"결측치 제거 후 Shape: {df.shape}")
    
    # 특성과 타겟 분리
    feature_columns = [col for col in df.columns if col != 'y']
    X = df[feature_columns]
    y = df['y']
    
    print(f"\n특성 개수: {len(feature_columns)}")
    print(f"샘플 수: {len(X)}")
    
    # 앱 결과
    app_intercept = 70.06515
    app_r2 = 0.7204
    
    print("\n" + "=" * 80)
    print("1. 전체 데이터로 Linear Regression (앱과 비교)")
    print("=" * 80)
    
    # Linear Regression (전체 데이터)
    lr_full = LinearRegression(fit_intercept=True)
    lr_full.fit(X, y)
    y_pred_full = lr_full.predict(X)
    r2_full = r2_score(y, y_pred_full)
    
    print(f"\n전체 데이터 Linear Regression:")
    print(f"  Intercept: {lr_full.intercept_:.6f}")
    print(f"  R² Score: {r2_full:.6f}")
    print(f"  앱과 차이 - Intercept: {abs(lr_full.intercept_ - app_intercept):.6f}")
    print(f"  앱과 차이 - R²: {abs(r2_full - app_r2):.6f}")
    
    # Lasso (전체 데이터)
    lasso_full = Lasso(alpha=1.0, fit_intercept=True, random_state=42, max_iter=10000)
    lasso_full.fit(X, y)
    y_pred_lasso_full = lasso_full.predict(X)
    r2_lasso_full = r2_score(y, y_pred_lasso_full)
    
    print(f"\n전체 데이터 Lasso (alpha=1.0):")
    print(f"  Intercept: {lasso_full.intercept_:.6f}")
    print(f"  R² Score: {r2_lasso_full:.6f}")
    print(f"  앱과 차이 - Intercept: {abs(lasso_full.intercept_ - app_intercept):.6f}")
    print(f"  앱과 차이 - R²: {abs(r2_lasso_full - app_r2):.6f}")
    
    # 다양한 alpha 값으로 Lasso 테스트
    print("\n" + "=" * 80)
    print("2. 다양한 Alpha 값으로 Lasso 테스트")
    print("=" * 80)
    
    alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, fit_intercept=True, random_state=42, max_iter=10000)
        lasso.fit(X, y)
        y_pred = lasso.predict(X)
        r2 = r2_score(y, y_pred)
        intercept_diff = abs(lasso.intercept_ - app_intercept)
        r2_diff = abs(r2 - app_r2)
        print(f"\nLasso (alpha={alpha}):")
        print(f"  Intercept: {lasso.intercept_:.6f} (차이: {intercept_diff:.6f})")
        print(f"  R² Score: {r2:.6f} (차이: {r2_diff:.6f})")
        if intercept_diff < 1.0 and r2_diff < 0.01:
            print(f"  *** 앱 결과와 가장 유사! ***")
    
    # 데이터 분할 테스트 (앱에서 Split을 했을 수도 있음)
    print("\n" + "=" * 80)
    print("3. 데이터 분할 후 분석 (Split 가능성 확인)")
    print("=" * 80)
    
    from sklearn.model_selection import train_test_split
    
    # random_state=42로 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=42, shuffle=True
    )
    
    print(f"Train 데이터: {len(X_train)} 샘플")
    print(f"Test 데이터: {len(X_test)} 샘플")
    
    # Train 데이터로 학습
    lr_train = LinearRegression(fit_intercept=True)
    lr_train.fit(X_train, y_train)
    y_pred_train = lr_train.predict(X_train)
    r2_train = r2_score(y_train, y_pred_train)
    
    print(f"\nTrain 데이터로 학습한 Linear Regression:")
    print(f"  Intercept: {lr_train.intercept_:.6f}")
    print(f"  R² Score (train): {r2_train:.6f}")
    print(f"  앱과 차이 - Intercept: {abs(lr_train.intercept_ - app_intercept):.6f}")
    print(f"  앱과 차이 - R²: {abs(r2_train - app_r2):.6f}")
    
    # Lasso (train 데이터)
    lasso_train = Lasso(alpha=1.0, fit_intercept=True, random_state=42, max_iter=10000)
    lasso_train.fit(X_train, y_train)
    y_pred_lasso_train = lasso_train.predict(X_train)
    r2_lasso_train = r2_score(y_train, y_pred_lasso_train)
    
    print(f"\nTrain 데이터로 학습한 Lasso (alpha=1.0):")
    print(f"  Intercept: {lasso_train.intercept_:.6f}")
    print(f"  R² Score (train): {r2_lasso_train:.6f}")
    print(f"  앱과 차이 - Intercept: {abs(lasso_train.intercept_ - app_intercept):.6f}")
    print(f"  앱과 차이 - R²: {abs(r2_lasso_train - app_r2):.6f}")
    
    # 특성 선택 테스트 (앱에서 일부 특성만 선택했을 수도 있음)
    print("\n" + "=" * 80)
    print("4. 특성 선택 영향 분석")
    print("=" * 80)
    
    # 상위 10개 특성만 사용
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    top_features = correlations.head(10).index.tolist()
    
    print(f"\n상위 10개 특성만 사용:")
    print(f"특성: {top_features}")
    
    X_top10 = X[top_features]
    lr_top10 = LinearRegression(fit_intercept=True)
    lr_top10.fit(X_top10, y)
    y_pred_top10 = lr_top10.predict(X_top10)
    r2_top10 = r2_score(y, y_pred_top10)
    
    print(f"  Intercept: {lr_top10.intercept_:.6f}")
    print(f"  R² Score: {r2_top10:.6f}")
    print(f"  앱과 차이 - Intercept: {abs(lr_top10.intercept_ - app_intercept):.6f}")
    print(f"  앱과 차이 - R²: {abs(r2_top10 - app_r2):.6f}")
    
    # 데이터 정규화 테스트
    print("\n" + "=" * 80)
    print("5. 데이터 정규화 영향 분석")
    print("=" * 80)
    
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)
    
    lr_scaled = LinearRegression(fit_intercept=True)
    lr_scaled.fit(X_scaled_df, y)
    y_pred_scaled = lr_scaled.predict(X_scaled_df)
    r2_scaled = r2_score(y, y_pred_scaled)
    
    print(f"\n정규화된 데이터로 Linear Regression:")
    print(f"  Intercept: {lr_scaled.intercept_:.6f}")
    print(f"  R² Score: {r2_scaled:.6f}")
    print(f"  앱과 차이 - Intercept: {abs(lr_scaled.intercept_ - app_intercept):.6f}")
    print(f"  앱과 차이 - R²: {abs(r2_scaled - app_r2):.6f}")
    
    # 결론
    print("\n" + "=" * 80)
    print("6. 결론 및 추정")
    print("=" * 80)
    print("\n앱 결과:")
    print(f"  Intercept: {app_intercept:.6f}")
    print(f"  R² Score: {app_r2:.6f}")
    print("\n가능한 원인:")
    print("1. 앱에서 데이터 분할(Split)을 사용했을 가능성")
    print("2. 앱에서 Lasso 또는 다른 정규화 모델을 사용했을 가능성")
    print("3. 앱에서 특정 특성만 선택했을 가능성")
    print("4. 앱에서 데이터 전처리(정규화 등)를 했을 가능성")
    print("5. JavaScript 수치 계산 정밀도 차이")

def main():
    """메인 실행 함수"""
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'compactiv.csv')
    
    if not os.path.exists(csv_path):
        print(f"오류: 파일을 찾을 수 없습니다: {csv_path}")
        return
    
    analyze_differences(csv_path)

if __name__ == "__main__":
    main()


















































