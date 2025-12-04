"""
앱과 파이썬 결과 차이 상세 분석
앱 결과: intercept=70.06515, R²=0.7204
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def detailed_analysis(csv_path: str):
    """
    앱과 파이썬 결과 차이를 상세히 분석합니다.
    """
    print("=" * 80)
    print("앱 vs 파이썬 결과 차이 상세 분석")
    print("=" * 80)
    
    # 앱 결과
    app_intercept = 70.06515
    app_r2 = 0.7204
    
    # 데이터 로드
    df = pd.read_csv(csv_path)
    df = df.dropna()
    
    feature_columns = [col for col in df.columns if col != 'y']
    X = df[feature_columns]
    y = df['y']
    
    print(f"\n데이터 Shape: {df.shape}")
    print(f"특성 개수: {len(feature_columns)}")
    
    # 가장 유사한 결과 찾기
    best_match = None
    best_diff = float('inf')
    
    print("\n" + "=" * 80)
    print("시나리오별 테스트")
    print("=" * 80)
    
    scenarios = []
    
    # 시나리오 1: 전체 데이터 + Linear Regression
    lr_full = LinearRegression(fit_intercept=True)
    lr_full.fit(X, y)
    y_pred = lr_full.predict(X)
    r2 = r2_score(y, y_pred)
    diff = abs(lr_full.intercept_ - app_intercept) + abs(r2 - app_r2)
    scenarios.append({
        'name': '전체 데이터 + Linear Regression',
        'intercept': lr_full.intercept_,
        'r2': r2,
        'diff': diff
    })
    print(f"\n1. 전체 데이터 + Linear Regression:")
    print(f"   Intercept: {lr_full.intercept_:.6f} (차이: {abs(lr_full.intercept_ - app_intercept):.6f})")
    print(f"   R²: {r2:.6f} (차이: {abs(r2 - app_r2):.6f})")
    
    # 시나리오 2: Split 데이터 + Linear Regression
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=42, shuffle=True
    )
    lr_split = LinearRegression(fit_intercept=True)
    lr_split.fit(X_train, y_train)
    y_pred_train = lr_split.predict(X_train)
    r2_train = r2_score(y_train, y_pred_train)
    diff = abs(lr_split.intercept_ - app_intercept) + abs(r2_train - app_r2)
    scenarios.append({
        'name': 'Split (train) + Linear Regression',
        'intercept': lr_split.intercept_,
        'r2': r2_train,
        'diff': diff
    })
    print(f"\n2. Split (train) + Linear Regression:")
    print(f"   Intercept: {lr_split.intercept_:.6f} (차이: {abs(lr_split.intercept_ - app_intercept):.6f})")
    print(f"   R²: {r2_train:.6f} (차이: {abs(r2_train - app_r2):.6f})")
    
    # 시나리오 3: Split 데이터 + Lasso (다양한 alpha)
    print(f"\n3. Split (train) + Lasso (다양한 alpha):")
    for alpha in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        lasso = Lasso(alpha=alpha, fit_intercept=True, random_state=42, max_iter=10000)
        lasso.fit(X_train, y_train)
        y_pred_lasso = lasso.predict(X_train)
        r2_lasso = r2_score(y_train, y_pred_lasso)
        diff = abs(lasso.intercept_ - app_intercept) + abs(r2_lasso - app_r2)
        scenarios.append({
            'name': f'Split (train) + Lasso (alpha={alpha})',
            'intercept': lasso.intercept_,
            'r2': r2_lasso,
            'diff': diff
        })
        print(f"   alpha={alpha:5.1f}: Intercept={lasso.intercept_:.6f}, R²={r2_lasso:.6f}, 총차이={diff:.6f}")
        if diff < best_diff:
            best_diff = diff
            best_match = {
                'name': f'Split (train) + Lasso (alpha={alpha})',
                'intercept': lasso.intercept_,
                'r2': r2_lasso,
                'alpha': alpha
            }
    
    # 시나리오 4: 다른 random_state로 Split
    print(f"\n4. 다른 random_state로 Split + Lasso (alpha=1.0):")
    for rs in [0, 1, 10, 20, 30, 40, 41, 43, 50, 100]:
        X_train_rs, X_test_rs, y_train_rs, y_test_rs = train_test_split(
            X, y, train_size=0.7, random_state=rs, shuffle=True
        )
        lasso_rs = Lasso(alpha=1.0, fit_intercept=True, random_state=42, max_iter=10000)
        lasso_rs.fit(X_train_rs, y_train_rs)
        y_pred_rs = lasso_rs.predict(X_train_rs)
        r2_rs = r2_score(y_train_rs, y_pred_rs)
        diff = abs(lasso_rs.intercept_ - app_intercept) + abs(r2_rs - app_r2)
        scenarios.append({
            'name': f'Split (rs={rs}) + Lasso (alpha=1.0)',
            'intercept': lasso_rs.intercept_,
            'r2': r2_rs,
            'diff': diff
        })
        if diff < best_diff:
            best_diff = diff
            best_match = {
                'name': f'Split (rs={rs}) + Lasso (alpha=1.0)',
                'intercept': lasso_rs.intercept_,
                'r2': r2_rs,
                'alpha': 1.0,
                'random_state': rs
            }
        if rs in [0, 1, 42, 43]:
            print(f"   random_state={rs:3d}: Intercept={lasso_rs.intercept_:.6f}, R²={r2_rs:.6f}, 총차이={diff:.6f}")
    
    # 시나리오 5: 특성 선택 (상위 N개)
    print(f"\n5. 특성 선택 + Split + Lasso:")
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    for n_features in [10, 15, 18, 20, 21]:
        top_features = correlations.head(n_features).index.tolist()
        X_top = X[top_features]
        X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(
            X_top, y, train_size=0.7, random_state=42, shuffle=True
        )
        lasso_top = Lasso(alpha=1.0, fit_intercept=True, random_state=42, max_iter=10000)
        lasso_top.fit(X_train_top, y_train_top)
        y_pred_top = lasso_top.predict(X_train_top)
        r2_top = r2_score(y_train_top, y_pred_top)
        diff = abs(lasso_top.intercept_ - app_intercept) + abs(r2_top - app_r2)
        scenarios.append({
            'name': f'Split + {n_features} features + Lasso',
            'intercept': lasso_top.intercept_,
            'r2': r2_top,
            'diff': diff
        })
        print(f"   {n_features} features: Intercept={lasso_top.intercept_:.6f}, R²={r2_top:.6f}, 총차이={diff:.6f}")
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("결과 요약")
    print("=" * 80)
    print(f"\n앱 결과:")
    print(f"  Intercept: {app_intercept:.6f}")
    print(f"  R² Score: {app_r2:.6f}")
    
    if best_match:
        print(f"\n가장 유사한 시나리오:")
        print(f"  {best_match['name']}")
        print(f"  Intercept: {best_match['intercept']:.6f} (차이: {abs(best_match['intercept'] - app_intercept):.6f})")
        print(f"  R² Score: {best_match['r2']:.6f} (차이: {abs(best_match['r2'] - app_r2):.6f})")
        if 'alpha' in best_match:
            print(f"  Alpha: {best_match['alpha']}")
        if 'random_state' in best_match:
            print(f"  Random State: {best_match['random_state']}")
    
    # 상위 5개 유사한 시나리오
    scenarios_sorted = sorted(scenarios, key=lambda x: x['diff'])[:5]
    print(f"\n상위 5개 유사한 시나리오:")
    for i, s in enumerate(scenarios_sorted, 1):
        print(f"\n{i}. {s['name']}")
        print(f"   Intercept: {s['intercept']:.6f} (차이: {abs(s['intercept'] - app_intercept):.6f})")
        print(f"   R²: {s['r2']:.6f} (차이: {abs(s['r2'] - app_r2):.6f})")
        print(f"   총 차이: {s['diff']:.6f}")
    
    # 결론
    print("\n" + "=" * 80)
    print("결론")
    print("=" * 80)
    print("\n가능한 원인:")
    print("1. 앱에서 데이터 분할(Split)을 사용했을 가능성이 높습니다.")
    print("2. 앱에서 Lasso 모델을 사용했을 가능성이 있습니다.")
    print("3. 앱에서 특정 alpha 값을 사용했을 수 있습니다.")
    print("4. 앱에서 다른 random_state를 사용했을 수 있습니다.")
    print("5. JavaScript의 수치 계산 정밀도 차이로 인한 오차가 있을 수 있습니다.")
    print("\n추천 조치:")
    print("- 앱에서 사용한 정확한 파라미터(Split 여부, 모델 타입, alpha 값 등)를 확인하세요.")
    print("- 앱의 TrainModel 모듈 설정을 확인하세요.")

if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'compactiv.csv')
    detailed_analysis(csv_path)


















