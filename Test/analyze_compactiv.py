"""
compactiv.csv 데이터를 사용하여 Linear Regression, Lasso, Ridge 분석
Split 없이 전체 데이터로 분석합니다.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def analyze_models(csv_path: str):
    """
    compactiv.csv 데이터를 사용하여 Linear Regression, Lasso, Ridge 모델을 분석합니다.
    
    Parameters:
    -----------
    csv_path : str
        CSV 파일 경로
    """
    print("=" * 80)
    print("compactiv.csv 데이터 분석 (Linear Regression, Lasso, Ridge)")
    print("=" * 80)
    
    # 데이터 로드
    print("\n1. 데이터 로드")
    print("-" * 80)
    df = pd.read_csv(csv_path)
    print(f"데이터 Shape: {df.shape}")
    print(f"컬럼: {list(df.columns)}")
    
    # 결측치 확인
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\n경고: 결측치 발견")
        print(missing_values[missing_values > 0])
        df = df.dropna()
        print(f"결측치 제거 후 Shape: {df.shape}")
    else:
        print("\n결측치 없음")
    
    # 특성과 타겟 분리
    feature_columns = [col for col in df.columns if col != 'y']
    X = df[feature_columns]
    y = df['y']
    
    print(f"\n특성 개수: {len(feature_columns)}")
    print(f"샘플 수: {len(X)}")
    print(f"타겟 변수: y")
    
    # 데이터 통계
    print("\n2. 데이터 통계")
    print("-" * 80)
    print(f"\n타겟 변수 (y) 통계:")
    print(y.describe())
    
    # 모델 정의 및 학습
    models = {
        'Linear Regression': LinearRegression(fit_intercept=True),
        'Lasso': Lasso(alpha=1.0, fit_intercept=True, random_state=42, max_iter=10000),
        'Ridge': Ridge(alpha=1.0, fit_intercept=True, random_state=42, max_iter=10000)
    }
    
    results = {}
    
    print("\n3. 모델 학습 및 평가")
    print("=" * 80)
    
    for model_name, model in models.items():
        print(f"\n{model_name}")
        print("-" * 80)
        
        # 모델 학습
        print("모델 학습 중...")
        model.fit(X, y)
        print("[OK] 모델 학습 완료")
        
        # 예측
        y_pred = model.predict(X)
        
        # 평가 지표 계산
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        # 결과 저장
        results[model_name] = {
            'model': model,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'y_pred': y_pred
        }
        
        # 결과 출력
        print(f"\n성능 지표:")
        print(f"  R² Score: {r2:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE: {mae:.6f}")
        
        # 계수 정보
        if hasattr(model, 'intercept_'):
            print(f"\n절편 (Intercept): {model.intercept_:.6f}")
        
        if hasattr(model, 'coef_'):
            print(f"\n계수 (Coefficients) - 처음 10개:")
            for i, feature in enumerate(feature_columns[:10]):
                print(f"  {feature}: {model.coef_[i]:.6f}")
            if len(feature_columns) > 10:
                print(f"  ... (총 {len(feature_columns)}개 특성)")
            
            # Lasso의 경우 0이 아닌 계수 개수 확인
            if model_name == 'Lasso':
                non_zero_coef = np.sum(model.coef_ != 0)
                print(f"\nLasso: 0이 아닌 계수 개수: {non_zero_coef} / {len(feature_columns)}")
    
    # 모델 비교
    print("\n4. 모델 비교")
    print("=" * 80)
    print(f"{'모델':<25} {'R² Score':<15} {'RMSE':<15} {'MAE':<15}")
    print("-" * 80)
    for model_name, result in results.items():
        print(f"{model_name:<25} {result['r2']:<15.6f} {result['rmse']:<15.6f} {result['mae']:<15.6f}")
    
    # 예측값 비교 (처음 10개 샘플)
    print("\n5. 예측값 비교 (처음 10개 샘플)")
    print("=" * 80)
    comparison_df = pd.DataFrame({
        '실제값 (y)': y.iloc[:10].values,
        'Linear Regression': results['Linear Regression']['y_pred'][:10],
        'Lasso': results['Lasso']['y_pred'][:10],
        'Ridge': results['Ridge']['y_pred'][:10]
    })
    print(comparison_df.to_string(index=False))
    
    # 예측 오차 분석
    print("\n6. 예측 오차 분석")
    print("=" * 80)
    for model_name, result in results.items():
        errors = y - result['y_pred']
        print(f"\n{model_name}:")
        print(f"  평균 오차: {np.mean(errors):.6f}")
        print(f"  오차 표준편차: {np.std(errors):.6f}")
        print(f"  최대 오차: {np.max(np.abs(errors)):.6f}")
        print(f"  최소 오차: {np.min(np.abs(errors)):.6f}")
    
    return results

def main():
    """메인 실행 함수"""
    import os
    
    # CSV 파일 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'compactiv.csv')
    
    if not os.path.exists(csv_path):
        print(f"오류: 파일을 찾을 수 없습니다: {csv_path}")
        return
    
    # 분석 실행
    results = analyze_models(csv_path)
    
    print("\n" + "=" * 80)
    print("분석 완료")
    print("=" * 80)
    print("\n참고:")
    print("- 전체 데이터를 사용하여 모델을 학습하고 평가했습니다.")
    print("- 데이터 분할(Split) 없이 분석을 수행했습니다.")
    print("- R² Score가 1에 가까울수록 모델이 데이터를 잘 설명합니다.")
    print("- RMSE와 MAE가 작을수록 예측 정확도가 높습니다.")

if __name__ == "__main__":
    main()









































