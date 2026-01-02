# 파이썬 데이터 분석 모듈 사용 가이드

이 문서는 `insure-auto-flow` 앱의 모든 데이터 분석 모듈을 파이썬으로 구현한 `data_analysis_modules.py` 파일의 사용 방법을 설명합니다.

## 설치

필요한 패키지를 설치합니다:

```bash
pip install -r requirements.txt
```

## 모듈 구조

### 1. 데이터 로딩 모듈

- `load_data()`: CSV 파일에서 데이터 로드
- `load_xol_data()`: 재보험 클레임 데이터 로드

### 2. 통계 분석 모듈

- `analyze_statistics()`: 기술 통계량 및 상관관계 분석

### 3. 데이터 전처리 모듈

- `select_data()`: 컬럼 선택
- `handle_missing_values()`: 결측치 처리
- `encode_categorical()`: 범주형 변수 인코딩
- `normalize_data()`: 데이터 정규화
- `transform_data()`: 수학적 변환 적용
- `resample_data()`: 클래스 불균형 처리
- `split_data()`: 데이터 분할

### 4. 머신러닝 모델 모듈

- `create_linear_model()`: 선형 회귀 모델 생성
- `create_logistic_regression()`: 로지스틱 회귀 모델 생성
- `create_poisson_regression()`: 포아송 회귀 모델 생성
- `create_decision_tree()`: 의사결정나무 모델 생성
- `create_random_forest()`: 랜덤 포레스트 모델 생성
- `create_svm()`: 서포트 벡터 머신 모델 생성
- `create_knn()`: K-최근접 이웃 모델 생성
- `create_naive_bayes()`: 나이브 베이즈 모델 생성
- `create_lda()`: 선형 판별 분석 모델 생성

### 5. 모델 훈련 및 평가 모듈

- `train_model()`: 모델 훈련
- `score_model()`: 모델 예측
- `evaluate_model()`: 모델 성능 평가

### 6. 비지도 학습 모듈

- `kmeans_clustering()`: K-Means 클러스터링
- `hierarchical_clustering()`: 계층적 클러스터링
- `dbscan_clustering()`: DBSCAN 클러스터링
- `pca_transform()`: 주성분 분석

### 7. 통계 모델 모듈 (statsmodels)

- `run_stats_model()`: 통계 모델 피팅
- `predict_with_statsmodel()`: 통계 모델 예측

### 8. 재보험 분석 모듈

- `fit_loss_distribution()`: 손실 분포 피팅
- `generate_exposure_curve()`: 노출 곡선 생성
- `price_xol_layer()`: XoL 레이어 가격 책정
- `apply_loss_threshold()`: 임계값 적용
- `calculate_ceded_loss()`: 인출 손실 계산
- `price_xol_contract()`: XoL 계약 가격 책정

## 사용 예제

### 예제 1: 기본 데이터 분석 파이프라인

```python
from data_analysis_modules import *

# 1. 데이터 로드
df = load_data('data.csv')

# 2. 통계 분석
desc_stats, corr_matrix = analyze_statistics(df)

# 3. 결측치 처리
df_clean = handle_missing_values(df, method='impute', strategy='mean')

# 4. 범주형 변수 인코딩
df_encoded = encode_categorical(df_clean, method='one_hot', columns=['category_col'])

# 5. 데이터 정규화
df_normalized = normalize_data(df_encoded, method='StandardScaler',
                               columns=['feature1', 'feature2'])

# 6. 데이터 분할
train_df, test_df = split_data(df_normalized, train_size=0.7, random_state=42)

# 7. 모델 생성 및 훈련
model = create_linear_model('LinearRegression')
model = train_model(model, train_df,
                    feature_columns=['feature1', 'feature2'],
                    label_column='target')

# 8. 예측 및 평가
scored_df = score_model(model, test_df,
                       feature_columns=['feature1', 'feature2'])
metrics = evaluate_model(model, scored_df,
                        label_column='target',
                        model_type='regression')
```

### 예제 2: 분류 모델 파이프라인

```python
from data_analysis_modules import *

# 데이터 로드 및 전처리
df = load_data('classification_data.csv')
df_clean = handle_missing_values(df, method='remove_row')

# 클래스 불균형 처리
df_resampled = resample_data(df_clean, method='SMOTE', target_column='label')

# 데이터 분할
train_df, test_df = split_data(df_resampled, train_size=0.7,
                               stratify=True, stratify_column='label')

# 로지스틱 회귀 모델 생성 및 훈련
model = create_logistic_regression(penalty='l2', C=1.0)
model = train_model(model, train_df,
                    feature_columns=['feature1', 'feature2', 'feature3'],
                    label_column='label')

# 예측 및 평가
scored_df = score_model(model, test_df,
                       feature_columns=['feature1', 'feature2', 'feature3'])
metrics = evaluate_model(model, scored_df,
                        label_column='label',
                        model_type='classification')
```

### 예제 3: 비지도 학습 - 클러스터링

```python
from data_analysis_modules import *

# 데이터 로드
df = load_data('clustering_data.csv')

# K-Means 클러스터링
df_clustered, kmeans_model = kmeans_clustering(df, n_clusters=3,
                                               feature_columns=['x', 'y', 'z'])

# 계층적 클러스터링
df_hierarchical = hierarchical_clustering(df, n_clusters=3,
                                         feature_columns=['x', 'y', 'z'])

# DBSCAN 클러스터링
df_dbscan, n_clusters, n_noise = dbscan_clustering(df, eps=0.5, min_samples=5,
                                                    feature_columns=['x', 'y', 'z'])

# PCA 차원 축소
df_pca, explained_variance, pca_model = pca_transform(df, n_components=2,
                                                      feature_columns=['x', 'y', 'z'])
```

### 예제 4: 통계 모델 (statsmodels)

```python
from data_analysis_modules import *

# 데이터 로드 및 전처리
df = load_data('statistical_data.csv')
df_clean = handle_missing_values(df, method='impute', strategy='mean')

# OLS 회귀 모델 피팅
results = run_stats_model(df_clean, model_type='OLS',
                         feature_columns=['feature1', 'feature2'],
                         label_column='target')

# 예측
predictions_df = predict_with_statsmodel(results, df_clean,
                                        feature_columns=['feature1', 'feature2'])
```

### 예제 5: 재보험 분석 파이프라인

```python
from data_analysis_modules import *

# 1. XoL 클레임 데이터 로드
xol_df = load_xol_data('xol_claims.csv')

# 2. 임계값 적용 (큰 손실만 필터링)
filtered_df = apply_loss_threshold(xol_df, threshold=100000, loss_column='loss')

# 3. 손실 분포 피팅
params = fit_loss_distribution(filtered_df, loss_column='loss', dist_type='Pareto')

# 4. 노출 곡선 생성
total_loss = filtered_df['loss'].sum()
curve = generate_exposure_curve('Pareto', params, total_loss)

# 5. 레이어 가격 책정
premium, expected_loss, rol = price_xol_layer(
    curve, total_loss, retention=1000000, limit=5000000, loading_factor=1.5
)

# 6. 경험 기반 가격 책정
contract = {
    'deductible': 250000,
    'limit': 1000000,
    'reinstatements': 1,
    'agg_deductible': 0,
    'expense_ratio': 0.3
}

ceded_df = calculate_ceded_loss(filtered_df,
                               deductible=contract['deductible'],
                               limit=contract['limit'],
                               loss_column='loss')

final_premium = price_xol_contract(ceded_df, contract,
                                  volatility_loading=25.0,
                                  year_column='year',
                                  ceded_loss_column='ceded_loss')
```

## 주요 파라미터 설명

### 결측치 처리 (`handle_missing_values`)

- `method`: 'remove_row' (행 제거), 'impute' (대체), 'knn' (KNN 대체)
- `strategy`: 'mean', 'median', 'mode' (method='impute'일 때)
- `n_neighbors`: KNN 방법 사용 시 이웃 수

### 범주형 인코딩 (`encode_categorical`)

- `method`: 'label' (라벨 인코딩), 'one_hot' (원-핫 인코딩), 'ordinal' (순서 인코딩)
- `drop`: 'first', 'if_binary', None (one_hot 인코딩 시)

### 데이터 정규화 (`normalize_data`)

- `method`: 'MinMax', 'StandardScaler', 'RobustScaler'

### 데이터 분할 (`split_data`)

- `train_size`: 훈련 세트 비율 (0.0 ~ 1.0)
- `stratify`: 계층화 여부 (클래스 비율 유지)
- `stratify_column`: 계층화 기준 컬럼

### 모델 평가 (`evaluate_model`)

- `model_type`: 'classification' (분류), 'regression' (회귀)
- 분류 모델: 정확도, 분류 리포트, 혼동 행렬
- 회귀 모델: MSE, RMSE, R²

## 주의사항

1. **데이터 타입**: 각 함수는 적절한 데이터 타입을 요구합니다. 수치형 변환이 필요한 경우 먼저 처리하세요.

2. **컬럼 이름**: 함수 호출 시 컬럼 이름이 정확히 일치해야 합니다.

3. **메모리**: 큰 데이터셋의 경우 메모리 사용량을 고려하세요.

4. **시각화**: `analyze_statistics()` 함수는 matplotlib을 사용하여 상관관계 히트맵을 표시합니다.

## 문제 해결

### ImportError 발생 시

```bash
pip install --upgrade -r requirements.txt
```

### 메모리 부족 시

- 데이터 샘플링 사용
- 배치 처리 고려
- 더 작은 데이터셋으로 테스트

### 모델 수렴 문제

- 하이퍼파라미터 조정
- 데이터 전처리 재검토
- 다른 모델 타입 시도

## 추가 리소스

- [scikit-learn 문서](https://scikit-learn.org/stable/)
- [pandas 문서](https://pandas.pydata.org/docs/)
- [statsmodels 문서](https://www.statsmodels.org/stable/index.html)

























































