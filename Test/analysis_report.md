# 앱 vs 파이썬 결과 차이 분석 보고서

## 앱 결과
- **Intercept (y-절편)**: 70.06515
- **R² Score**: 0.7204

## 파이썬 결과

### 전체 데이터 사용
- **Linear Regression**: Intercept=65.498498, R²=0.728924
- **Lasso (alpha=1.0)**: Intercept=65.995145, R²=0.724256

### Split 데이터 사용 (train_size=0.7, random_state=42)
- **Linear Regression**: Intercept=65.333559, R²=0.725773
- **Lasso (alpha=1.0)**: Intercept=65.750716, R²=0.721494

## 차이 분석

### 1. Intercept 차이
- **앱**: 70.06515
- **파이썬 (전체 데이터 Linear Regression)**: 65.498498
- **차이**: 약 4.57

### 2. R² Score 차이
- **앱**: 0.7204
- **파이썬 (전체 데이터 Linear Regression)**: 0.728924
- **차이**: 약 0.0085

## 가장 유사한 시나리오

**Split (random_state=10) + Lasso (alpha=1.0)**
- Intercept: 69.858489 (차이: 0.21)
- R²: 0.739557 (차이: 0.02)

## 가능한 원인

### 1. 데이터 분할 (Split) 사용
- 앱에서 데이터를 분할했을 가능성이 높습니다
- Split을 사용하면 학습 데이터가 줄어들어 결과가 달라질 수 있습니다

### 2. 모델 타입 차이
- 앱에서 **Lasso** 모델을 사용했을 가능성이 있습니다
- R²=0.7204는 Lasso의 결과와 더 유사합니다 (Linear Regression: 0.728924, Lasso: 0.724256)

### 3. Random State 차이
- 앱에서 사용한 random_state가 파이썬의 기본값(42)과 다를 수 있습니다
- random_state=10일 때 Intercept가 69.86으로 앱 결과(70.07)에 가장 가깝습니다

### 4. Alpha 값 차이
- Lasso의 alpha 값이 1.0이 아닐 수 있습니다
- 다양한 alpha 값 테스트 결과, alpha=1.0 근처에서 유사한 결과가 나옵니다

### 5. JavaScript 수치 계산 정밀도
- 앱은 JavaScript로 직접 구현한 행렬 연산을 사용합니다
- Python의 sklearn과 JavaScript 구현 간의 수치 계산 정밀도 차이가 있을 수 있습니다
- 특히 행렬 역행렬 계산에서 미세한 차이가 발생할 수 있습니다

### 6. 특성 선택
- 앱에서 일부 특성만 선택했을 가능성도 있습니다
- 하지만 21개 특성 모두 사용하는 것이 가장 유사한 결과를 보입니다

## 결론

앱과 파이썬 결과의 차이는 다음과 같은 요인들의 조합으로 발생한 것으로 보입니다:

1. **데이터 분할**: 앱에서 Split을 사용했을 가능성 높음
2. **모델 타입**: Lasso 모델 사용 가능성
3. **Random State**: 42가 아닌 다른 값 사용 가능성
4. **JavaScript 수치 계산**: 행렬 연산의 정밀도 차이

## 추천 조치

1. 앱의 TrainModel 모듈 설정 확인:
   - Split Data 모듈 사용 여부
   - Random State 값
   - 모델 타입 (Linear Regression vs Lasso)
   - Alpha 값 (Lasso 사용 시)

2. 앱의 JavaScript 구현 검증:
   - 행렬 연산 정확도 확인
   - 수치 계산 정밀도 검증

3. 동일한 조건으로 재실행:
   - 앱과 동일한 파라미터로 파이썬 코드 실행
   - 결과 비교 및 검증









































