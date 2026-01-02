# Change History

## 2025-01-XX XX:XX:XX

### feat(modules): Update Random Forest module and fix Decision Tree visualization

**Description:**
- Random Forest 모듈 파라미터 레이블 변경 (N Estimators → n_estimators, Max Depth → max_depth)
- Random Forest 모듈에 max_features 파라미터 추가 (기본값: None - 전체 특징 사용)
- PropertiesPanel에 max_features UI 추가 (None, auto, sqrt, log2, 커스텀 숫자 입력 옵션)
- data_analysis_modules.py의 create_random_forest 함수에 max_features 파라미터 지원 추가
- Decision Tree View Details 시각화 수정: TrainedModelOutput 인터페이스에 trainingData와 modelParameters 필드 추가

**Files Affected:**
- `types.ts` - TrainedModelOutput 인터페이스에 trainingData와 modelParameters 필드 추가
- `constants.ts` - Random Forest 모듈에 max_features 파라미터 추가
- `components/PropertiesPanel.tsx` - Random Forest 파라미터 레이블 변경 및 max_features UI 추가
- `data_analysis_modules.py` - create_random_forest 함수에 max_features 파라미터 추가

**Reason:**
- Random Forest 모듈 파라미터 이름을 Python 스타일로 통일
- max_features 파라미터를 통해 각 트리에서 고려할 특징 수를 제어할 수 있도록 기능 확장
- Decision Tree 모듈의 View Details에서 트리 시각화가 정상적으로 표시되도록 수정

**Commit Hash:** b5fdcfe

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard b5fdcfe

# Or direct recovery
git reset --hard b5fdcfe
```

## 2025-01-XX XX:XX:XX

### feat(modules): Add Column Plot module with comprehensive chart visualization

**Description:**
- Column Plot 모듈 추가: 단일열 또는 2개열 선택을 통한 다양한 차트 시각화 기능
- 컬럼 데이터 타입(숫자형/범주형)에 따라 자동으로 적절한 차트 옵션 제공
- View Details 모달에서 차트 타입 선택 및 실시간 차트 생성
- matplotlib 기반 Python 차트 생성 (seaborn 의존성 제거)
- 지원하는 차트 타입:
  - 단일열 숫자형: Histogram, KDE Plot, Boxplot, Violin Plot, ECDF Plot, QQ-Plot, Line Plot, Area Plot
  - 단일열 범주형: Bar Plot, Count Plot, Pie Chart, Frequency Table
  - 2개열 숫자형+숫자형: Scatter Plot, Hexbin Plot, Joint Plot, Line Plot, Regression Plot, Heatmap
  - 2개열 숫자형+범주형: Box Plot, Violin Plot, Bar Plot, Strip Plot, Swarm Plot
  - 2개열 범주형+범주형: Grouped Bar Plot, Heatmap, Mosaic Plot

**Files Affected:**
- `types.ts` - ModuleType.ColumnPlot 및 ColumnPlotOutput 타입 추가
- `constants.ts` - Column Plot 모듈 정의 추가
- `components/Toolbox.tsx` - Column Plot 모듈을 Toolbox에 추가
- `components/PropertiesPanel.tsx` - Column Plot 속성 UI 구현 (단일열/2개열 선택)
- `components/ColumnPlotPreviewModal.tsx` - View Details 모달 생성 (새 파일)
- `utils/pyodideRunner.ts` - createColumnPlotPython 함수 추가 (matplotlib만 사용)
- `App.tsx` - Column Plot 실행 로직 및 모달 연결 추가

**Reason:**
- 데이터 시각화 기능 확장
- 사용자가 데이터를 다양한 방식으로 시각화할 수 있도록 지원
- 컬럼 타입에 따른 자동 차트 옵션 제공으로 사용자 편의성 향상

**Commit Hash:** 1b7ea562b784a8fbd28a3e8041efd74f6ee6d8cf

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard 1b7ea562b784a8fbd28a3e8041efd74f6ee6d8cf

# Or direct recovery
git reset --hard 1b7ea562b784a8fbd28a3e8041efd74f6ee6d8cf
```

## 2025-12-14 09:35:26

### feat(pca): Improve PCA visualization with JavaScript-based implementation

**Description:**
- PCA 시각화를 JavaScript 기반(ml-pca)으로 전환하여 성능 개선
- Python Pyodide 의존성 제거로 안정성 향상
- Label Column을 선택 사항으로 변경하고 Predict를 기본값으로 설정
- 이진 분류(0/1)를 위한 간소한 색상 체계 적용 (파란색: 클래스 0, 빨간색: 클래스 1)
- Color Scale 범례 제거 및 그래프 너비 1400px로 확장
- 콤보박스에서 "None (Basic PCA)" 옵션 제거
- 그래프 가시성 개선 (그리드 라인, 축 레이블, 레이아웃 개선)

**Files Affected:**
- `utils/pcaCalculator.ts` - ml-pca 라이브러리를 사용한 JavaScript 기반 PCA 계산 함수 추가
- `components/DataPreviewModal.tsx` - PCA Visualization 개선 (Label Column 선택 사항화, 그래프 크기 및 스타일 개선)
- `package.json` - ml-pca 라이브러리 의존성 추가
- `pnpm-lock.yaml` - 의존성 업데이트

**Reason:**
- Python Pyodide 기반 PCA 구현에서 발생한 패키지 로딩 및 데이터 마샬링 문제 해결
- 브라우저 환경에서 더 안정적이고 빠른 PCA 계산을 위해 JavaScript 기반 구현으로 전환
- 사용자 경험 개선을 위한 시각화 개선

**Commit Hash:** de7bb9092853b58ba903cf6788e0904a2c4d05d7

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard de7bb9092853b58ba903cf6788e0904a2c4d05d7

# Or direct recovery
git reset --hard de7bb9092853b58ba903cf6788e0904a2c4d05d7
```

## 2025-12-12 17:50:00

### feat(samples): Add Samples folder support and Linear Regression-1 sample

**Description:**
- Samples 폴더 기능 추가 및 Linear Regression-1 샘플 추가
- Samples 폴더의 파일을 자동으로 읽어서 Samples 메뉴에 표시하는 기능 구현
- Save 버튼으로 저장한 .mla 파일을 samples 폴더에 넣으면 자동으로 표시되도록 개선
- File System Access API 오류 처리 개선
- 파일 이름의 공백 및 특수문자 처리 (URL 인코딩/디코딩)

**Files Affected:**
- `App.tsx` - Samples 폴더 파일 로드 기능 추가, File System Access API 오류 처리 개선
- `server/samples-server.js` - Samples 폴더 파일 목록 및 읽기 API 구현
- `savedSamples.ts` - Linear Regression-1 샘플 추가
- `samples/README.md` - Samples 폴더 사용 방법 문서 추가
- `samples/example.json` - 예제 파일 추가
- `package.json` - samples-server 스크립트 추가
- `vite.config.ts` - /api/samples 프록시 설정 추가
- `types.ts` - StatsModelFamily에 Logit, QuasiPoisson 추가, DiversionCheckerOutput, EvaluateStatOutput 타입 추가

**Reason:**
- 사용자가 Save 버튼으로 저장한 모델을 samples 폴더에 넣으면 자동으로 Samples 메뉴에 표시되도록 하기 위해
- Linear Regression-1 샘플을 공유 가능한 샘플로 추가

**Commit Hash:** b7dfe9fc6c744f5d41e2d417afa575205c80fbec

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard b7dfe9fc6c744f5d41e2d417afa575205c80fbec

# Or direct recovery
git reset --hard b7dfe9fc6c744f5d41e2d417afa575205c80fbec
```
