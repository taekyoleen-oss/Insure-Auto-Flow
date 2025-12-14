# Change History

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
