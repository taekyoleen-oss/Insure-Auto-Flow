## [2025-12-04 17:45:00] - Linear Regression 모듈 하이퍼파라미터 튜닝 추가

변경 사항:

- Linear Regression 모듈에 Ridge/Lasso/ElasticNet용 하이퍼파라미터 튜닝 옵션(Alpha/L1 Ratio 후보, CV 폴드, 스코어링) 추가
- Pyodide 기반 학습 로직에 GridSearchCV를 적용하여 최적 파라미터 및 점수를 계산
- TrainedModelPreviewModal에 튜닝 결과(전략, 스코어, 최적 파라미터, 후보 점수표) 표시 섹션 추가
- HISTORY 및 history 로그 업데이트

영향받은 파일:

- App.tsx
- constants.ts
- components/PropertiesPanel.tsx
- components/TrainedModelPreviewModal.tsx
- utils/pyodideRunner.ts
- types.ts
- HISTORY.md
- history

커밋 해시: 48c3d80

복구 방법

# 특정 커밋으로 되돌리기

git checkout 48c3d80

# 또는 현재 브랜치에서 이 커밋 상태로 되돌리기 (변경사항 유지)

git reset --soft 48c3d80

# 완전히 이 커밋 상태로 되돌리기 (변경사항 삭제)

git reset --hard 48c3d80

---

## [2025-12-04 16:10:45] - Samples 버튼 클릭 이벤트 및 에러 처리 개선

변경 사항:

- Samples 버튼 클릭 시 이벤트 전파 방지 추가

- 메뉴 표시 시 z-index를 z-50에서 z-[100]으로 상향 조정하여 다른 요소에 가려지지 않도록 개선

- handleLoadSample 함수에 try-catch 블록 추가하여 에러 처리 강화

- defaultModule이 undefined인 경우 에러 처리 추가

- 연결(connection) 유효성 검사 추가

영향받은 파일:

- App.tsx

복구 방법

# 특정 커밋으로 되돌리기

git checkout <커밋해시>

# 또는 현재 브랜치에서 이 커밋 상태로 되돌리기 (변경사항 유지)

git reset --soft <커밋해시>

# 완전히 이 커밋 상태로 되돌리기 (변경사항 삭제)

git reset --hard <커밋해시>

---

## [2025-12-04 16:04:46] - Rotate Modules 버튼 간격 조정 기능 추가

변경 사항:

- 가로 모드로 변환 시 모듈 간격을 2배로 넓게 조정

- 세로 모드로 변환 시 모듈 간격을 2배로 작게 조정

- 회전 후 모듈들의 중심점을 기준으로 간격 자동 조정

- spacingMultiplier 변수를 사용하여 가로/세로 모드에 따라 간격 조정

영향받은 파일:

- App.tsx

복구 방법

# 특정 커밋으로 되돌리기

git checkout <커밋해시>

# 또는 현재 브랜치에서 이 커밋 상태로 되돌리기 (변경사항 유지)

git reset --soft <커밋해시>

# 완전히 이 커밋 상태로 되돌리기 (변경사항 삭제)

git reset --hard <커밋해시>

---

## [2025-12-04 16:02:06] - 가로/세로 변환 버튼 추가

변경 사항:

- 하단 중앙 컨트롤 패널에 Rotate Modules 버튼 추가

- ArrowPathIcon 아이콘 추가 (components/icons.tsx)

- handleRotateModules 함수 구현

- 세로가 긴 경우 90도 반시계방향으로 회전

- 가로가 긴 경우 90도 시계방향으로 회전

- 모듈들의 중심점을 기준으로 회전 처리

영향받은 파일:

- App.tsx

- components/icons.tsx

복구 방법

# 특정 커밋으로 되돌리기

git checkout <커밋해시>

# 또는 현재 브랜치에서 이 커밋 상태로 되돌리기 (변경사항 유지)

git reset --soft <커밋해시>

# 완전히 이 커밋 상태로 되돌리기 (변경사항 삭제)

git reset --hard <커밋해시>

---

## [2025-12-04 15:58:53] - 프로젝트 설정 및 문서 업데이트

변경 사항:

- .cursorrules 파일 추가

- 히스토리 로그 업데이트

- 테스트 파일 및 문서 업데이트

영향받은 파일:

- .cursorrules

- history

- 기타 문서 파일 27개

복구 방법

# 특정 커밋으로 되돌리기

git checkout 7171581

# 또는 현재 브랜치에서 이 커밋 상태로 되돌리기 (변경사항 유지)

git reset --soft 7171581

# 완전히 이 커밋 상태로 되돌리기 (변경사항 삭제)

git reset --hard 7171581

---

## [2025-12-04 14:44:03] - Samples 버튼 위치 변경 및 전체 코드 패널 기능 추가

변경 사항:

- Samples 버튼을 햄버거 버튼 우측으로 이동

- 전체 코드 패널 및 파이프라인 코드 생성 기능 추가

- 히스토리 로그 파일 추가

영향받은 파일:

- App.tsx

- components/PipelineCodePanel.tsx

- utils/generatePipelineCode.ts

복구 방법

# 특정 커밋으로 되돌리기

git checkout <커밋해시>

# 또는 현재 브랜치에서 이 커밋 상태로 되돌리기 (변경사항 유지)

git reset --soft <커밋해시>

# 완전히 이 커밋 상태로 되돌리기 (변경사항 삭제)

git reset --hard <커밋해시>

---

## [2025-12-04 14:40:24] - Samples 버튼 위치 변경

변경 사항:

- Samples 버튼을 두 번째 줄에서 세 번째 줄로 이동

- 햄버거 버튼 바로 우측에 배치

영향받은 파일:

- App.tsx

복구 방법

# 특정 커밋으로 되돌리기

git checkout <커밋해시>

# 또는 현재 브랜치에서 이 커밋 상태로 되돌리기 (변경사항 유지)

git reset --soft <커밋해시>

# 완전히 이 커밋 상태로 되돌리기 (변경사항 삭제)

git reset --hard <커밋해시>

---

## [2025-12-04 14:38:24] - generatePipelineCode.ts 파일 복구

변경 사항:

- 삭제된 generatePipelineCode.ts 파일 복구

- generateFullPipelineCode 함수 재생성

- 전체 파이프라인 코드 생성 기능 복원

영향받은 파일:

- utils/generatePipelineCode.ts

복구 방법

# 특정 커밋으로 되돌리기

git checkout <커밋해시>

# 또는 현재 브랜치에서 이 커밋 상태로 되돌리기 (변경사항 유지)

git reset --soft <커밋해시>

# 완전히 이 커밋 상태로 되돌리기 (변경사항 삭제)

git reset --hard <커밋해시>

---

## [2025-12-04 14:33:57] - 전체 코드 버튼 및 패널 복구

변경 사항:

- PipelineCodePanel 컴포넌트 재생성

- App.tsx에 통합

- 오른쪽에서 나타나도록 구현

- 복사 버튼 추가

영향받은 파일:

- components/PipelineCodePanel.tsx

- App.tsx

복구 방법

# 특정 커밋으로 되돌리기

git checkout <커밋해시>

# 또는 현재 브랜치에서 이 커밋 상태로 되돌리기 (변경사항 유지)

git reset --soft <커밋해시>

# 완전히 이 커밋 상태로 되돌리기 (변경사항 삭제)

git reset --hard <커밋해시>

---

## [2025-12-04 14:30:45] - Toolbox Panel 위치 수정

변경 사항:

- Toolbox Panel이 항상 가장 왼쪽(left: 0)에 위치하도록 style 속성 추가

- 왼쪽 모듈이 항상 가장 왼쪽에서 열리도록 명시적으로 left: 0 스타일 추가

영향받은 파일:

- App.tsx

복구 방법

# 특정 커밋으로 되돌리기

git checkout <커밋해시>

# 또는 현재 브랜치에서 이 커밋 상태로 되돌리기 (변경사항 유지)

git reset --soft <커밋해시>

# 완전히 이 커밋 상태로 되돌리기 (변경사항 삭제)

git reset --hard <커밋해시>

---

## [2025-12-04 14:28:59] - 전체 코드 패널 z-index 조정

변경 사항:

- PipelineCodePanel의 z-index를 z-20에서 z-10으로 변경

- 팝업이 항상 위에 표시되도록 수정

- 전체 코드 패널이 열려있을 때 팝업이 가운데로 몰려서 발생하는 문제 해결

영향받은 파일:

- components/PipelineCodePanel.tsx

복구 방법

# 특정 커밋으로 되돌리기

git checkout <커밋해시>

# 또는 현재 브랜치에서 이 커밋 상태로 되돌리기 (변경사항 유지)

git reset --soft <커밋해시>

# 완전히 이 커밋 상태로 되돌리기 (변경사항 삭제)

git reset --hard <커밋해시>

---

## [2025-12-04 14:27:17] - .cursorrules 파일 생성 및 history 자동 기록 규칙 추가

변경 사항:

- 프로젝트 루트에 .cursorrules 파일 생성

- 명령 실행 시마다 history 파일에 위치, 명령어, 설명, 시간 등을 자동으로 기록하는 규칙 추가

영향받은 파일:

- .cursorrules

복구 방법

# 특정 커밋으로 되돌리기

git checkout <커밋해시>

# 또는 현재 브랜치에서 이 커밋 상태로 되돌리기 (변경사항 유지)

git reset --soft <커밋해시>

# 완전히 이 커밋 상태로 되돌리기 (변경사항 삭제)

git reset --hard <커밋해시>

---
