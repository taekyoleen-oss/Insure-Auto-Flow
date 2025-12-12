# Samples 관리 가이드

## 개요

이 프로젝트는 두 가지 유형의 예시 모델을 지원합니다:

1. **커밋 가능한 예시** - 모든 사용자가 공유하는 예시
2. **로컬 전용 예시** - 각 PC 환경에서만 사용하는 예시

## 예시 유형

### 1. 기본 예시 (커밋 가능)
- 위치: `constants.ts`의 `SAMPLE_MODELS`
- 모든 사용자가 공유
- Git으로 커밋/푸시 가능

### 2. 로컬 예시 (커밋 가능)
- 위치: `localSamples.json` 파일
- 로컬에서 추가한 예시를 이 파일에 저장
- Git으로 커밋/푸시하여 모든 사용자와 공유 가능

### 3. 저장된 모델 (로컬 전용)
- 위치: 브라우저의 `localStorage`
- 웹에서 "현재 모델 저장"으로 저장한 모델
- 해당 PC 환경에서만 사용 가능
- Git에 커밋되지 않음

### 4. 초기 화면 설정 (로컬 전용)
- 위치: 브라우저의 `localStorage`
- 웹에서 "초기 화면으로 설정"으로 설정한 모델
- 해당 PC 환경에서만 적용
- Git에 커밋되지 않음

## 사용 방법

### 로컬 예시 추가하기 (커밋 가능)

1. `localSamples.json` 파일을 편집합니다.
2. 다음 형식으로 예시를 추가합니다:

```json
[
  {
    "name": "Example: OLS Model",
    "modules": [
      {
        "type": "LoadData",
        "position": { "x": 100, "y": 100 },
        "name": "Load Data"
      },
      {
        "type": "OLSModel",
        "position": { "x": 100, "y": 250 },
        "name": "OLS Model"
      },
      {
        "type": "ResultModel",
        "position": { "x": 100, "y": 400 },
        "name": "Result Model"
      }
    ],
    "connections": [
      {
        "fromModuleIndex": 0,
        "fromPort": "data_out",
        "toModuleIndex": 2,
        "toPort": "data_in"
      },
      {
        "fromModuleIndex": 1,
        "fromPort": "model_out",
        "toModuleIndex": 2,
        "toPort": "model_in"
      }
    ]
  }
]
```

3. Git으로 커밋하고 푸시합니다:
```bash
git add localSamples.json
git commit -m "feat: Add new local sample example"
git push
```

### 웹에서 모델 저장하기 (로컬 전용)

1. Samples 메뉴에서 "현재 모델 저장 (로컬 전용)" 클릭
2. 모델 이름 입력
3. 브라우저의 localStorage에 저장됨 (해당 PC에서만 사용 가능)

### 초기 화면 설정하기 (로컬 전용)

1. 먼저 모델을 저장합니다 (위의 "웹에서 모델 저장하기" 참조)
2. Samples 메뉴에서 "초기 화면으로 설정 (로컬 전용)" 클릭
3. 앱을 다시 열면 자동으로 해당 모델이 로드됩니다

## Samples 메뉴 구조

Samples 메뉴는 다음과 같이 구성됩니다:

1. **현재 모델 저장 (로컬 전용)** - 웹에서 현재 모델을 localStorage에 저장
2. **초기 화면으로 설정 (로컬 전용)** - 저장된 모델을 초기 화면으로 설정
3. **구분선**
4. **저장된 모델 (로컬 전용)** - localStorage에 저장된 모델 목록
5. **구분선**
6. **로컬 예시 (커밋 가능)** - `localSamples.json`의 예시 목록
7. **구분선**
8. **기본 예시** - `constants.ts`의 기본 예시 목록

## 주의사항

- `localSamples.json`은 선택적 파일입니다. 파일이 없어도 앱은 정상 작동합니다.
- localStorage에 저장된 모델은 브라우저를 삭제하면 사라집니다.
- 로컬 예시를 추가한 후에는 반드시 Git으로 커밋하고 푸시하여 다른 사용자와 공유하세요.

