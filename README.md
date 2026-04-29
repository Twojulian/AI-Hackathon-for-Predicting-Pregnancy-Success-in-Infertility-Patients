# 난임 환자 대상 임신 성공 여부 예측 AI 모델 개발

**데이콘 [초격차] AI 헬스케어 머신러닝 트랙 해커톤**

---

## 대회 개요

난임 환자 데이터를 분석하여 임신 성공 여부를 예측하는 AI 모델을 개발합니다.
임신 성공에 영향을 미치는 주요 요인을 탐색하고, 최적의 예측 모델을 구축하는 것이 목표입니다.

> 임신 성공 정의: 출산까지 성공적으로 진행된 임신

- **주최 / 주관**: 데이콘
- **대회 기간**: 2026년 4월 23일 (목) 13:00 ~ 2026년 4월 29일 (수) 23:59
- **평가 지표**: ROC-AUC (Test 데이터 100% 기준)
- **일 최대 제출**: 3회

---

## 최종 결과

| 제출 | 모델 | OOF AUC | 리더보드 AUC | 
|---|---|---|---|
| 최종 제출 | EXP-040 (30모델 Rank Avg) | 0.74091 | **0.74223** | 

### 핵심 실험 흐름

| 실험 | 내용 | OOF AUC |
|---|---|---|
| EXP-010 | CAT/XGB Optuna 튜닝 | 0.74046 |
| EXP-015 | LGB Optuna 튜닝 | 0.74063 |
| EXP-020 | Interaction Target Encoding (ITE) | 0.74068 |
| EXP-028 | Multi-seed LGB × 5 seeds | 0.74082 |
| EXP-032 | LGB+CAT+XGB 피처 다양성 앙상블 | 0.74071 → **제출 0.74219** |
| EXP-038 | EXP-032 × 5 seeds = 15모델 | 0.74090 |
| EXP-040 | EXP-032 × 10 seeds = 30모델 | 0.74091 → **최종 제출 0.74223** |

### 주요 인사이트

- **피처 다양성** (LGB: FE-v1 / CAT: FE-v2+TE / XGB: FE-v2+TE+ITE)이 단일 모델 튜닝보다 효과적
- **Multi-seed 앙상블**: 1→5 seeds +0.00019 OOF, 5→10 seeds +0.00001 (수확 체감)
- **Target Encoding**: fold 내부 계산으로 data leakage 원천 차단
- **Rank Average**: 단조 변환에 AUC 불변 → 정규화 불필요

---

## 프로젝트 구조

```
.
├── data/
│   ├── raw/                # 원본 데이터 (train.csv, test.csv, sample_submission.csv)
│   ├── submissions/        # 제출 파일 (submission_expNNN_이름_점수.csv)
│   └── checkpoints/        # Optuna DB 등 체크포인트
├── notebooks/
│   ├── 01_eda_yjcho.ipynb              # EDA
│   ├── 02_baseline_yjcho.ipynb         # 베이스라인 (LightGBM)
│   ├── 04_hparam_yjcho.ipynb           # LGB Optuna 튜닝
│   ├── 10_hparam_cat_xgb_yjcho.ipynb   # CAT/XGB Optuna 튜닝
│   ├── 15_hparam_lgb_yjcho.ipynb       # LGB TE 환경 재튜닝
│   ├── 34_multi_seed_exp032_yjcho.ipynb # EXP-038: 5-seed 앙상블
│   ├── 36_multi_seed_10s_yjcho.ipynb   # EXP-040: 10-seed 앙상블 (최종 제출)
│   └── ...                             # EXP-003~042 실험 노트북
├── src/
│   └── preprocessing.py    # 공통 전처리 모듈
├── docs/
│   └── leaderboard.xlsx    # 실험 기록 (EXP-001~042)
├── requirements.txt
└── README.md
```

---

## 환경 설정

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 데이터

| 파일 | 행 수 | 설명 |
|------|------:|------|
| train.csv | 256,351 | 학습 데이터 (피처 67개 + 타겟) |
| test.csv | 90,067 | 예측 대상 |
| sample_submission.csv | 90,067 | 제출 양식 (`ID`, `probability`) |

- 타겟: `임신 성공 여부` (0/1)
- 클래스 비율: 성공 25.8% / 실패 74.2% (불균형)

---

## 팀 협업 규칙

### 브랜치 전략

```
main          ← 최종 제출본만 머지
├── feature/이름-작업명   ← 개인 실험 브랜치
└── fix/이름-버그명       ← 버그 수정
```

- 작업은 항상 `feature/이름-작업명` 브랜치에서 시작
- `main` 브랜치에 머지로 PR → 팀원 1명 이상 리뷰 후 머지
- `main` 직접 push 금지

### 노트북 네이밍

```
{번호}_{설명}_{이름이니셜}.ipynb
예) 03_feature_engineering_yjcho.ipynb
```

### 실험 기록

- 실험할 때마다 `docs/leaderboard.xlsx` 에 기록
- 번호는 순서대로 (EXP-001, EXP-002, ...)

### 제출 파일 네이밍

```
submission_{exp번호}_{이름}_{AUC점수}.csv
예) submission_exp040_조여진_07409.csv
```

---

## 유의 사항

- 외부 데이터 사용 불가 / 사전 학습 모델(Pre-trained) 사용 가능
- 인코딩·스케일링·결측치 처리 시 **test 데이터의 통계값 활용 금지** (Data Leakage)
- Target Encoding은 반드시 **fold 내부**에서 train fold 기준으로만 계산


## 팀원 실험 흐름 

### 설윤재 핵심 실험 흐름
| 단계 | 실험 | 핵심 변경 | 모델 | 피처 수 | Val/OOF AUC | 제출 AUC | 변화 |
|------|------|-----------|------|---------|-------------|----------|------|
| EDA | — | 클래스 불균형·결측 패턴·파생변수 근거 도출 | — | 67 | — | — | — |
| 베이스라인 | base RFC | 고결측 컬럼 제거, RF 기본 학습 | RF | 전체 | 0.7301 | 0.73303 | — |
| 피처 엔지니어링 | V5 | 파생변수 12개, log1p, 피처 N 스윕 | RF | Top 30 | 0.7493 ⚠ | 0.7384 | 과적합 |
| 모델 전환 | exp017 | LGB 전환 + Optuna 50 trials, 5-fold | LGB | 81 | 0.7372 | — | +0.0071 |
| 앙상블 | **exp021** ⭐ | LGB+CAT+XGB 3모델, Optuna 가중치 200 trials | LGB·CAT·XGB | 85 | 0.7374 | **0.7413** | +0.0029 |
| 검증 개선 | exp022 | 파생 피처 +14개, Hold-out → OOF 5-fold 전환 | LGB·CAT·XGB | 99 | 0.73990 | — | +0.0025 |
| 전처리 개선 | exp023 | DI 구조적 결측 처리, is_DI 추가 | LGB·CAT·XGB | 102 | 0.73997 | — | +0.0001 |
| 하이퍼파라미터 | **exp024** ⭐ | 3모델 전부 Optuna 재튜닝 (102 피처) | LGB·CAT·XGB | 102 | **0.74025** | **0.74169** | +0.0003 |
| 피처 선별 | exp027 | importance 하위 제거 (113 → 81개) | LGB·CAT·XGB | 81 | 0.74021 | — | -0.0001 |
| 앙상블 구조 | exp029 | Rank Average → Ridge Stacking 메타 모델 | LGB·CAT·XGB | 102 | 0.74024 | — | -0.0001 |
| 인코딩 | **exp032** ⭐ | fold-wise TE + ITE(조합 TE) + 팀원 파라미터 결합 | LGB·CAT·XGB | 혼합 | **0.74037** | **0.74170** | +0.0001 |
