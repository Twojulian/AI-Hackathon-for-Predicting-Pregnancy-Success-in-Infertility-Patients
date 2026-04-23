# 난임 환자 대상 임신 성공 여부 예측 AI 모델 개발

**데이콘 [초격차] AI 헬스케어 머신러닝 트랙 해커톤**

---

## 대회 개요

난임 환자 데이터를 분석하여 임신 성공 여부를 예측하는 AI 모델을 개발합니다.
임신 성공에 영향을 미치는 주요 요인을 탐색하고, 최적의 예측 모델을 구축하는 것이 목표입니다.

> 임신 성공 정의: 출산까지 성공적으로 진행된 임신

- **주최 / 주관**: 데이콘
- **대회 기간**: 2026년 4월 23일 (목) 13:00 ~ 2026년 5월 4일 (월) 23:59
- **평가 지표**: ROC-AUC (Test 데이터 100% 기준)
- **일 최대 제출**: 3회

---

## 프로젝트 구조

```
.
├── data/
│   ├── raw/                # 원본 데이터 (train.csv, test.csv, sample_submission.csv)
│   └── submissions/        # 제출 파일
├── notebooks/
│   ├── 01_eda.ipynb        # 탐색적 데이터 분석
│   └── 02_baseline.ipynb   # 베이스라인 모델 (LightGBM)
├── src/                    # 모듈화 코드
├── experiments/            # 실험 기록 (expNNN_설명.md)
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
└── fix/이름-버그명    ← 버그 수정
```

- 작업은 항상 `feat/이름-작업명` 브랜치에서 시작
- `main` 브랜치에 머지로 PR → 팀원 1명 이상 리뷰 후 머지
- `main` 직접 push 금지

### 노트북 네이밍

```
{번호}_{설명}_{이름이니셜}.ipynb
예) 03_feature_engineering_yjc.ipynb
```

### 실험 기록

- 실험할 때마다 `experiments/expNNN_설명.md` 파일 생성
- 번호는 순서대로 (exp001, exp002, ...)
- 템플릿: `experiments/_template/experiment.md` 참고

### 제출 파일 네이밍

```
submission_{exp번호}_{이름이니셜}_{AUC점수}.csv
예) submission_exp003_yjc_0.6821.csv
```

---

## 유의 사항

- 외부 데이터 사용 불가 / 사전 학습 모델(Pre-trained) 사용 가능
- 인코딩·스케일링·결측치 처리 시 **test 데이터의 train 활용 금지** (Data Leakage)
- `pd.get_dummies()` 적용 시 test 데이터셋에도 동일 적용 필요
