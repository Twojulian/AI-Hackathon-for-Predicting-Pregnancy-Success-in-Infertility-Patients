# 🏥 난임 시술 임신 성공 예측 프로젝트 (V5 + Optuna RF Top30)

난임 시술 환자의 임상 데이터를 기반으로 **임신 성공 여부를 예측**하는 이진 분류 머신러닝 프로젝트입니다.  
V5 전처리 파이프라인 위에 **Optuna 하이퍼파라미터 최적화**와 **Top 30 피처 고정**을 적용해 성능을 개선했습니다.

---

## 📁 파일 구조

```
├── 난임_프로젝트_V5_Optuna.ipynb       # 메인 노트북
├── train.csv                           # 학습 데이터
├── test.csv                            # 테스트 데이터
├── optuna_rf_top30.csv                 # 최종 제출 파일
├── optuna_rf.db                        # Optuna 탐색 기록 (SQLite)
└── feature_importance_top30.png        # Top 30 피처 중요도 시각화
```

---

## 🎯 문제 정의

- **태스크**: 이진 분류 (임신 성공: 1 / 실패: 0)
- **평가 지표**: AUC-ROC
- **타겟 변수**: `임신 성공 여부`

---

## 🔧 전처리 파이프라인 (V5 동일)

### Step 1. 고결측 컬럼 제거

| 컬럼 | 결측률 |
|------|--------|
| 착상 전 유전 검사 사용 여부 | 90.9% |
| PGD 시술 여부 | 99.2% |
| PGS 시술 여부 | 99.2% |
| 난자 해동 경과일 | 99.4% |
| 임신 시도 또는 마지막 임신 경과 연수 | 96.3% |
| 배아 해동 경과일 | 84.2% |

### Step 2. 경과일 변수 — 결측 플래그 생성 후 중앙값 대체

결측 패턴별 임신 성공률이 2%~39%로 극단적 차이를 보여, 결측 여부 자체를 피처로 활용합니다.  
대상: `난자 채취 경과일`, `난자 혼합 경과일`, `배아 이식 경과일`

### Step 3~4. 결측치 처리

- 수치형 → `0` 대체
- 범주형 → `"Unknown"` 대체

### Step 5. 횟수형 변수 정제

"N회", "6회 이상" 형태 문자열을 정수로 변환합니다.

### Step 6. 나이 변수 수치화

연령 구간을 대표 나이로 변환합니다. (예: 만35-37세 → 36)

### Step 7. log1p 변환

우편향(skewness > 0.75)인 배아 관련 수치 변수 8개에 로그 변환을 적용합니다.

---

## ✨ 파생 변수 (EDA 기반)

| 파생 변수 | 계산식 | 의미 |
|---------|--------|------|
| `IVF_DI_시술_합산` | IVF 시술 + DI 시술 | 전체 시술 경험 총량 |
| `IVF_DI_임신_합산` | IVF 임신 + DI 임신 | 전체 임신 경험 총량 |
| `IVF_DI_출산_합산` | IVF 출산 + DI 출산 | 전체 출산 경험 총량 |
| `IVF_시술_비율` | IVF 시술 / (합산 + ε) | 전체 시술 중 IVF 비중 |
| `IVF_임신_비율` | IVF 임신 / (합산 + ε) | 전체 임신 중 IVF 비중 |
| `IVF_출산_비율` | IVF 출산 / (합산 + ε) | 전체 출산 중 IVF 비중 |
| `시술_대비_임신_비율` | 임신 합산 / (시술 합산 + ε) | 과거 시술 대비 임신 성공률 |
| `불임_원인_개수` | 10개 원인 컬럼 합산 | 복합 원인 유무 반영 |
| `배아_사용_조합` | 동결+신선+기증 여부 문자열 결합 | 시술 프로세스 유형 구분 |
| `*_결측여부` (×3) | 경과일 컬럼별 결측 플래그 | 결측 패턴 정보 보존 |

---

## 🤖 모델 및 최적화

### 피처 선택

전체 피처로 RF를 1회 학습한 뒤, 중요도 기준 **상위 30개 피처**를 고정해 이후 탐색에 활용합니다.

**Top 30 피처 목록**

| 순위 | 피처명 | 순위 | 피처명 |
|------|--------|------|--------|
| 1 | 이식된 배아 수 | 16 | 배아_사용_조합 |
| 2 | 배아 이식 경과일_결측여부 | 17 | 미세주입된 난자 수 |
| 3 | 배아 이식 경과일 | 18 | 난자 채취 경과일_결측여부 |
| 4 | 총 생성 배아 수 | 19 | 시술 시기 코드 |
| 5 | 시술 당시 나이 | 20 | 해동된 배아 수 |
| 6 | 저장된 배아 수 | 21 | 난자 출처 |
| 7 | 혼합된 난자 수 | 22 | 난자 혼합 경과일_결측여부 |
| 8 | 미세주입 배아 이식 수 | 23 | 난자 기증자 나이 |
| 9 | 배아 생성 주요 이유 | 24 | 배란 유도 유형 |
| 10 | 수집된 신선 난자 수 | 25 | 동결 배아 사용 여부 |
| 11 | 파트너 정자와 혼합된 난자 수 | 26 | IVF 시술 횟수 |
| 12 | 미세주입에서 생성된 배아 수 | 27 | IVF_시술_비율 |
| 13 | 특정 시술 유형 | 28 | 정자 기증자 나이 |
| 14 | 미세주입 후 저장된 배아 수 | 29 | 시술_대비_임신_비율 |
| 15 | 단일 배아 이식 여부 | 30 | 클리닉 내 총 시술 횟수 |

### Optuna 하이퍼파라미터 탐색

```python
RandomForestClassifier(
    n_estimators = trial.suggest_int("n_estimators", 100, 500, step=50),
    max_depth    = trial.suggest_int("max_depth", 5, 25),
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20),
    min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 10),
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7]),
    class_weight = trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
)
```

| 항목 | 설정 |
|------|------|
| Sampler | TPE (Tree-structured Parzen Estimator) |
| Pruner | MedianPruner (n_warmup_steps=10) |
| Trials | 50 |
| 검증 방법 | StratifiedKFold (5-fold) |
| 탐색 시간 | 약 2시간 32분 |

### 최적 파라미터 (Best Params)

| 파라미터 | 값 |
|----------|----|
| n_estimators | 300 |
| max_depth | 10 |
| min_samples_split | 2 |
| min_samples_leaf | 9 |
| max_features | 0.7 |
| class_weight | balanced |

---

## 📊 실험 결과 비교

| 실험 | 모델 | 피처 수 | Optuna | 클래스 불균형 처리 | CV AUC | Val AUC |
|------|------|---------|--------|-------------------|--------|---------|
| Top 30 RF | RandomForest | 30 | ❌ | balanced | - | 0.7317 |
| 베이스라인 Optuna | RandomForest | 전체 | ✅ (50 trials) | balanced | - | 0.7344 |
| **Top 30 RF + Optuna** | **RandomForest** | **30** | **✅ (50 trials)** | **balanced** | **0.7366** | **0.7493** |

> Top 30 피처로 범위를 좁힌 상태에서 Optuna 탐색을 결합했을 때 가장 높은 Val AUC를 기록했습니다.  
> 피처 수 축소만으로는 +0.0027, Optuna만으로는 +0.0027 개선됐으나,  
> 두 전략을 함께 적용했을 때 Val AUC 기준 **+0.0176** 향상이 확인되었습니다.

---

## 🚀 실행 방법

### 환경 설정

```bash
pip install koreanize-matplotlib optuna xgboost
apt-get install -y fonts-nanum
```

### 데이터 경로 수정

```python
# Kaggle
train = pd.read_csv('/kaggle/input/.../train.csv')
test  = pd.read_csv('/kaggle/input/.../test.csv')

# 로컬 환경
train = pd.read_csv('./train.csv')
test  = pd.read_csv('./test.csv')
```

### 노트북 실행 순서

1. 환경 설정 및 라이브러리 임포트
2. 전처리 함수 정의 (V5 공통 설정값 포함)
3. 데이터 로드 및 전처리 → 파생 변수 생성
4. X/y 분리 및 train/val split (8:2, stratified)
5. RF로 Top 30 피처 선택
6. Optuna 탐색 (50 trials, StratifiedKFold 5-fold)
7. 최적 파라미터로 최종 모델 학습 → 제출 파일 생성

---

## 📦 주요 라이브러리

| 라이브러리 | 용도 |
|-----------|------|
| `pandas` / `numpy` | 데이터 처리 및 수치 연산 |
| `scikit-learn` | 모델 학습, 평가, 전처리 |
| `optuna` | 하이퍼파라미터 자동 최적화 |
| `matplotlib` | 시각화 |
| `koreanize-matplotlib` | 한글 폰트 지원 |

---

## 🔄 V5 대비 변경점

| 항목 | V5 | V5 + Optuna RF Top30 |
|------|----|-----------------------|
| 피처 수 | 전체 | Top 30 고정 |
| 하이퍼파라미터 | 수동 설정 | Optuna 50 trials 자동 탐색 |
| 검증 방식 | Hold-out 20% | 5-fold CV + Hold-out 20% |
| Val AUC | - | 0.7493 |

---

## 📝 향후 개선 방향

- [ ] XGBoost / LightGBM 등 부스팅 계열 모델 적용
- [ ] RF + XGB + ExtraTrees 앙상블
- [ ] Optuna trials 수 증가 (100+)
- [ ] SHAP을 활용한 예측 근거 시각화
- [ ] 나이 × 시술 횟수 교호작용 피처 추가  -> 5위 30위로 이미 들어가 있는데 만들어서 할지 의견 물어보자.
