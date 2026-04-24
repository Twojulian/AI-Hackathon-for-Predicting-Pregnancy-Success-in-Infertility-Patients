# 🏥 불임 시술 임신 성공 여부 예측 프로젝트

## 📁 프로젝트 구조

```
.
├── train.csv                          # 학습 데이터
├── test.csv                           # 제출용 테스트 데이터
│
├── baseline_RFC.py                    # v1: 베이스라인 (결측 컬럼 삭제)
├── compare_RFC.py                     # v2: 결측 처리 방식 A/B 비교
├── optuna_RFC.py                      # v3: Optuna 하이퍼파라미터 튜닝
│
├── feature_importance.png             # 베이스라인 피처 중요도
├── feature_importance_comparison.png  # A/B 비교 피처 중요도
├── optuna_tuning_result.png           # Optuna 튜닝 결과 시각화
│
├── 03_baseRFC_SYJ.csv                 # 베이스라인 제출 파일
└── optuna_RFC_submission.csv          # Optuna 튜닝 후 제출 파일
```

---

## 🎯 목표

환자의 시술 이력, 나이, 임신/출산 횟수 등의 데이터를 기반으로  
**임신 성공 여부(0/1)를 예측**하고, 확률값(`probability`)을 제출.

- **평가 지표**: AUC-ROC

---

## ⚙️ 공통 전처리

### 결측치 처리 대상 컬럼 (`HIGH_NULL_COLS`)

| 컬럼명 | 처리 방식 |
|--------|----------|
| 착상 전 유전 검사 사용 여부 | 컬럼 삭제 |
| PGD 시술 여부 | 컬럼 삭제 |
| PGS 시술 여부 | 컬럼 삭제 |
| 난자 해동 경과일 | 컬럼 삭제 |
| 임신 시도 또는 마지막 임신 경과 연수 | 컬럼 삭제 |
| 배아 해동 경과일 | 컬럼 삭제 |

> 버전 A(삭제) vs 버전 B(살리기) 비교 실험 후 **버전 A가 동등 이상**으로 확인 → 삭제 방식 유지

### 나이 매핑

```python
AGE_MAP = {
    "만18-34세": 26, "만35-37세": 36, "만38-39세": 38,
    "만40-42세": 41, "만43-44세": 43, "만45-50세": 47
}
```

### 시술 횟수 변환 (`COUNT_COLS`)
- `"N회"` → 정수 N
- `"6회 이상"` → 6
- 결측 → 0

### 인코딩
- 범주형 컬럼 전체 `LabelEncoder` 적용
- test 시 학습 때 없던 값은 `"Unknown"`으로 대체 후 인코딩

---

## 📄 버전별 코드 설명

### v1. `baseline_RFC.py` — 베이스라인

가장 기본적인 Random Forest 분류기.

**모델 파라미터**
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

**성능 결과**

| 지표 | 값 |
|------|----|
| AUC-ROC | 0.7301 |
| F1 Score | 0.51 |
| Recall | 0.76 |
| Precision | 0.38 |
| Accuracy | 0.62 |

---

### v2. `compare_RFC.py` — 결측 처리 방식 비교

결측치가 많은 컬럼을 **삭제(A)** vs **살리기(B)** 두 방식으로 비교.

| 버전 | 처리 방식 | AUC-ROC |
|------|----------|---------|
| **A (베이스라인)** | 결측 컬럼 삭제 | **0.7301** |
| B | `-1` 채우기 + 결측 플래그 컬럼 추가 + 범주형 `"미시행"` 처리 | 0.7299 |

**결론**: 차이 `-0.0002`로 사실상 동일 → 버전 A 유지

**버전 B 처리 방식 상세**
```python
# 수치형: -1 채우기 + 결측 플래그
df[f"{col}_결측"] = df[col].isna().astype(int)
df[col] = df[col].fillna(-1)

# 범주형: 명시적 카테고리 처리
df["PGD 시술 여부"] = df["PGD 시술 여부"].fillna("미시행")
```

---

### v3. `optuna_RFC.py` — Optuna 하이퍼파라미터 튜닝

베이지안 최적화(Optuna)로 Random Forest 하이퍼파라미터 자동 탐색.  
속도 최적화를 위해 **Pruning + 3-Fold CV + trials 축소** 적용.

**탐색 파라미터 범위**

| 파라미터 | 탐색 범위 | 설명 |
|----------|----------|------|
| `n_estimators` | 100 ~ 500 (step 100) | 트리 개수 |
| `max_depth` | 5 ~ 20 | 트리 깊이 |
| `min_samples_split` | 2 ~ 20 | 분기 최소 샘플 수 |
| `min_samples_leaf` | 1 ~ 10 | 리프 최소 샘플 수 |
| `max_features` | sqrt / log2 / 0.5 / 0.7 | 피처 샘플링 비율 |
| `class_weight` | balanced / balanced_subsample | 클래스 불균형 처리 |
| `bootstrap` | True / False | 부트스트랩 여부 |

**속도 최적화 전략**

```
기존 (느림)                     수정 (빠름)
─────────────────────────────────────────────
n_trials = 50        →     n_trials = 20
5-Fold CV            →     3-Fold CV
Pruning 없음         →     MedianPruner 적용
                           (5개 기준 수집 후
                            중앙값 이하 trial 조기 종료)
```

**Optuna 설정**
```python
study = optuna.create_study(
    direction="maximize",
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1)
)
study.optimize(objective, n_trials=20, show_progress_bar=True)
```

**출력물**
- `optuna_tuning_result.png`: 최적화 히스토리 + 파라미터 중요도 그래프
- `optuna_RFC_submission.csv`: 최적 파라미터로 예측한 제출 파일

---

## 📊 전체 성능 흐름

```
베이스라인 (v1)        결측 처리 비교 (v2)       Optuna 튜닝 (v3)
AUC 0.7301      →      AUC 0.7299 (B)      →      진행 중
                        ≈ 동일, A 유지
```

---

## 🔧 실행 환경

```bash
pip install pandas numpy scikit-learn matplotlib optuna
```

| 라이브러리 | 버전 |
|-----------|------|
| pandas | 최신 |
| scikit-learn | 최신 |
| optuna | 최신 |
| matplotlib | 최신 |

> 한글 폰트 깨짐 방지: `pip install koreanize-matplotlib` 후 코드 상단에 `import koreanize_matplotlib` 추가

---

## 📌 다음 실험 예정

- [ ] IVF / DI 시술 횟수 합산 & 비율 피처 추가
- [ ] LightGBM / XGBoost 모델 교체
- [ ] 피처 선택 (Feature Selection)
