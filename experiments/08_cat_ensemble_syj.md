# 🧬 IVF 임신 성공 예측 모델 실험 기록 — SYJ

> 불임 시술 데이터 기반 임신 성공 여부 예측 해커톤 25일 기록

---

## 📚 참고 논문

### 1. Shingshetty et al. (2024)
**"Predictors of success after in vitro fertilization"**
*Fertility and Sterility, 121, 742–751*

- 이전 임신/출산 이력이 성공률에 미치는 영향
- 시술 횟수와 임신 성공률 관계
- 나이(38세 이상)에 따른 성공률 감소 근거

### 2. Alson et al. (2026)
**"Machine learning prediction of live birth after IVF using the morphological uterus sonographic assessment group features of adenomyosis"**
*Scientific Reports, 16:4324*

- Optuna 기반 하이퍼파라미터 최적화 방법론
- Stratified 5-fold cross-validation 구조
- SHAP 기반 피처 중요도 해석
- XGBoost 모델 AUC 0.66~0.75 참고

---

## 🛠️ 피처 엔지니어링

| 피처명 | 설명 | 근거 |
|--------|------|------|
| `이전_임신_여부` | 총 임신 횟수 > 0 | Shingshetty 2024 |
| `이전_출산_여부` | 총 출산 횟수 > 0 | Shingshetty 2024 |
| `전체_임신_성공률` | 총 임신 / 총 시술 | Shingshetty 2024 |
| `고령_여부` | 시술 나이 ≥ 38세 | Shingshetty 2024 |
| `첫_시술_여부` | 총 시술 횟수 == 1 | Shingshetty 2024 |
| `난자_배아_전환율` | 총 생성 배아 / 수집 난자 | Alson 2026 |
| `IVF_DI_시술_합산` | IVF + DI 시술 합산 | 도메인 지식 |
| `시술_대비_임신_비율` | 총 임신 / 총 시술 | 도메인 지식 |
| `불임_원인_개수` | 불임 원인 플래그 합산 | 도메인 지식 |
| `배아_사용_조합` | 동결/신선/기증 배아 조합 | 도메인 지식 |
| `임신시도_결측여부` | 임신 시도 경과 연수 결측 | 결측치 처리 |
| `배아_이식_경과일_구간` | 이식 경과일 bin 처리 | XGB Feature Importance |
| `조기_이식_여부` | 배아 이식 경과일 ≤ 3 | XGB Feature Importance |
| `채취_이식_경과일_차이` | 이식 - 채취 경과일 | XGB Feature Importance |
| `혼합_이식_경과일_차이` | 이식 - 혼합 경과일 | XGB Feature Importance |

---

## 🔬 실험 기록

### Phase 1 — RandomForest 베이스라인

#### exp014 — RF 전체 피처
- **모델**: RandomForest
- **파라미터**: n_estimators=300, max_depth=10, min_samples_split=2, min_samples_leaf=9, max_features=0.7, class_weight=balanced
- **피처 수**: 81개
- **Val AUC**: 0.7340
- **제출 AUC**: 0.7384
- **문제점**: `이식된 배아 수` 단일 피처 중요도 0.389로 과도한 의존

#### exp015 — RF Top 30 피처
- **모델**: RandomForest (동일 파라미터)
- **피처 수**: 30개
- **Val AUC**: 0.7316
- **문제점**: 전체 피처 대비 성능 하락

#### 기존 베스트 V5 — RF Top 30
- **Val AUC**: 0.7493
- **제출 AUC**: 0.7384
- **분석**: Val-제출 gap 0.0109 → **과적합** 확인
- **원인**: RF Top 30 선택 시 val 데이터 정보 누수

---

### Phase 2 — LightGBM 전환

#### RF → LGB 전환 이유
- `이식된 배아 수` 단일 피처 의존도 해소 필요
- Gradient Boosting으로 다양한 피처 활용
- `scale_pos_weight`로 클래스 불균형 직접 보정
- 클래스 비율: 실패 38,025건 vs 성공 13,246건 (약 3:1)

#### exp016 — LGB 베이스라인
- **모델**: LGBMClassifier
- **파라미터**: n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=-1, scale_pos_weight=2.87
- **피처 수**: 81개
- **Val AUC**: 0.7358

#### exp017 — LGB + Optuna 튜닝
- **Optuna**: n_trials=50, StratifiedKFold 5-fold
- **Best CV AUC**: 0.7398
- **Val AUC**: 0.7372
- **Best Params**:
```python
{
    "n_estimators":      400,
    "learning_rate":     0.04956,
    "num_leaves":        112,
    "max_depth":         4,
    "min_child_samples": 26,
    "subsample":         0.7524,
    "colsample_bytree":  0.6354,
    "reg_alpha":         5.6828,
    "reg_lambda":        8.9551,
}
```
- **변화**: RF 대비 피처 중요도 분산 (이식된 배아 수 의존도 해소)

---

### Phase 3 — 앙상블 실험

#### exp018 — RF + LGB 앙상블
| 버전 | 가중치 | Val AUC |
|------|--------|---------|
| v1 | RF 0.6 + LGB 0.4 | 0.7363 |
| v2 | RF 0.4 + LGB 0.6 | 0.7369 |
| v3 | RF 0.5 + LGB 0.5 | 0.7366 |

- **결론**: LGB 단독(0.7372)보다 낮음 → RF+LGB 유사 피처 의존으로 앙상블 효과 미미

#### exp019 — LGB + XGB 앙상블
| 버전 | 가중치 | Val AUC |
|------|--------|---------|
| v1 | LGB 0.6 + XGB 0.4 | 0.7370 |
| v2 | LGB 0.5 + XGB 0.5 | 0.7368 |
| v3 | LGB 0.4 + XGB 0.6 | 0.7366 |

- **XGB Val AUC**: 0.7352
- **결론**: LGB 단독보다 낮음. 단, LGB vs XGB Feature Importance 패턴 차이 발견
  - LGB: `시술 당시 나이`, `난자_배아_전환율` 상위
  - XGB: `배아 이식 경과일_결측여부`(0.200), `이식된 배아 수`(0.174) 상위

---

### Phase 4 — 피처 추가

#### exp020 — LGB + 배아 이식 경과일 파생 피처
- **추가 피처**: `배아_이식_경과일_구간`, `조기_이식_여부`, `채취_이식_경과일_차이`, `혼합_이식_경과일_차이`
- **피처 수**: 81 → 85개
- **Val AUC**: 0.7372 (변화 없음)
- **특이사항**: `혼합_이식_경과일_차이` Top 15 진입 (11위)

---

### Phase 5 — 3모델 앙상블 (최종)

#### exp021 — LGB + CAT + XGB Optuna 가중치 앙상블 ⭐
- **모델**: LGBMClassifier + CatBoostClassifier + XGBClassifier
- **Optuna**: n_trials=200 (가중치 최적화)
- **피처 수**: 85개
- **클래스 불균형 처리**: scale_pos_weight=2.8707 / is_unbalance(CAT)

**개별 모델 Val AUC**
| 모델 | Val AUC |
|------|---------|
| LGB | 0.7372 |
| CAT | 0.7371 |
| XGB | 0.7351 |

**최적 가중치**: LGB=0.522, CAT=0.455, XGB=0.023

**결과**
| 지표 | 값 |
|------|-----|
| Val AUC | 0.7374 |
| **제출 AUC** | **0.7412775224** |
| F1 Score | 0.6021 |
| Recall | 0.6722 |
| Precision | 0.6323 |
| Accuracy | 0.6213 |

- **파일명**: submission_exp021_SYJ.csv
- **인사이트**: Val(0.7374) < 제출(0.7413) → 과적합 없음, 일반화 성공

---

## 📈 전체 실험 AUC 흐름

```
exp014 RF 전체         Val 0.7340 → 제출 0.7384
exp015 RF Top30        Val 0.7316
V5     RF Top30        Val 0.7493 → 제출 0.7384 (과적합)
exp016 LGB 베이스       Val 0.7358
exp017 LGB Optuna      Val 0.7372 | CV 0.7398
exp018 RF+LGB 앙상블    Val 0.7369 (best)
exp019 LGB+XGB 앙상블   Val 0.7370 (best)
exp020 LGB+파생피처     Val 0.7372
exp021 LGB+CAT+XGB     Val 0.7374 → 제출 0.7413 ⭐ 최종 베스트
```

---

## 💡 핵심 인사이트

1. **Val AUC가 높다고 제출 점수가 높지 않음** — V5가 Val 0.7493이었지만 제출은 0.7384 (과적합)
2. **LGB로 전환 후 피처 의존도 분산** — RF에서 `이식된 배아 수` 0.389 독주 → LGB에서 고른 분포
3. **앙상블은 모델 다양성이 핵심** — RF+LGB, LGB+XGB는 유사 피처 의존으로 효과 미미
4. **CatBoost 추가로 다양성 확보** — LGB+CAT+XGB 조합에서 Val < 제출 달성
5. **팀원과 가중치 패턴 반대** — 여진님은 XGB 강세(0.518), 나는 LGB 강세(0.522) → 전처리 차이

---

## ⚙️ 환경

```
Python 3.x
lightgbm
catboost
xgboost
scikit-learn
pandas / numpy
matplotlib
optuna
Kaggle Notebook (No Accelerator)
```
