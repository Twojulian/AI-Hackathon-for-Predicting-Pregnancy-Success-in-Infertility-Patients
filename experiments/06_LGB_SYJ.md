# 🧬 LightGBM 실험 기록 — SYJ

> RandomForest 성능 한계 확인 후 LightGBM으로 모델 전환한 실험 기록

---

## 🔄 모델 전환 배경

### RF 실험 결과 요약
| 실험 | 피처 | Val AUC | 제출 AUC |
|------|------|---------|---------|
| exp014 | 전체 피처 | 0.7340 | 0.7384 |
| exp015 | Top 30 피처 | 0.7316 | - |
| 기존 베스트 (V5) | - | 0.7493 | - |

### RF의 문제점
- `이식된 배아 수` 단일 피처 중요도 **0.389**로 과도한 의존
- Class 1 (임신 성공) precision **0.39**로 낮음
- 클래스 불균형 (실패 38,025건 vs 성공 13,246건, 약 3:1) 대응 한계

### LightGBM 전환 이유
- Gradient Boosting 특성상 잔차 반복 학습 → 다양한 피처 활용
- `scale_pos_weight`로 클래스 불균형 직접 보정
- RF보다 빠른 학습 속도
- 하이퍼파라미터 튜닝 효과가 RF보다 뚜렷함

---

## 🧪 실험 기록

### exp016-base — LightGBM 베이스라인

**파라미터**
```python
{
    "n_estimators":      500,
    "learning_rate":     0.05,
    "num_leaves":        31,
    "max_depth":         -1,
    "min_child_samples": 20,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "scale_pos_weight":  2.87,  # (y==0).sum() / (y==1).sum()
}
```

**결과**
- Val AUC: **0.7358**
- 사용 피처 수: 81개
- RF 전체 피처(0.7340) 대비 소폭 상승

---

### exp017 — LightGBM + Optuna 튜닝

**Optuna 설정**
```python
n_trials = 50
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = "roc_auc"
```

**탐색 범위**
| 파라미터 | 범위 |
|----------|------|
| n_estimators | 100 ~ 500 |
| learning_rate | 0.01 ~ 0.3 (log) |
| num_leaves | 20 ~ 150 |
| max_depth | 3 ~ 12 |
| min_child_samples | 5 ~ 100 |
| subsample | 0.5 ~ 1.0 |
| colsample_bytree | 0.5 ~ 1.0 |
| reg_alpha | 1e-8 ~ 10.0 (log) |
| reg_lambda | 1e-8 ~ 10.0 (log) |

**Best Params**
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
    "scale_pos_weight":  2.87,
}
```

**결과**
- Best CV AUC: **0.7398**
- Val AUC: **0.7372**
- 사용 피처 수: 81개

**Classification Report**
```
              precision  recall  f1-score  support
0              0.88      0.56    0.69      38025
1              0.38      0.78    0.51      13246
accuracy                         0.62      51271
macro avg      0.63      0.67    0.60      51271
weighted avg   0.75      0.62    0.64      51271
```

---

## 📊 Feature Importance 변화 (RF → LGB)

| 순위 | RF (exp014) | LGB Optuna (exp017) |
|------|------------|---------------------|
| 1 | 이식된 배아 수 (0.389) | **시술 당시 나이** (425) |
| 2 | 저장된 배아 수 (0.108) | **난자_배아_전환율** (400) |
| 3 | 배아 이식 경과일_결측여부 (0.101) | 시술 시기 코드 (341) |
| 4 | 배아 이식 경과일 (0.074) | 이식된 배아 수 (309) |
| 5 | 시술 당시 나이 (0.062) | 배아 이식 경과일 (300) |

> LightGBM에서 `이식된 배아 수` 단일 의존도가 해소되고 `시술 당시 나이`, `난자_배아_전환율` 등 다양한 피처가 고르게 활용됨

---

## 📈 전체 실험 AUC 비교

```
RF 전체 피처    (exp014): Val 0.7340 | 제출 0.7384
RF Top30       (exp015): Val 0.7316
LGB 베이스라인  (exp016): Val 0.7358
LGB + Optuna   (exp017): Val 0.7372 | CV 0.7398
기존 베스트 V5         : Val 0.7493
```

---

## 🔜 다음 실험 방향

- [ ] Optuna n_trials 늘리기 (50 → 100)
- [ ] 피처 엔지니어링 추가 (배아 이식 경과일 관련 파생 피처)
- [ ] XGBoost 비교 실험
- [ ] 앙상블 (LGB + RF + XGB)
- [ ] `시술 당시 나이` 구간화 세분화 (현재 6구간)
