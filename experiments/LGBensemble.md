# 🔀 앙상블 실험 기록 — SYJ

> LightGBM 단독 성능 한계 확인 후 RF+LGB, LGB+XGB 앙상블 실험 기록

---

## 📊 실험 배경

LGB Optuna 튜닝(exp017) 후 Val AUC 0.7372로 기존 베스트(0.7493) 미달.
단일 모델 한계를 극복하기 위해 앙상블 시도.

---

## 🧪 exp018 — RF + LGB 앙상블

### 모델 구성
| 모델 | 파라미터 |
|------|---------|
| RF | n_estimators=300, max_depth=10, min_samples_split=2, min_samples_leaf=9, max_features=0.7, class_weight=balanced |
| LGB | n_estimators=400, learning_rate=0.04956, num_leaves=112, max_depth=4, min_child_samples=26, subsample=0.7524, colsample_bytree=0.6354, reg_alpha=5.6828, reg_lambda=8.9551, scale_pos_weight=2.87 |

### Val AUC 결과
| 버전 | 가중치 | Val AUC |
|------|--------|---------|
| RF 단독 | - | 0.7340 |
| LGB 단독 | - | 0.7372 |
| v1 | RF 0.6 + LGB 0.4 | 0.7363 |
| v2 | RF 0.4 + LGB 0.6 | 0.7369 |
| v3 | RF 0.5 + LGB 0.5 | 0.7366 |

### 결론
- LGB 단독(0.7372)이 모든 앙상블 버전보다 높음
- RF와 LGB가 유사한 피처에 의존 → 앙상블 효과 미미
- 제출 파일: submission_exp018_v1~v3_SYJ.csv

---

## 🧪 exp019 — LGB + XGB 앙상블

### 모델 구성
| 모델 | 파라미터 |
|------|---------|
| LGB | exp017 best params 동일 |
| XGB | n_estimators=500, learning_rate=0.05, max_depth=6, min_child_weight=1, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=2.87 |

### Val AUC 결과
| 버전 | 가중치 | Val AUC |
|------|--------|---------|
| LGB 단독 | - | 0.7372 |
| XGB 단독 | - | 0.7352 |
| v1 | LGB 0.6 + XGB 0.4 | 0.7370 |
| v2 | LGB 0.5 + XGB 0.5 | 0.7368 |
| v3 | LGB 0.4 + XGB 0.6 | 0.7366 |

### XGB Classification Report
```
              precision  recall  f1-score  support
0              0.87      0.58    0.69      38025
1              0.38      0.76    0.51      13246
accuracy                         0.62      51271
macro avg      0.63      0.67    0.60      51271
weighted avg   0.75      0.62    0.65      51271
```

### XGB Feature Importance Top 15
| 순위 | 피처 | 중요도 |
|------|------|--------|
| 1 | 배아 이식 경과일_결측여부 | 0.200 |
| 2 | 이식된 배아 수 | 0.174 |
| 3 | 시술 유형 | 0.165 |
| 4 | 난자 채취 경과일_결측여부 | 0.057 |
| 5 | 배아 생성 주요 이유 | 0.027 |
| 6 | 난자 출처 | 0.025 |
| 7 | 저장된 배아 수 | 0.024 |
| 8 | 고령_여부 | 0.022 |
| 9 | 배아 이식 경과일 | 0.021 |
| 10 | 시술 당시 나이 | 0.021 |

### 결론
- LGB 단독(0.7372)이 모든 앙상블 버전보다 높음
- LGB vs XGB Feature Importance 패턴 차이 발견
  - LGB: `시술 당시 나이`, `난자_배아_전환율` 상위
  - XGB: `배아 이식 경과일_결측여부`, `이식된 배아 수` 상위
- 제출 파일: submission_exp019_v1~v3_SYJ.csv

---

## 📈 전체 실험 AUC 흐름

```
RF 전체 피처    (exp014): Val 0.7340 | 제출 0.7384
RF Top30       (exp015): Val 0.7316
LGB 베이스라인  (exp016): Val 0.7358
LGB + Optuna   (exp017): Val 0.7372 | CV 0.7398  ← 현재 베스트
RF+LGB 앙상블  (exp018): Val 0.7369 (best v2)
LGB+XGB 앙상블 (exp019): Val 0.7370 (best v1)
기존 베스트 V5         : Val 0.7493
```

---

## 💡 앙상블 실험 인사이트

- 두 실험 모두 단일 LGB 모델보다 낮은 성능
- XGB에서 `배아 이식 경과일_결측여부`가 압도적 1위(0.200) → 파생 피처 추가 가능성
- 앙상블보다 **피처 엔지니어링**이 더 효과적일 것으로 판단

---

## 🔜 다음 실험 방향

- [ ] `배아 이식 경과일` 관련 파생 피처 추가
  - `배아_이식_경과일_구간` (bin 처리)
  - `채취_이식_경과일_차이`
  - `조기_이식_여부`
  - `혼합_이식_경과일_차이`
- [ ] XGB Optuna 튜닝 후 앙상블 재시도
- [ ] LGB Optuna n_trials 100으로 증가
