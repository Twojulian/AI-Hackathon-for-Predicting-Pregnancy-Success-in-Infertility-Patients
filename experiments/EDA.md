# 📊 EDA (Exploratory Data Analysis)

## 1. 데이터 개요

| 항목 | 내용 |
|------|------|
| 전체 샘플 수 | 256,351개 |
| 피처 수 | 67개 (ID·타겟 제외) |
| 수치형 변수 | 47개 |
| 범주형 변수 | 20개 |
| 타겟 변수 | `임신 성공 여부` (0: 실패, 1: 성공) |

---

## 2. 타겟 변수 분포

<img width="1660" height="742" alt="eda_01_target_distribution" src="https://github.com/user-attachments/assets/a09537d6-9b45-42d2-9583-8e497e0d204d" />


| 클래스 | 샘플 수 | 비율 |
|--------|---------|------|
| 0 (실패) | 190,123 | 74.2% |
| 1 (성공) | 66,228 | 25.8% |

> ⚠️ **클래스 불균형**: 실패:성공 = 약 2.9:1 비율로 불균형이 존재함.  
> → 모델 학습 시 `class_weight`, `stratified split`, `오버샘플링(SMOTE)` 등 고려 필요.

---

## 3. 결측치 분석

<img width="1489" height="1843" alt="eda_02_missing_values" src="https://github.com/user-attachments/assets/25ad055d-b4ca-40ff-bc58-4df5cfeaed3c" />


### 결측률 구간별 분류

| 구간 | 컬럼 | 처리 전략 |
|------|------|-----------|
| 결측률 90% 이상 | `난자 해동 경과일`, `PGS 시술 여부`, `PGD 시술 여부`, `착상 전 유전 검사 사용 여부` 등 | 컬럼 제거 또는 결측 여부 플래그 변수화 |
| 결측률 20~90% | `배아 해동 경과일`, `난자 채취 경과일` 등 | 결측 여부 플래그 추가 후 대체값 처리 |
| 결측률 20% 미만 | 배아 관련 수치 변수 등 | 중앙값/최빈값 대체 |

---

## 4. 수치형 변수 분포

<img width="2685" height="1773" alt="eda_03_numeric_distribution" src="https://github.com/user-attachments/assets/32f74f5d-281e-440c-a8dc-eb58f467b151" />


- 대부분의 배아 관련 수치 변수(`총 생성 배아 수`, `이식된 배아 수` 등)
- 경과일 관련 변수들은 특정 값에 집중된 분포를 보임
- → 로그 변환(`log1p`) 또는 Robust Scaling 고려

---

## 5. 범주형 변수 분포

<img width="2685" height="1476" alt="eda_04_categorical_distribution" src="https://github.com/user-attachments/assets/812f4e8d-416a-49f3-bfe9-8cc64c0cefd5" />


| 변수 | 주요 관찰 |
|------|-----------|
| 시술 당시 나이 | 만18-34세 비중 가장 높음 |
| 시술 유형 | IVF, DI 등 다수 유형 혼재 |
| 난자/정자 출처 | 자가 vs 기증 비율 불균형 |

---

## 6. 타겟과 변수 관계

<img width="2084" height="889" alt="eda_05_target_vs_features" src="https://github.com/user-attachments/assets/6221ac2f-c892-4083-a5ab-af1f1dcaef6d" /><img width="2601" height="890" alt="eda_06_correlation" src="https://github.com/user-attachments/assets/d098450e-79a8-4672-b6f0-8a8ca280ea6b" />


- **나이**: 나이가 증가할수록 임신 성공률 감소 경향 (만18-34세 > 만35-37세 > … > 만45-50세)
- **시술 유형**: IVF 시술이 DI 시술보다 임신 성공률이 약 2배 높음 (IVF ~26%, DI ~13%)

---

## 7. 수치형 변수 상관관계

<img width="2601" height="890" alt="eda_06_correlation" src="https://github.com/user-attachments/assets/deaef93f-5d5d-4709-a8d1-72ce47297633" />


| 구분 | 주요 변수 |
|------|-----------|
| 양의 상관 | `이식된 배아 수`, `미세주입된 난자 수` 등 |
| 음의 상관 | `임신 시도 또는 마지막 임신 경과 연수` 등 |

> 다중공선성 가능성 있는 배아 관련 변수들 간 상관관계 높음 → PCA 또는 변수 선택 고려

---

## 8. 배아 관련 변수 분석

<img width="2684" height="742" alt="eda_07_embryo_analysis" src="https://github.com/user-attachments/assets/e13de49e-d324-4de8-a554-9ea6e1043635" />


- 성공 그룹이 실패 그룹 대비 일부 변수에서 소폭 차이
- 이상치(outlier) 다수 존재 → IQR 기반 클리핑 또는 로버스트 처리 고려

---

## 9. 주요 인사이트 요약

1. **클래스 불균형** (74:26) → 불균형 대응 전략 필수
2. **고결측 컬럼 다수** (90%↑ 결측 4개 컬럼) → 제거 또는 플래그화
3. **나이는 핵심 피처** → 나이 증가 시 성공률 유의미하게 감소
4. **배아 수치 변수군** → 임신 성공과 양의 상관, 다중공선성 주의
5. **수치형 변수 편향** → 로그 변환 또는 이상치 처리 필요

## 추가로 해보면 좋을 항목
+ **불임 원인 조합 분석** 이진 플래그 컬럼 12개가 있는데, 어떤 조합이 성공률에 영향 주는지 분석하면 피처 엔지니어링 근거가 됨
+ **시술 횟수 vs 성공률** 총 시술 횟수가 많을수록 성공률이 어떻게 변하는지 — 경험 효과인지 선택 편향인지 흥미로운 인사이트
+ **경과일 변수 분석** 배아 이식 경과일, 난자 혼합 경과일 등 결측 패턴이 시술 유형과 연관될 수 있어서 결측 이유 분석이 의미 있음
