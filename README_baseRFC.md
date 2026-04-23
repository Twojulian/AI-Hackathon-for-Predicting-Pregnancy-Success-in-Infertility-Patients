# 임신 성공 여부 예측 모델

> 불임 시술 데이터를 기반으로 임신 성공 확률을 예측하는 이진 분류 모델

---

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 대회 | 헬스케어 해커톤 |
| 목표 | 불임 시술 환자 데이터로 임신 성공 확률 예측 |
| 데이터 | train.csv (256,351행 × 69컬럼), test.csv |
| 타겟 | `probability` (임신 성공 확률 0.0 ~ 1.0) |
| 평가 지표 | AUC-ROC, F1, Precision, Recall |

---

## 클래스 불균형

```
임신 실패 (0): 190,123건 (74.2%)
임신 성공 (1):  66,228건 (25.8%)
```

→ 약 **3:1 불균형** 데이터로, 단순 정확도(Accuracy)가 아닌 AUC-ROC와 F1을 주요 지표로 사용

---

## 파일 구조

```
├── train.csv              # 학습 데이터
├── test.csv               # 예측 대상 데이터
├── sample_submission.csv  # 제출 양식 (ID, probability)
├── preprocessing.py       # 전처리 + 모델 학습 + 제출 파일 생성
├── submission.csv         # 최종 제출 파일 (자동 생성)
└── feature_importance.png # 변수 중요도 시각화 (자동 생성)
```

---

## 전처리 전략

### 1. 고결측 컬럼 제거 (전략 A)
결측치가 전체의 99% 이상인 컬럼은 정보가 없다고 판단하여 제거

| 제거 컬럼 | 결측 비율 |
|-----------|-----------|
| 착상 전 유전 검사 사용 여부 | 99.0% |
| PGD 시술 여부 | 99.1% |
| PGS 시술 여부 | 99.2% |
| 난자 해동 경과일 | 99.4% |
| 임신 시도 또는 마지막 임신 경과 연수 | 96.3% |
| 배아 해동 경과일 | 84.3% |

### 2. 수치형 결측 → 0 (전략 B)
배아 수, 난자 수 등 시술 관련 수치 컬럼의 결측은 "해당 시술을 받지 않음"을 의미하므로 0으로 채움

### 3. 범주형 결측 → "Unknown" (전략 C)
범주형 컬럼의 결측은 별도의 카테고리 `"Unknown"`으로 처리

---

## 인코딩 전략

### 횟수형 컬럼 수치 변환
`"3회"`, `"6회 이상"` 형태의 문자열을 정수로 변환

```
"0회" → 0, "1회" → 1, ..., "6회 이상" → 6
```

대상 컬럼: 총 시술 횟수, IVF/DI 시술·임신·출산 횟수 등 10개

### 나이 구간 → 중앙값 변환

| 원본 값 | 변환값 |
|---------|--------|
| 만18-34세 | 26 |
| 만35-37세 | 36 |
| 만38-39세 | 38 |
| 만40-42세 | 41 |
| 만43-44세 | 43 |
| 만45-50세 | 47 |

### Label Encoding
나머지 범주형 컬럼은 Label Encoding 적용
- **중요**: train에서 `fit`한 인코더를 저장하여 test에는 `transform`만 적용
- test에 train에 없던 값이 등장하면 `"Unknown"`으로 대체하여 에러 방지

---

## 모델

### RandomForestClassifier

```python
RandomForestClassifier(
    n_estimators=100,   # 결정 트리 100개
    max_depth=10,       # 과적합 방지를 위한 깊이 제한
    class_weight="balanced",  # 클래스 불균형 자동 보정
    random_state=42,
    n_jobs=-1           # 전체 CPU 코어 사용
)
```

### 불균형 대응 방법

| 방법 | 적용 위치 | 설명 |
|------|-----------|------|
| `stratify=y` | train_test_split | train/val의 클래스 비율을 동일하게 유지 |
| `class_weight="balanced"` | RandomForestClassifier | 소수 클래스(성공)에 더 높은 가중치 부여 |

---

## 실행 방법

### 환경 설정
```bash
pip install pandas numpy scikit-learn matplotlib
```

### 실행
```bash
python preprocessing.py
```

### 코랩에서 실행
```python
# 경로를 본인 드라이브 경로로 수정 후 실행
train_raw = pd.read_csv("/content/drive/MyDrive/.../train.csv")
test_raw  = pd.read_csv("/content/drive/MyDrive/.../test.csv")
```

---

## 출력 결과

실행 후 아래 파일이 자동 생성됩니다.

- `submission.csv` : 제출용 파일 (ID, probability)
- `feature_importance.png` : 상위 10개 중요 변수 시각화

### 제출 파일 형식
```
           ID  probability
0  TEST_00000     0.049606
1  TEST_00001     0.034206
2  TEST_00002     0.365099
3  TEST_00003     0.290601
4  TEST_00004     0.686711
총 90067개 예측 완료
```

---

## 성능 평가 지표

```
--- 검증 성능 결과 ---
              precision    recall  f1-score   support

       실패(0)       0.87      0.57      0.69     38025
       성공(1)       0.38      0.76      0.51     13246

    accuracy                           0.62     51271
   macro avg       0.63      0.67      0.60     51271
weighted avg       0.75      0.62      0.65     51271

AUC-ROC: 0.7301
```

> 클래스 불균형이 있으므로 **성공(1) 클래스의 Recall과 F1**, 그리고 **AUC-ROC**를 중점적으로 확인

### baseline으로 적합한지 확인했는데 LightGBM 이 더 높게 나옴.
