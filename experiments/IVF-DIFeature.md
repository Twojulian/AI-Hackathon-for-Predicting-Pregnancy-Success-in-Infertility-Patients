# 🏥 난임 시술 임신 성공 예측 프로젝트

난임 시술 환자의 임상 데이터를 기반으로 **임신 성공 여부를 예측**하는 이진 분류 머신러닝 프로젝트입니다.  
베이스라인 모델(Random Forest)에 **시술 종류(IVF/DI) 관련 파생 변수**를 추가하여 예측 성능을 개선했습니다.

---

## 📁 파일 구조

```
├── 난임_프로젝트_전처리_추가하는_중.ipynb   # 메인 노트북
├── train.csv                               # 학습 데이터
├── test.csv                                # 테스트 데이터
├── baseline_RFC_v2.csv                     # 제출용 예측 결과
└── feature_importance_v2.png               # 변수 중요도 시각화
```

---

## 🎯 문제 정의

- **태스크**: 이진 분류 (임신 성공: 1 / 실패: 0)
- **평가 지표**: AUC-ROC
- **타겟 변수**: `임신 성공 여부`

---

## 🔧 전처리 파이프라인

### 1. 결측치 처리
- **결측 비율이 높은 컬럼 제거**
  - `착상 전 유전 검사 사용 여부`, `PGD 시술 여부`, `PGS 시술 여부`
  - `난자 해동 경과일`, `배아 해동 경과일`, `임신 시도 또는 마지막 임신 경과 연수`
- 수치형 컬럼 결측 → `0`으로 대체
- 범주형 컬럼 결측 → `"Unknown"`으로 대체

### 2. 횟수형 변수 정제 (`convert_count`)
"N회", "6회 이상" 등 문자열 형태의 값을 정수로 변환

| 원본 값 | 변환 결과 |
|--------|--------|
| `"3회"` | `3` |
| `"6회 이상"` | `6` |
| `NaN` / `"Unknown"` | `0` |

대상 컬럼: `총 시술 횟수`, `IVF 시술 횟수`, `DI 시술 횟수`, `총 임신 횟수` 등 10개

### 3. 나이 변수 수치화

| 변수 | 매핑 방식 |
|------|--------|
| `시술 당시 나이` | 연령 구간 → 대표 나이 (예: 만35-37세 → 36) |
| `난자 기증자 나이` | 연령 구간 → 대표 나이 |
| `정자 기증자 나이` | 연령 구간 → 대표 나이 |

### 4. 범주형 인코딩
- `LabelEncoder` 사용
- Train에서 fit, Test에는 transform만 적용 (unseen label → "Unknown" 처리)

---

## ✨ 파생 변수 생성 (핵심 개선 사항)

IVF(체외수정)와 DI(인공수정) 시술 이력을 기반으로 **7개의 파생 피처**를 추가 생성했습니다.

### 합산 피처
| 파생 변수 | 계산식 |
|---------|------|
| `IVF_DI_시술_합산` | IVF 시술 횟수 + DI 시술 횟수 |
| `IVF_DI_임신_합산` | IVF 임신 횟수 + DI 임신 횟수 |
| `IVF_DI_출산_합산` | IVF 출산 횟수 + DI 출산 횟수 |

### 비율 피처
| 파생 변수 | 계산식 | 의미 |
|---------|------|-----|
| `IVF_시술_비율` | IVF 시술 횟수 / (합산 + ε) | 전체 시술 중 IVF 비중 |
| `IVF_임신_비율` | IVF 임신 횟수 / (합산 + ε) | 전체 임신 중 IVF 비중 |
| `IVF_출산_비율` | IVF 출산 횟수 / (합산 + ε) | 전체 출산 중 IVF 비중 |
| `시술_대비_임신_비율` | 임신 합산 / (시술 합산 + ε) | 과거 시술 대비 임신 성공률 |

> ε = 1e-6 (0 나눗셈 방지)

---

## 🤖 모델

**Random Forest Classifier**

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight="balanced",   # 클래스 불균형 대응
    random_state=42,
    n_jobs=-1
)
```

- 학습/검증 분리: 8:2 (stratified split)
- 클래스 불균형 처리: `class_weight="balanced"`

---

## 📊 검증 성능

| 지표 | 값 |
|------|---|
| AUC-ROC | 노트북 실행 결과 참고 |
| Classification Report | 실패(0) / 성공(1) 각각 precision, recall, f1 출력 |

변수 중요도 시각화 (`feature_importance_v2.png`)에서  
**주황색 bar = 새로 추가된 파생 변수**, 파란색 bar = 기존 변수로 구분됩니다.

---

## 🚀 실행 방법

### 환경 설정

```bash
pip install koreanize-matplotlib
apt-get install -y fonts-nanum
```

### 데이터 경로 설정

노트북 내 데이터 경로를 본인 환경에 맞게 수정하세요.

```python
# Google Colab
train_raw = pd.read_csv("/content/drive/MyDrive/.../train.csv")
test_raw  = pd.read_csv("/content/drive/MyDrive/.../test.csv")

# 로컬 환경
train_raw = pd.read_csv("./train.csv")
test_raw  = pd.read_csv("./test.csv")
```

### 노트북 순서대로 실행

1. 환경 설정 및 라이브러리 임포트
2. 전처리 함수 정의
3. 데이터 로드 및 전처리
4. X/y 분리 및 train/val split
5. Random Forest 모델 학습
6. 검증 성능 출력 및 변수 중요도 시각화
7. 제출 파일 생성 (`baseline_RFC_v2.csv`)

---

## 📦 주요 라이브러리

| 라이브러리 | 용도 |
|-----------|-----|
| `pandas` | 데이터 처리 |
| `numpy` | 수치 연산 |
| `scikit-learn` | 모델 학습 및 평가 |
| `matplotlib` | 시각화 |
| `koreanize-matplotlib` | 한글 폰트 지원 |

---
## 결과 

--- 검증 성능 결과 ---
              precision    recall  f1-score   support

       실패(0)       0.87      0.57      0.69     38025
       성공(1)       0.38      0.76      0.51     13246

    accuracy                           0.62     51271
   macro avg       0.63      0.67      0.60     51271
weighted avg       0.75      0.62      0.65     51271

AUC-ROC: 0.7304

-> 베이스 라인과 0.0003 차이로 미세하게 좋아짐.
하지만 주요 top 10 에서는 보이지 않고, 원래의 변수만 생성

## 📝 향후 개선 방향
- [ ] 가장 안좋은 변수 삭제
- [ ] XGBoost / LightGBM 등 부스팅 계열 모델 적용
- [ ] 하이퍼파라미터 튜닝 (GridSearch / Optuna)
- [ ] 추가 파생 변수 발굴 (예: 나이 × 시술 횟수 교호작용)
- [ ] SHAP을 활용한 모델 해석
