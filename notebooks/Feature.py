import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib
import koreanize_matplotlib
import matplotlib.font_manager as fm

import subprocess
subprocess.run(['apt-get', 'install', '-y', 'fonts-nanum'], capture_output=True)
fm._load_fontmanager(try_read_cache=False)
korean_fonts = [f for f in fm.findSystemFonts() if any(k in f for k in ['nanum', 'Nanum', 'gothic', 'Gothic'])]
font_path = korean_fonts[0]
fm.fontManager.addfont(font_path)
matplotlib.rc('font', family='NanumGothic')
matplotlib.rcParams['axes.unicode_minus'] = False


# ════════════════════════════════════════════════════════════
# 공통 설정값
# ════════════════════════════════════════════════════════════

# EDA 결과: 결측률 84~99% → 제거
HIGH_NULL_COLS = [
    "착상 전 유전 검사 사용 여부",   # 결측 90.9%
    "PGD 시술 여부",                # 결측 99.2%
    "PGS 시술 여부",                # 결측 99.2%
    "난자 해동 경과일",              # 결측 99.4%
    "임신 시도 또는 마지막 임신 경과 연수",  # 결측 96.3%
    "배아 해동 경과일",              # 결측 84.2%
]

COUNT_COLS = [
    "총 시술 횟수", "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수", "DI 시술 횟수",
    "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수",
    "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수",
]

AGE_MAP = {
    "만18-34세": 26, "만35-37세": 36, "만38-39세": 38,
    "만40-42세": 41, "만43-44세": 43, "만45-50세": 47, "Unknown": 36
}

DONOR_AGE_MAP = {
    "만20세 이하": 19, "만21-25세": 23, "만26-30세": 28,
    "만31-35세": 33, "만36-40세": 38, "만41-45세": 43,
    "알 수 없음": 0, "Unknown": 0
}

# EDA: 불임 원인 개수 합산에 쓸 세부 원인 컬럼
CAUSE_COLS = [
    "불임 원인 - 난관 질환", "불임 원인 - 남성 요인", "불임 원인 - 배란 장애",
    "불임 원인 - 여성 요인", "불임 원인 - 자궁경부 문제", "불임 원인 - 자궁내막증",
    "불임 원인 - 정자 농도", "불임 원인 - 정자 면역학적 요인",
    "불임 원인 - 정자 운동성", "불임 원인 - 정자 형태",
]

# EDA: log1p 변환 적용할 우편향 수치 변수 (skewness > 0.75)
LOG_COLS = [
    "총 생성 배아 수",           # skew 1.19
    "미세주입된 난자 수",         # skew 1.49
    "미세주입에서 생성된 배아 수", # skew 1.74
    "저장된 배아 수",             # skew 3.78
    "수집된 신선 난자 수",        # skew 0.89
    "혼합된 난자 수",             # skew 0.99
    "파트너 정자와 혼합된 난자 수",# skew 1.02
    "미세주입 배아 이식 수",      # skew 0.74
]

label_encoders = {}


# ════════════════════════════════════════════════════════════
# 헬퍼 함수
# ════════════════════════════════════════════════════════════

def convert_count(val):
    """'0회' ~ '6회 이상' 문자열 → 정수 변환"""
    if pd.isna(val) or val == "Unknown":
        return 0
    if "이상" in str(val):
        return 6
    try:
        return int(str(val).replace("회", "").strip())
    except:
        return 0


# ════════════════════════════════════════════════════════════
# 전처리 함수
# ════════════════════════════════════════════════════════════

def preprocess(df, is_train=True):
    df = df.copy()

    # ── 0. ID 분리 ────────────────────────────────────────
    ids = df["ID"].copy() if "ID" in df.columns else None
    df = df.drop(columns=["ID"], errors="ignore")

    # ── 1. 고결측 컬럼 제거 (결측률 84%↑) ────────────────
    df = df.drop(columns=[c for c in HIGH_NULL_COLS if c in df.columns])

    # ── 2. 경과일 변수: 결측 여부 플래그 생성 후 중앙값 대체 ──
    # EDA: 결측 패턴별 성공률이 2%~39%로 극단적 차이
    #      → 결측 여부 자체가 유효한 피처
    date_cols = [c for c in ["난자 채취 경과일", "난자 혼합 경과일", "배아 이식 경과일"]
                 if c in df.columns]
    for col in date_cols:
        df[col + "_결측여부"] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(df[col].median() if is_train else 0)

    # ── 3. 수치형 결측 → 0 대체 ──────────────────────────
    num_cols = df.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in num_cols if c != "임신 성공 여부"]
    df[num_cols] = df[num_cols].fillna(0)

    # ── 4. 범주형 결측 → "Unknown" 대체 ──────────────────
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    # ── 5. 횟수 컬럼 문자열 → 정수 변환 ──────────────────
    for col in COUNT_COLS:
        if col in df.columns:
            df[col] = df[col].apply(convert_count)

    # ── 6. 나이 컬럼 수치화 ───────────────────────────────
    if "시술 당시 나이" in df.columns:
        df["시술 당시 나이"] = df["시술 당시 나이"].map(AGE_MAP).fillna(36)
    for col in ["난자 기증자 나이", "정자 기증자 나이"]:
        if col in df.columns:
            df[col] = df[col].map(DONOR_AGE_MAP).fillna(0)

    # ── 7. 파생 피처: IVF/DI 합산 & 비율 ────────────────
    if "IVF 시술 횟수" in df.columns and "DI 시술 횟수" in df.columns:
        df["IVF_DI_시술_합산"] = df["IVF 시술 횟수"] + df["DI 시술 횟수"]
        df["IVF_시술_비율"]    = df["IVF 시술 횟수"] / (df["IVF_DI_시술_합산"] + 1e-6)

    if "IVF 임신 횟수" in df.columns and "DI 임신 횟수" in df.columns:
        df["IVF_DI_임신_합산"] = df["IVF 임신 횟수"] + df["DI 임신 횟수"]
        df["IVF_임신_비율"]    = df["IVF 임신 횟수"] / (df["IVF_DI_임신_합산"] + 1e-6)

    if "IVF 출산 횟수" in df.columns and "DI 출산 횟수" in df.columns:
        df["IVF_DI_출산_합산"] = df["IVF 출산 횟수"] + df["DI 출산 횟수"]
        df["IVF_출산_비율"]    = df["IVF 출산 횟수"] / (df["IVF_DI_출산_합산"] + 1e-6)

    if "IVF_DI_시술_합산" in df.columns and "IVF_DI_임신_합산" in df.columns:
        df["시술_대비_임신_비율"] = df["IVF_DI_임신_합산"] / (df["IVF_DI_시술_합산"] + 1e-6)

    # ── 8. 파생 피처: 불임 원인 개수 합산 ────────────────
    # EDA: 원인 종류·조합이 성공률에 영향 → 합산 개수 피처 추가
    cause_exist = [c for c in CAUSE_COLS if c in df.columns]
    if cause_exist:
        df["불임_원인_개수"] = df[cause_exist].sum(axis=1)

    # ── 9. 파생 피처: 배아 사용 유형 조합 ───────────────
    # EDA: 동결/신선/기증 조합 패턴이 시술 프로세스를 구분
    if all(c in df.columns for c in ["동결 배아 사용 여부", "신선 배아 사용 여부", "기증 배아 사용 여부"]):
        df["배아_사용_조합"] = (
            df["동결 배아 사용 여부"].fillna(0).astype(int).astype(str) +
            df["신선 배아 사용 여부"].fillna(0).astype(int).astype(str) +
            df["기증 배아 사용 여부"].fillna(0).astype(int).astype(str)
        )

    # ── 10. log1p 변환 (우편향 배아 수치 변수) ────────────
    # EDA: skewness 0.75↑ → 로그 변환으로 분포 정규화
    log_exist = [c for c in LOG_COLS if c in df.columns]
    for col in log_exist:
        df[col] = np.log1p(df[col])

    # ── 11. 범주형 → LabelEncoding ───────────────────────
    remaining_cat = df.select_dtypes(include=["object"]).columns.tolist()
    for col in remaining_cat:
        if is_train:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            if col in label_encoders:
                le = label_encoders[col]
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in le.classes_ else "Unknown"
                )
                df[col] = le.transform(df[col])
            else:
                df[col] = 0

    return df, ids


# ════════════════════════════════════════════════════════════
# 1. 데이터 불러오기 및 전처리
# ════════════════════════════════════════════════════════════
# train_raw = pd.read_csv("/Users/admin/Desktop/infertility/open (1)/train.csv")
# test_raw  = pd.read_csv("/Users/admin/Desktop/infertility/open (1)/test.csv")

train_raw = pd.read_csv("/content/drive/MyDrive/헬스케어 3기/프로젝트/미니프로젝트3/open (1)/train.csv")
test_raw  = pd.read_csv("/content/drive/MyDrive/헬스케어 3기/프로젝트/미니프로젝트3/open (1)/test.csv")

train_df, _       = preprocess(train_raw, is_train=True)
test_df, test_ids = preprocess(test_raw,  is_train=False)

# ════════════════════════════════════════════════════════════
# 2. X / y 분리
# ════════════════════════════════════════════════════════════
X = train_df.drop("임신 성공 여부", axis=1)
y = train_df["임신 성공 여부"]

X_submit = test_df.drop(columns=["임신 성공 여부"], errors="ignore")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Submit: {X_submit.shape}")
print(f"\n추가된 피처 목록:")
new_features = [c for c in X.columns if any(k in c for k in
    ["IVF_DI", "비율", "합산", "결측여부", "불임_원인_개수", "배아_사용_조합"])]
for f in new_features:
    print(f"  + {f}")
print(f"\n클래스 비율 (train) - 0: {(y_train==0).sum()}, 1: {(y_train==1).sum()}")

# ════════════════════════════════════════════════════════════
# 3. 모델 학습
# ════════════════════════════════════════════════════════════
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ════════════════════════════════════════════════════════════
# 4. 검증 성능 평가
# ════════════════════════════════════════════════════════════
val_preds = model.predict(X_val)
val_proba = model.predict_proba(X_val)[:, 1]

print("\n--- 검증 성능 결과 ---")
print(classification_report(y_val, val_preds, target_names=["실패(0)", "성공(1)"]))
print(f"AUC-ROC: {roc_auc_score(y_val, val_proba):.4f}")

# ════════════════════════════════════════════════════════════
# 5. 중요 변수 확인
# ════════════════════════════════════════════════════════════
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
top20 = feature_importances.nlargest(20)

colors = ["coral" if any(k in idx for k in
    ["IVF_DI", "비율", "합산", "결측여부", "불임_원인_개수", "배아_사용_조합"])
    else "steelblue" for idx in top20.index]

top20.plot(kind="barh", color=colors, figsize=(10, 8))
plt.title("Top 20 Important Features\n(주황: 새로 추가된 파생 피처)", fontsize=13)
plt.tight_layout()
plt.savefig("feature_importance_v3.png", dpi=150)
plt.show()

# ════════════════════════════════════════════════════════════
# 6. 제출 파일 생성
# ════════════════════════════════════════════════════════════
submit_proba = model.predict_proba(X_submit)[:, 1]

submission = pd.DataFrame({
    "ID": test_ids,
    "probability": submit_proba
})

submission.to_csv("baseline_RFC_v3.csv", index=False)
print("\n제출용 파일 'baseline_RFC_v3.csv' 생성 완료!")
print(submission.head())
print(f"총 {len(submission)}개 예측 완료")