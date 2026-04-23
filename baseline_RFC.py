import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ────────────────────────────────────────────
# 공통 설정값
# ────────────────────────────────────────────
HIGH_NULL_COLS = [
    "착상 전 유전 검사 사용 여부",
    "PGD 시술 여부",
    "PGS 시술 여부",
    "난자 해동 경과일",
    "임신 시도 또는 마지막 임신 경과 연수",
    "배아 해동 경과일",
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

label_encoders = {}

# ────────────────────────────────────────────
# 전처리 함수 정의 (← 호출보다 반드시 먼저!)
# ────────────────────────────────────────────
def convert_count(val):
    if pd.isna(val) or val == "Unknown":
        return 0
    if "이상" in str(val):
        return 6
    try:
        return int(str(val).replace("회", "").strip())
    except:
        return 0

def preprocess(df, is_train=True):
    df = df.copy()

    ids = df["ID"].copy() if "ID" in df.columns else None
    df = df.drop(columns=["ID"], errors="ignore")

    df = df.drop(columns=[c for c in HIGH_NULL_COLS if c in df.columns])

    num_cols = df.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in num_cols if c != "임신 성공 여부"]
    df[num_cols] = df[num_cols].fillna(0)

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    for col in COUNT_COLS:
        if col in df.columns:
            df[col] = df[col].apply(convert_count)

    if "시술 당시 나이" in df.columns:
        df["시술 당시 나이"] = df["시술 당시 나이"].map(AGE_MAP).fillna(36)

    for col in ["난자 기증자 나이", "정자 기증자 나이"]:
        if col in df.columns:
            df[col] = df[col].map(DONOR_AGE_MAP).fillna(0)

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

# ────────────────────────────────────────────
# 1. 데이터 불러오기 및 전처리 (← 함수 정의 후에 호출)
# ────────────────────────────────────────────
train_raw = pd.read_csv("/Users/admin/Desktop/infertility/open (1)/train.csv")
test_raw  = pd.read_csv("/Users/admin/Desktop/infertility/open (1)/test.csv")

train_df, _       = preprocess(train_raw, is_train=True)
test_df, test_ids = preprocess(test_raw,  is_train=False)

# ────────────────────────────────────────────
# 2. X / y 분리
# ────────────────────────────────────────────
X = train_df.drop("임신 성공 여부", axis=1)
y = train_df["임신 성공 여부"]

X_submit = test_df.drop(columns=["임신 성공 여부"], errors="ignore")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Submit: {X_submit.shape}")
print(f"클래스 비율 (train) - 0: {(y_train==0).sum()}, 1: {(y_train==1).sum()}")

# ────────────────────────────────────────────
# 3. 모델 학습
# ────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ────────────────────────────────────────────
# 4. 검증 성능 평가
# ────────────────────────────────────────────
val_preds = model.predict(X_val)
val_proba = model.predict_proba(X_val)[:, 1]

print("\n--- 검증 성능 결과 ---")
print(classification_report(y_val, val_preds, target_names=["실패(0)", "성공(1)"]))
print(f"AUC-ROC: {roc_auc_score(y_val, val_proba):.4f}")

# ────────────────────────────────────────────
# 5. 중요 변수 확인
# ────────────────────────────────────────────
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind="barh")
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

# ────────────────────────────────────────────
# 6. 제출 파일 생성
# ────────────────────────────────────────────
submit_preds = model.predict(X_submit)

submission = pd.DataFrame({
    "ID": test_ids,
    "임신 성공 여부": submit_preds
})

submission.to_csv("03_baseRFC_SYJ.csv", index=False)
print("\n제출용 파일 '03_baseRFC_SYJ.csv' 생성 완료!")
print(submission.head())
print(f"총 {len(submission)}개 예측 완료")

# ────────────────────────────────────────────
# 6. 제출 파일 생성
# ────────────────────────────────────────────
submit_proba = model.predict_proba(X_submit)[:, 1]  # 확률값 (1일 확률)

submission = pd.DataFrame({
    "ID": test_ids,
    "probability": submit_proba  # 컬럼명도 변경!
})

submission.to_csv("03_baseRFC_SYJ.csv", index=False)
print("\n제출용 파일 '03_baseRFC_SYJ.csv' 생성 완료!")
print(submission.head())
print(f"총 {len(submission)}개 예측 완료")