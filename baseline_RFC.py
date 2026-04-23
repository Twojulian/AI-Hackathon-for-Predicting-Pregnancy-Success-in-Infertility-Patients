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

# ────────────────────────────────────────────
# 공통 전처리 함수
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

def base_preprocess(df, label_encoders, is_train=True):
    """공통 전처리 (나이 변환, count 변환, 인코딩 등)"""
    df = df.copy()

    ids = df["ID"].copy() if "ID" in df.columns else None
    df = df.drop(columns=["ID"], errors="ignore")

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
# 버전 A: 결측 컬럼 삭제 (기존 베이스라인)
# ────────────────────────────────────────────
def preprocess_v1(df, label_encoders, is_train=True):
    df = df.copy()
    df = df.drop(columns=[c for c in HIGH_NULL_COLS if c in df.columns])
    return base_preprocess(df, label_encoders, is_train)

# ────────────────────────────────────────────
# 버전 B: 결측 컬럼 살리기 (플래그 + 의미 있는 채우기)
# ────────────────────────────────────────────
def preprocess_v2(df, label_encoders, is_train=True):
    df = df.copy()

    # 수치형: -1로 채우고 결측 플래그 추가 (0일 vs 결측 구분)
    numeric_null_cols = ["난자 해동 경과일", "배아 해동 경과일", "임신 시도 또는 마지막 임신 경과 연수"]
    for col in numeric_null_cols:
        if col in df.columns:
            df[f"{col}_결측"] = df[col].isna().astype(int)
            df[col] = df[col].fillna(-1)

    # 범주형: "미시행" 카테고리로 명시적 처리
    category_null_cols = ["착상 전 유전 검사 사용 여부", "PGD 시술 여부", "PGS 시술 여부"]
    for col in category_null_cols:
        if col in df.columns:
            df[col] = df[col].fillna("미시행")

    return base_preprocess(df, label_encoders, is_train)

# ────────────────────────────────────────────
# 모델 학습 및 평가 공통 함수
# ────────────────────────────────────────────
def train_and_evaluate(X_train, y_train, X_val, y_val, label=""):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    val_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_proba)

    print(f"\n{'='*50}")
    print(f"[{label}] 검증 성능 결과")
    print(f"{'='*50}")
    print(classification_report(y_val, val_preds, target_names=["실패(0)", "성공(1)"]))
    print(f"AUC-ROC: {auc:.4f}")

    return model, auc, val_proba

# ────────────────────────────────────────────
# 1. 데이터 불러오기
# ────────────────────────────────────────────
train_raw = pd.read_csv("/Users/admin/Desktop/infertility/open (1)/train.csv")
test_raw  = pd.read_csv("/Users/admin/Desktop/infertility/open (1)/test.csv")

# ────────────────────────────────────────────
# 2. 버전 A 전처리 (결측 컬럼 삭제)
# ────────────────────────────────────────────
le_v1 = {}
train_v1, _         = preprocess_v1(train_raw, le_v1, is_train=True)
test_v1, test_ids   = preprocess_v1(test_raw,  le_v1, is_train=False)

X_v1 = train_v1.drop("임신 성공 여부", axis=1)
y_v1 = train_v1["임신 성공 여부"]
X_submit_v1 = test_v1.drop(columns=["임신 성공 여부"], errors="ignore")

X_train_v1, X_val_v1, y_train_v1, y_val_v1 = train_test_split(
    X_v1, y_v1, test_size=0.2, random_state=42, stratify=y_v1
)

# ────────────────────────────────────────────
# 3. 버전 B 전처리 (결측 컬럼 살리기)
# ────────────────────────────────────────────
le_v2 = {}
train_v2, _         = preprocess_v2(train_raw, le_v2, is_train=True)
test_v2, test_ids   = preprocess_v2(test_raw,  le_v2, is_train=False)

X_v2 = train_v2.drop("임신 성공 여부", axis=1)
y_v2 = train_v2["임신 성공 여부"]
X_submit_v2 = test_v2.drop(columns=["임신 성공 여부"], errors="ignore")

X_train_v2, X_val_v2, y_train_v2, y_val_v2 = train_test_split(
    X_v2, y_v2, test_size=0.2, random_state=42, stratify=y_v2
)

print(f"\n[버전 A] 피처 수: {X_v1.shape[1]}")
print(f"[버전 B] 피처 수: {X_v2.shape[1]} (결측 플래그 컬럼 추가됨)")

# ────────────────────────────────────────────
# 4. 학습 및 평가
# ────────────────────────────────────────────
model_v1, auc_v1, proba_v1 = train_and_evaluate(
    X_train_v1, y_train_v1, X_val_v1, y_val_v1,
    label="버전 A: 결측 컬럼 삭제 (베이스라인)"
)

model_v2, auc_v2, proba_v2 = train_and_evaluate(
    X_train_v2, y_train_v2, X_val_v2, y_val_v2,
    label="버전 B: 결측 컬럼 살리기 (플래그 + 의미 채우기)"
)

# ────────────────────────────────────────────
# 5. AUC 비교 요약
# ────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"📊 AUC-ROC 비교 요약")
print(f"{'='*50}")
print(f"버전 A (삭제):   {auc_v1:.4f}")
print(f"버전 B (살리기): {auc_v2:.4f}")
diff = auc_v2 - auc_v1
print(f"차이 (B - A):    {diff:+.4f}  {'✅ B가 더 좋음' if diff > 0 else '❌ A가 더 좋음' if diff < 0 else '➖ 동일'}")

# ────────────────────────────────────────────
# 6. 피처 중요도 비교 시각화
# ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

fi_v1 = pd.Series(model_v1.feature_importances_, index=X_v1.columns)
fi_v1.nlargest(10).plot(kind="barh", ax=axes[0], color="steelblue")
axes[0].set_title(f"버전 A - Top 10 Features\nAUC: {auc_v1:.4f}", fontsize=13)
axes[0].invert_yaxis()

fi_v2 = pd.Series(model_v2.feature_importances_, index=X_v2.columns)
fi_v2.nlargest(10).plot(kind="barh", ax=axes[1], color="coral")
axes[1].set_title(f"버전 B - Top 10 Features\nAUC: {auc_v2:.4f}", fontsize=13)
axes[1].invert_yaxis()

plt.suptitle("피처 중요도 비교: 결측 컬럼 삭제 vs 살리기", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("feature_importance_comparison.png", dpi=150)
plt.show()
print("\n비교 그래프 저장 완료: feature_importance_comparison.png")

# ────────────────────────────────────────────
# 7. 더 좋은 버전으로 제출 파일 생성
# ────────────────────────────────────────────
if auc_v2 >= auc_v1:
    best_model    = model_v2
    X_submit_best = X_submit_v2
    best_version  = "B"
else:
    best_model    = model_v1
    X_submit_best = X_submit_v1
    best_version  = "A"

submit_proba = best_model.predict_proba(X_submit_best)[:, 1]
submission = pd.DataFrame({
    "ID": test_ids,
    "probability": submit_proba
})
submission.to_csv(f"submission_best_v{best_version}.csv", index=False)
print(f"\n✅ 버전 {best_version} 기준으로 제출 파일 생성 완료!")
print(submission.head())
print(f"총 {len(submission)}개 예측 완료")