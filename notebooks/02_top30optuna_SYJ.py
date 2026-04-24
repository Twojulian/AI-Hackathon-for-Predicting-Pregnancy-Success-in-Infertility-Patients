# ════════════════════════════════════════════════════════════
# Optuna — RandomForest, Top 30 피처 고정
# ════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import optuna
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ════════════════════════════════════════════════════════════
# 공통 설정값 (V5 그대로)
# ════════════════════════════════════════════════════════════

HIGH_NULL_COLS = [
    "착상 전 유전 검사 사용 여부", "PGD 시술 여부", "PGS 시술 여부",
    "난자 해동 경과일", "임신 시도 또는 마지막 임신 경과 연수", "배아 해동 경과일",
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
CAUSE_COLS = [
    "불임 원인 - 난관 질환", "불임 원인 - 남성 요인", "불임 원인 - 배란 장애",
    "불임 원인 - 여성 요인", "불임 원인 - 자궁경부 문제", "불임 원인 - 자궁내막증",
    "불임 원인 - 정자 농도", "불임 원인 - 정자 면역학적 요인",
    "불임 원인 - 정자 운동성", "불임 원인 - 정자 형태",
]
LOG_COLS = [
    "총 생성 배아 수", "미세주입된 난자 수", "미세주입에서 생성된 배아 수",
    "저장된 배아 수", "수집된 신선 난자 수", "혼합된 난자 수",
    "파트너 정자와 혼합된 난자 수", "미세주입 배아 이식 수",
]

label_encoders = {}


# ════════════════════════════════════════════════════════════
# 전처리 함수 (V5 그대로)
# ════════════════════════════════════════════════════════════

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

    date_cols = [c for c in ["난자 채취 경과일", "난자 혼합 경과일", "배아 이식 경과일"] if c in df.columns]
    for col in date_cols:
        df[col + "_결측여부"] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(df[col].median() if is_train else 0)

    num_cols = [c for c in df.select_dtypes(include="number").columns if c != "임신 성공 여부"]
    df[num_cols] = df[num_cols].fillna(0)
    df[df.select_dtypes(include="object").columns] = df.select_dtypes(include="object").fillna("Unknown")

    for col in COUNT_COLS:
        if col in df.columns:
            df[col] = df[col].apply(convert_count)

    if "시술 당시 나이" in df.columns:
        df["시술 당시 나이"] = df["시술 당시 나이"].map(AGE_MAP).fillna(36)
    for col in ["난자 기증자 나이", "정자 기증자 나이"]:
        if col in df.columns:
            df[col] = df[col].map(DONOR_AGE_MAP).fillna(0)

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

    cause_exist = [c for c in CAUSE_COLS if c in df.columns]
    if cause_exist:
        df["불임_원인_개수"] = df[cause_exist].sum(axis=1)

    if all(c in df.columns for c in ["동결 배아 사용 여부", "신선 배아 사용 여부", "기증 배아 사용 여부"]):
        df["배아_사용_조합"] = (
            df["동결 배아 사용 여부"].fillna(0).astype(int).astype(str) +
            df["신선 배아 사용 여부"].fillna(0).astype(int).astype(str) +
            df["기증 배아 사용 여부"].fillna(0).astype(int).astype(str)
        )

    for col in [c for c in LOG_COLS if c in df.columns]:
        df[col] = np.log1p(df[col])

    for col in df.select_dtypes(include="object").columns:
        if is_train:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            if col in label_encoders:
                le = label_encoders[col]
                df[col] = df[col].astype(str).apply(lambda x: x if x in le.classes_ else "Unknown")
                df[col] = le.transform(df[col])
            else:
                df[col] = 0

    return df, ids


# ════════════════════════════════════════════════════════════
# 데이터 로드 (경로 수정)
# ════════════════════════════════════════════════════════════

train = pd.read_csv('/kaggle/input/datasets/yjsheila/infertility/train.csv')
test = pd.read_csv('/kaggle/input/datasets/yjsheila/infertility/test.csv')
sample = pd.read_csv('/kaggle/input/datasets/yjsheila/infertility/sample_submission.csv')

train_df, _       = preprocess(train, is_train=True)
test_df, test_ids = preprocess(test,  is_train=False)

X = train_df.drop("임신 성공 여부", axis=1)
y = train_df["임신 성공 여부"]
X_submit = test_df.drop(columns=["임신 성공 여부"], errors="ignore")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ════════════════════════════════════════════════════════════
# Top 30 피처 선택 (V5와 동일한 기준 RF로 추출)
# ════════════════════════════════════════════════════════════

_selector = RandomForestClassifier(
    n_estimators=100, max_depth=10,
    class_weight="balanced", random_state=42, n_jobs=-1
)
_selector.fit(X_train, y_train)

top30_features = (
    pd.Series(_selector.feature_importances_, index=X_train.columns)
    .nlargest(30)
    .index.tolist()
)

print(f"Top 30 피처 확정:")
for i, f in enumerate(top30_features, 1):
    print(f"  {i:>2}. {f}")

# Top 30으로 슬라이싱
X_train_30  = X_train[top30_features]
X_val_30    = X_val[top30_features]
X_submit_30 = X_submit[top30_features]


# ════════════════════════════════════════════════════════════
# Optuna Objective — RF 하이퍼파라미터만 탐색
# ════════════════════════════════════════════════════════════

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth":         trial.suggest_int("max_depth", 5, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7]),
        "class_weight":      trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
        "random_state": 42,
        "n_jobs": -1,
    }
    model = RandomForestClassifier(**params)
    scores = cross_val_score(model, X_train_30, y_train,
                            cv=CV, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


# ════════════════════════════════════════════════════════════
# Optuna 실행
# ════════════════════════════════════════════════════════════

N_TRIALS = 50

print(f"\n{'='*50}")
print(f"  Optuna 탐색 시작 — {N_TRIALS} trials / Top 30 피처")
print(f"{'='*50}\n")

storage = "sqlite:///optuna_study.db"
study_name = "rf_top30"

study = optuna.create_study(
    study_name=study_name,
    storage=storage,
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    load_if_exists=True,  # 이어서 하기 핵심!
)

# 이미 완료된 trials 확인
completed = len([t for t in study.trials if t.state.name == "COMPLETE"])
remaining = max(0, N_TRIALS - completed)
print(f"이미 완료된 trials: {completed}, 남은 trials: {remaining}")

study.optimize(objective, n_trials=remaining, show_progress_bar=True)

print(f"\n✅ 탐색 완료!")
print(f"   Best CV AUC : {study.best_value:.4f}")
print(f"   Best Params :")
for k, v in study.best_params.items():
    print(f"     {k}: {v}")


# ════════════════════════════════════════════════════════════
# 최적 파라미터로 최종 모델 학습 & 평가
# ════════════════════════════════════════════════════════════

final_model = RandomForestClassifier(
    **study.best_params,
    random_state=42,
    n_jobs=-1,
)
# 전체 데이터(train+val)로 최종 학습
final_model.fit(X[top30_features], y)

val_auc = roc_auc_score(y_val, final_model.predict_proba(X_val_30)[:, 1])
print(f"\n--- 최종 모델 Val AUC: {val_auc:.4f} ---\n")


# ════════════════════════════════════════════════════════════
# 제출 파일 생성
# ════════════════════════════════════════════════════════════

submission = pd.DataFrame({
    "ID": test_ids,
    "probability": final_model.predict_proba(X_submit_30)[:, 1]
})
submission.to_csv("optuna_rf_top30.csv", index=False)
print("제출 파일 'optuna_rf_top30.csv' 생성 완료!")
print(submission.head())
print(f"총 {len(submission)}개 예측 완료")