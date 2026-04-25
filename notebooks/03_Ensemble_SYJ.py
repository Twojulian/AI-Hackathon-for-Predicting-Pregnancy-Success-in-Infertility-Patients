import pandas as pd
import numpy as np
import optuna
import warnings
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
        df["IVF_시술_비율"] = df["IVF 시술 횟수"] / (df["IVF_DI_시술_합산"] + 1e-6)
    if "IVF 임신 횟수" in df.columns and "DI 임신 횟수" in df.columns:
        df["IVF_DI_임신_합산"] = df["IVF 임신 횟수"] + df["DI 임신 횟수"]
        df["IVF_임신_비율"] = df["IVF 임신 횟수"] / (df["IVF_DI_임신_합산"] + 1e-6)
    if "IVF 출산 횟수" in df.columns and "DI 출산 횟수" in df.columns:
        df["IVF_DI_출산_합산"] = df["IVF 출산 횟수"] + df["DI 출산 횟수"]
        df["IVF_출산_비율"] = df["IVF 출산 횟수"] / (df["IVF_DI_출산_합산"] + 1e-6)
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

train = pd.read_csv('/kaggle/input/datasets/yjsheila/infertility/train.csv')
test = pd.read_csv('/kaggle/input/datasets/yjsheila/infertility/test.csv')
sample = pd.read_csv('/kaggle/input/datasets/yjsheila/infertility/sample_submission.csv')

train_df, _ = preprocess(train, is_train=True)
test_df, test_ids = preprocess(test, is_train=False)

X = train_df.drop("임신 성공 여부", axis=1)
y = train_df["임신 성공 여부"]
X_submit = test_df.drop(columns=["임신 성공 여부"], errors="ignore")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

_selector = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
_selector.fit(X_train, y_train)
top30_features = pd.Series(_selector.feature_importances_, index=X_train.columns).nlargest(30).index.tolist()

X_train_30 = X_train[top30_features]
X_val_30 = X_val[top30_features]
X_submit_30 = X_submit[top30_features]

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ════ XGBoost Optuna ════
def objective_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
        "random_state": 42, "n_jobs": -1, "eval_metric": "auc",
    }
    model = xgb.XGBClassifier(**params)
    scores = cross_val_score(model, X_train_30, y_train, cv=CV, scoring="roc_auc", n_jobs=-1)
    return scores.mean()

study_xgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study_xgb.optimize(objective_xgb, n_trials=20, show_progress_bar=True)

print(f"\n✅ XGBoost Best AUC: {study_xgb.best_value:.4f}")
print(f"XGBoost Best Params:")
for k, v in study_xgb.best_params.items():
    print(f"  {k}: {v}")

# ════ ExtraTrees Optuna ════
def objective_et(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
        "random_state": 42, "n_jobs": -1,
    }
    model = ExtraTreesClassifier(**params)
    scores = cross_val_score(model, X_train_30, y_train, cv=CV, scoring="roc_auc", n_jobs=-1)
    return scores.mean()

study_et = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study_et.optimize(objective_et, n_trials=20, show_progress_bar=True)

print(f"\n✅ ExtraTrees Best AUC: {study_et.best_value:.4f}")
print(f"ExtraTrees Best Params:")
for k, v in study_et.best_params.items():
    print(f"  {k}: {v}")

# ════ 앙상블 ════
final_rf = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_split=2,
    min_samples_leaf=9, max_features=0.7, class_weight="balanced",
    random_state=42, n_jobs=-1
)
final_xgb = xgb.XGBClassifier(**study_xgb.best_params, random_state=42, n_jobs=-1)
final_et = ExtraTreesClassifier(**study_et.best_params, random_state=42, n_jobs=-1)

final_rf.fit(X[top30_features], y)
final_xgb.fit(X[top30_features], y)
final_et.fit(X[top30_features], y)

pred_rf = final_rf.predict_proba(X_submit_30)[:, 1]
pred_xgb = final_xgb.predict_proba(X_submit_30)[:, 1]
pred_et = final_et.predict_proba(X_submit_30)[:, 1]

ensemble_pred = (pred_rf + pred_xgb + pred_et) / 3

# 각 모델 Val AUC
val_rf = roc_auc_score(y_val, final_rf.predict_proba(X_val_30)[:, 1])
val_xgb = roc_auc_score(y_val, final_xgb.predict_proba(X_val_30)[:, 1])
val_et = roc_auc_score(y_val, final_et.predict_proba(X_val_30)[:, 1])

# 앙상블 Val AUC
ensemble_val = (
    final_rf.predict_proba(X_val_30)[:, 1] +
    final_xgb.predict_proba(X_val_30)[:, 1] +
    final_et.predict_proba(X_val_30)[:, 1]
) / 3

val_ensemble_auc = roc_auc_score(y_val, ensemble_val)
val_ensemble_pred = (ensemble_val >= 0.5).astype(int)

print(f"\n{'='*40}")
print(f"  각 모델 Val AUC")
print(f"  RF  : {val_rf:.4f}")
print(f"  XGB : {val_xgb:.4f}")
print(f"  ET  : {val_et:.4f}")
print(f"  앙상블: {val_ensemble_auc:.4f}")
print(f"{'='*40}")
print(f"\n분류 리포트 (앙상블):")
print(classification_report(y_val, val_ensemble_pred))


# submission = pd.DataFrame({"ID": test_ids, "probability": ensemble_pred})
# submission.to_csv("ensemble_rf_xgb_et.csv", index=False)
# print("\n✅ 앙상블 제출 파일 생성 완료!")
# print(submission.head())
# print(f"총 {len(submission)}개 예측 완료")


# RF에 가중치 더 주는 앙상블
pred_rf = final_rf.predict_proba(X_submit_30)[:, 1]
pred_xgb = final_xgb.predict_proba(X_submit_30)[:, 1]
pred_et = final_et.predict_proba(X_submit_30)[:, 1]

# RF 0.5, XGB 0.25, ET 0.25
ensemble_pred_weighted = (pred_rf * 0.5 + pred_xgb * 0.25 + pred_et * 0.25)

# Val AUC 확인
val_weighted = (
    final_rf.predict_proba(X_val_30)[:, 1] * 0.5 +
    final_xgb.predict_proba(X_val_30)[:, 1] * 0.25 +
    final_et.predict_proba(X_val_30)[:, 1] * 0.25
)
print(f"가중치 앙상블 Val AUC: {roc_auc_score(y_val, val_weighted):.4f}")

submission_weighted = pd.DataFrame({"ID": test_ids, "probability": ensemble_pred_weighted})
submission_weighted.to_csv("ensemble_weighted.csv", index=False)
print("✅ 가중치 앙상블 파일 생성 완료!")

# RF + ET 앙상블
val_rf_et = (
    final_rf.predict_proba(X_val_30)[:, 1] * 0.6 +
    final_et.predict_proba(X_val_30)[:, 1] * 0.4
)
print(f"RF+ET 앙상블 Val AUC: {roc_auc_score(y_val, val_rf_et):.4f}")

# RF + XGB 앙상블
val_rf_xgb = (
    final_rf.predict_proba(X_val_30)[:, 1] * 0.6 +
    final_xgb.predict_proba(X_val_30)[:, 1] * 0.4
)
print(f"RF+XGB 앙상블 Val AUC: {roc_auc_score(y_val, val_rf_xgb):.4f}")


pred_rf_et = (
    final_rf.predict_proba(X_submit_30)[:, 1] * 0.6 +
    final_et.predict_proba(X_submit_30)[:, 1] * 0.4
)

submission_rf_et = pd.DataFrame({"ID": test_ids, "probability": pred_rf_et})
submission_rf_et.to_csv("ensemble_rf_et.csv", index=False)
print("✅ RF+ET 앙상블 파일 생성 완료!")
print(submission_rf_et.head())