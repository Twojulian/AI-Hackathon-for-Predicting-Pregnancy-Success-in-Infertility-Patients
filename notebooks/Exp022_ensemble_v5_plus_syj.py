# ════════════════════════════════════════════════════════════
# exp022 — LGB + CAT + XGB 앙상블
#          v3 전처리 + 조여진님 파생 피처 추가
#          OOF 기반 StratifiedKFold + Rank Average 비교
# 기준선: exp021 (Val 0.7374, 제출 0.7413)
# ════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, classification_report,
                            confusion_matrix, f1_score,
                            recall_score, precision_score, accuracy_score)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve
import os

warnings.filterwarnings("ignore")

SEED     = 42
N_FOLDS  = 5
TARGET   = "임신 성공 여부"
EXP_NO   = 22
AUTHOR   = "SYJ"
BASELINE = 0.7374   # exp021 Val AUC


# ════════════════════════════════════════════════════════════
# 설정값
# ════════════════════════════════════════════════════════════

HIGH_NULL_COLS = [
    "착상 전 유전 검사 사용 여부", "PGD 시술 여부", "PGS 시술 여부",
    "난자 해동 경과일", "배아 해동 경과일",
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

# 조여진님 코드에서 추가로 필요한 불임원인 컬럼
MALE_COLS   = ["남성 주 불임 원인", "남성 부 불임 원인", "불임 원인 - 남성 요인"]
FEMALE_COLS = ["여성 주 불임 원인", "여성 부 불임 원인", "불임 원인 - 난관 질환",
                "불임 원인 - 배란 장애", "불임 원인 - 자궁내막증", "불임 원인 - 자궁경부 문제"]
COUPLE_COLS = ["부부 주 불임 원인", "부부 부 불임 원인"]
SPERM_COLS  = ["불임 원인 - 정자 농도", "불임 원인 - 정자 운동성",
                "불임 원인 - 정자 형태", "불임 원인 - 정자 면역학적 요인"]

label_encoders = {}


# ════════════════════════════════════════════════════════════
# 전처리 함수 (v3 + 조여진님 파생 피처)
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

    # 날짜 컬럼 결측 처리
    date_cols = [c for c in ["난자 채취 경과일", "난자 혼합 경과일", "배아 이식 경과일"] if c in df.columns]
    for col in date_cols:
        df[col + "_결측여부"] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(df[col].median() if is_train else 0)

    num_cols = [c for c in df.select_dtypes(include="number").columns if c != TARGET]
    df[num_cols] = df[num_cols].fillna(0)
    df[df.select_dtypes(include="object").columns] = df.select_dtypes(include="object").fillna("Unknown")

    # 횟수 변환
    for col in COUNT_COLS:
        if col in df.columns:
            df[col] = df[col].apply(convert_count)

    # 나이 변환
    if "시술 당시 나이" in df.columns:
        df["시술 당시 나이"] = df["시술 당시 나이"].map(AGE_MAP).fillna(36)
    for col in ["난자 기증자 나이", "정자 기증자 나이"]:
        if col in df.columns:
            df[col] = df[col].map(DONOR_AGE_MAP).fillna(0)

    # IVF/DI 비율 피처 (v3)
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

    # 불임 원인 개수 (v3)
    cause_exist = [c for c in CAUSE_COLS if c in df.columns]
    if cause_exist:
        df["불임_원인_개수"] = df[cause_exist].sum(axis=1)

    # 배아 사용 조합 (v3)
    if all(c in df.columns for c in ["동결 배아 사용 여부", "신선 배아 사용 여부", "기증 배아 사용 여부"]):
        df["배아_사용_조합"] = (
            df["동결 배아 사용 여부"].fillna(0).astype(int).astype(str) +
            df["신선 배아 사용 여부"].fillna(0).astype(int).astype(str) +
            df["기증 배아 사용 여부"].fillna(0).astype(int).astype(str)
        )

    # 임신 시도 경과 연수 (v3)
    if "임신 시도 또는 마지막 임신 경과 연수" in df.columns:
        df["임신 시도 또는 마지막 임신 경과 연수"] = pd.to_numeric(
            df["임신 시도 또는 마지막 임신 경과 연수"], errors="coerce"
        )
        df["임신시도_결측여부"] = df["임신 시도 또는 마지막 임신 경과 연수"].isnull().astype(int)
        df["임신 시도 또는 마지막 임신 경과 연수"] = df["임신 시도 또는 마지막 임신 경과 연수"].fillna(
            df["임신 시도 또는 마지막 임신 경과 연수"].median() if is_train else 0
        )

    # 이전 임신/출산 여부 (v3)
    if "총 임신 횟수" in df.columns:
        df["이전_임신_여부"] = (df["총 임신 횟수"] > 0).astype(int)
    if "총 출산 횟수" in df.columns:
        df["이전_출산_여부"] = (df["총 출산 횟수"] > 0).astype(int)

    # 시술 성공률 (v3)
    if "총 임신 횟수" in df.columns and "총 시술 횟수" in df.columns:
        df["전체_임신_성공률"] = df["총 임신 횟수"] / (df["총 시술 횟수"] + 1e-6)

    # 난자 품질 지표 (v3)
    if "수집된 신선 난자 수" in df.columns and "총 생성 배아 수" in df.columns:
        df["난자_배아_전환율"] = df["총 생성 배아 수"] / (df["수집된 신선 난자 수"] + 1e-6)

    # 고령/첫 시술 여부 (v3)
    if "시술 당시 나이" in df.columns:
        df["고령_여부"] = (df["시술 당시 나이"] >= 38).astype(int)
    if "총 시술 횟수" in df.columns:
        df["첫_시술_여부"] = (df["총 시술 횟수"] == 1).astype(int)

    # 배아 이식 경과일 파생 (v3 exp020)
    if "배아 이식 경과일" in df.columns:
        df["배아_이식_경과일_구간"] = pd.cut(
            df["배아 이식 경과일"], bins=[-1, 2, 5, 9999], labels=[0, 1, 2]
        ).astype(int)
        df["조기_이식_여부"] = (df["배아 이식 경과일"] <= 3).astype(int)
    if "난자 채취 경과일" in df.columns and "배아 이식 경과일" in df.columns:
        df["채취_이식_경과일_차이"] = df["배아 이식 경과일"] - df["난자 채취 경과일"]
    if "난자 혼합 경과일" in df.columns and "배아 이식 경과일" in df.columns:
        df["혼합_이식_경과일_차이"] = df["배아 이식 경과일"] - df["난자 혼합 경과일"]

    # ── 신규: 조여진님 파생 피처 ──

    # 기증 난자/정자 여부
    if "난자 출처" in df.columns:
        df["기증_난자_여부"] = (df["난자 출처"] == "기증 제공").astype(int)
    if "정자 출처" in df.columns:
        df["기증_정자_여부"] = df["정자 출처"].isin(["기증 제공", "배우자 및 기증 제공"]).astype(int)

    # 수정률 / 이식률 / 저장률 / ICSI 비율
    eps = 1e-6
    if "총 생성 배아 수" in df.columns and "혼합된 난자 수" in df.columns:
        df["수정률"] = df["총 생성 배아 수"] / (df["혼합된 난자 수"] + eps)
    if "이식된 배아 수" in df.columns and "총 생성 배아 수" in df.columns:
        df["이식률"] = df["이식된 배아 수"] / (df["총 생성 배아 수"] + eps)
    if "저장된 배아 수" in df.columns and "총 생성 배아 수" in df.columns:
        df["저장률"] = df["저장된 배아 수"] / (df["총 생성 배아 수"] + eps)
    if "미세주입된 난자 수" in df.columns and "혼합된 난자 수" in df.columns:
        df["ICSI_비율"] = df["미세주입된 난자 수"] / (df["혼합된 난자 수"] + eps)

    # 배아 발달일
    if "배아 이식 경과일" in df.columns and "난자 혼합 경과일" in df.columns:
        df["배아_발달일"] = df["배아 이식 경과일"] - df["난자 혼합 경과일"]

    # 신선 시술 여부
    if "수집된 신선 난자 수" in df.columns:
        df["신선_시술_여부"] = df["수집된 신선 난자 수"].notna().astype(int)

    # 불임원인 성별/유형별 합계 (v3의 불임_원인_개수와 다른 세분화)
    male_exist   = [c for c in MALE_COLS   if c in df.columns]
    female_exist = [c for c in FEMALE_COLS if c in df.columns]
    couple_exist = [c for c in COUPLE_COLS if c in df.columns]
    sperm_exist  = [c for c in SPERM_COLS  if c in df.columns]
    if male_exist:
        df["남성_불임_합계"] = df[male_exist].sum(axis=1)
    if female_exist:
        df["여성_불임_합계"] = df[female_exist].sum(axis=1)
    if couple_exist:
        df["부부_불임_합계"] = df[couple_exist].sum(axis=1)
    if sperm_exist:
        df["정자_문제_합계"] = df[sperm_exist].sum(axis=1)

    # 나이 × 시술횟수 상호작용
    if "시술 당시 나이" in df.columns and "총 시술 횟수" in df.columns:
        df["나이_시술횟수_상호작용"] = df["시술 당시 나이"] * df["총 시술 횟수"]

    # 신선 난자 저장 여부
    if "저장된 신선 난자 수" in df.columns:
        df["신선_난자_저장_있음"] = (df["저장된 신선 난자 수"] > 0).astype(int)

    # log 변환
    for col in [c for c in LOG_COLS if c in df.columns]:
        df[col] = np.log1p(df[col])

    # Label Encoding
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
# 데이터 로드
# ════════════════════════════════════════════════════════════

# 로컬 경로에 맞게 수정해주세요
DATA_DIR = "../data/raw"   # 또는 절대경로

train  = pd.read_csv(f"{DATA_DIR}/train.csv")
test   = pd.read_csv(f"{DATA_DIR}/test.csv")
sample = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

train_df, _       = preprocess(train, is_train=True)
test_df, test_ids = preprocess(test,  is_train=False)

X_all    = train_df.drop(TARGET, axis=1)
y_all    = train_df[TARGET]
X_submit = test_df.drop(columns=[TARGET], errors="ignore")

neg_pos_ratio = (y_all == 0).sum() / (y_all == 1).sum()

print(f"사용 피처 수    : {X_all.shape[1]}")
print(f"scale_pos_weight: {neg_pos_ratio:.4f}")
print(f"추가된 신규 피처 예시: 수정률, 이식률, 저장률, ICSI_비율, 남성_불임_합계, 나이_시술횟수_상호작용 등")


# ════════════════════════════════════════════════════════════
# 모델 파라미터 (exp021 그대로)
# ════════════════════════════════════════════════════════════

LGB_PARAMS = {
    "n_estimators":      400,
    "learning_rate":     0.04956090868322492,
    "num_leaves":        112,
    "max_depth":         4,
    "min_child_samples": 26,
    "subsample":         0.7523886151586271,
    "colsample_bytree":  0.6353997510251372,
    "reg_alpha":         5.682819774457299,
    "reg_lambda":        8.955144228161172,
    "scale_pos_weight":  neg_pos_ratio,
    "random_state":      SEED,
    "n_jobs":            -1,
    "verbose":           -1,
}

CAT_PARAMS = {
    "iterations":       500,
    "learning_rate":    0.05,
    "depth":            6,
    "scale_pos_weight": neg_pos_ratio,
    "random_seed":      SEED,
    "verbose":          0,
    "eval_metric":      "AUC",
    "thread_count":     -1,
}

XGB_PARAMS = {
    "n_estimators":     500,
    "learning_rate":    0.05,
    "max_depth":        6,
    "min_child_weight": 1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": neg_pos_ratio,
    "random_state":     SEED,
    "n_jobs":           -1,
    "verbosity":        0,
    "eval_metric":      "auc",
}


# ════════════════════════════════════════════════════════════
# OOF 학습 (StratifiedKFold 5-Fold)
# ════════════════════════════════════════════════════════════

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb  = np.zeros(len(X_all))
oof_cat  = np.zeros(len(X_all))
oof_xgb  = np.zeros(len(X_all))
test_lgb = np.zeros(len(X_submit))
test_cat = np.zeros(len(X_submit))
test_xgb = np.zeros(len(X_submit))

print(f"\n{'='*60}")
print(f"OOF 학습 시작 (Stratified {N_FOLDS}-Fold)")
print(f"{'='*60}")

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_all), 1):
    X_tr, X_val = X_all.iloc[tr_idx], X_all.iloc[val_idx]
    y_tr, y_val = y_all.iloc[tr_idx], y_all.iloc[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMClassifier(**LGB_PARAMS)
    lgb_model.fit(X_tr, y_tr)
    oof_lgb[val_idx]  = lgb_model.predict_proba(X_val)[:, 1]
    test_lgb         += lgb_model.predict_proba(X_submit)[:, 1] / N_FOLDS

    # CatBoost
    cat_model = CatBoostClassifier(**CAT_PARAMS)
    cat_model.fit(X_tr, y_tr)
    oof_cat[val_idx]  = cat_model.predict_proba(X_val)[:, 1]
    test_cat         += cat_model.predict_proba(X_submit)[:, 1] / N_FOLDS

    # XGBoost
    xgb_model = XGBClassifier(**XGB_PARAMS)
    xgb_model.fit(X_tr, y_tr)
    oof_xgb[val_idx]  = xgb_model.predict_proba(X_val)[:, 1]
    test_xgb         += xgb_model.predict_proba(X_submit)[:, 1] / N_FOLDS

    fold_lgb = roc_auc_score(y_val, oof_lgb[val_idx])
    fold_cat = roc_auc_score(y_val, oof_cat[val_idx])
    fold_xgb = roc_auc_score(y_val, oof_xgb[val_idx])
    print(f"  Fold {fold}  LGB={fold_lgb:.4f}  CAT={fold_cat:.4f}  XGB={fold_xgb:.4f}")

auc_lgb = roc_auc_score(y_all, oof_lgb)
auc_cat = roc_auc_score(y_all, oof_cat)
auc_xgb = roc_auc_score(y_all, oof_xgb)

print(f"\n{'='*60}")
print(f"OOF AUC  LGB={auc_lgb:.5f}  CAT={auc_cat:.5f}  XGB={auc_xgb:.5f}")
print(f"{'='*60}")


# ════════════════════════════════════════════════════════════
# 앙상블 4가지 비교
# ════════════════════════════════════════════════════════════

oofs  = np.stack([oof_lgb,  oof_cat,  oof_xgb],  axis=1)
tests = np.stack([test_lgb, test_cat, test_xgb], axis=1)
aucs  = np.array([auc_lgb, auc_cat, auc_xgb])

results = {}

# 1. Simple Average
results["Simple Average"] = (
    roc_auc_score(y_all, oofs.mean(axis=1)),
    tests.mean(axis=1)
)

# 2. AUC-weighted
w_auc = aucs / aucs.sum()
results["AUC-weighted"] = (
    roc_auc_score(y_all, (oofs * w_auc).sum(axis=1)),
    (tests * w_auc).sum(axis=1)
)

# 3. Rank Average
oof_ranks  = np.stack([rankdata(oofs[:, i])  for i in range(3)], axis=1)
test_ranks = np.stack([rankdata(tests[:, i]) for i in range(3)], axis=1)
results["Rank Average"] = (
    roc_auc_score(y_all, oof_ranks.mean(axis=1)),
    test_ranks.mean(axis=1)
)

# 4. Optuna 가중치
def optuna_objective(trial):
    w = np.array([
        trial.suggest_float("w_lgb", 0.0, 1.0),
        trial.suggest_float("w_cat", 0.0, 1.0),
        trial.suggest_float("w_xgb", 0.0, 1.0),
    ])
    w = w / w.sum()
    return roc_auc_score(y_all, (oofs * w).sum(axis=1))

study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(optuna_objective, n_trials=200, show_progress_bar=True)
best_w = np.array([study.best_params["w_lgb"],
                    study.best_params["w_cat"],
                    study.best_params["w_xgb"]])
best_w = best_w / best_w.sum()
results["Optuna Weights"] = (
    roc_auc_score(y_all, (oofs * best_w).sum(axis=1)),
    (tests * best_w).sum(axis=1)
)

print(f"\n{'='*65}")
print(f"  개별: LGB={auc_lgb:.5f}  CAT={auc_cat:.5f}  XGB={auc_xgb:.5f}")
print(f"  기준선 (exp021 Val): {BASELINE}")
print(f"{'-'*65}")
best_method, best_auc, best_test = "", 0, None
for method, (auc, test_pred) in results.items():
    flag = " ← best" if auc > max(aucs) else ""
    diff = auc - BASELINE
    print(f"  {method:20s}: {auc:.5f}  ({diff:+.5f} vs exp021){flag}")
    if auc > best_auc:
        best_auc, best_method, best_test = auc, method, test_pred
print(f"{'='*65}")
print(f"\n최적 앙상블: {best_method}  OOF AUC={best_auc:.5f}")
print(f"Optuna 가중치  LGB={best_w[0]:.3f}  CAT={best_w[1]:.3f}  XGB={best_w[2]:.3f}")


# ════════════════════════════════════════════════════════════
# 제출 파일 저장
# ════════════════════════════════════════════════════════════

submission = pd.DataFrame({"ID": test_ids, "probability": best_test})
out_fname  = f"submission_exp{EXP_NO:03d}_{AUTHOR}.csv"
submission.to_csv(out_fname, index=False)
print(f"\n제출 파일 저장 완료: {out_fname}")
print(submission.head())


# ════════════════════════════════════════════════════════════
# 시각화
# ════════════════════════════════════════════════════════════

# 마지막 fold 모델의 feature importance 사용 (LGB)
feat_imp = pd.Series(lgb_model.feature_importances_, index=X_all.columns)

new_features = [
    # v3 기존
    "IVF_DI_시술_합산", "IVF_시술_비율", "IVF_DI_임신_합산", "IVF_임신_비율",
    "IVF_DI_출산_합산", "IVF_출산_비율", "시술_대비_임신_비율", "불임_원인_개수",
    "배아_사용_조합", "임신시도_결측여부", "이전_임신_여부", "이전_출산_여부",
    "전체_임신_성공률", "난자_배아_전환율", "고령_여부", "첫_시술_여부",
    "배아_이식_경과일_구간", "조기_이식_여부", "채취_이식_경과일_차이", "혼합_이식_경과일_차이",
    # 신규 추가
    "기증_난자_여부", "기증_정자_여부", "수정률", "이식률", "저장률", "ICSI_비율",
    "배아_발달일", "신선_시술_여부", "남성_불임_합계", "여성_불임_합계",
    "부부_불임_합계", "정자_문제_합계", "나이_시술횟수_상호작용", "신선_난자_저장_있음",
]

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Feature Importance Top 20
top20  = feat_imp.nlargest(20)
colors = ["tomato" if f in new_features else "steelblue" for f in top20.index]
top20.plot(kind="barh", ax=axes[0], color=colors)
axes[0].set_title(f"exp{EXP_NO:03d} Feature Importance Top 20")
axes[0].invert_yaxis()
axes[0].legend(handles=[
    Patch(facecolor="steelblue", label="Raw/v3 Feature"),
    Patch(facecolor="tomato",    label="New Feature (YJCho)"),
], loc="lower right")

# 앙상블 방법별 AUC 비교
method_names = list(results.keys())
method_aucs  = [results[m][0] for m in method_names]
bar_colors   = ["tomato" if m == best_method else "steelblue" for m in method_names]
axes[1].bar(method_names, method_aucs, color=bar_colors)
axes[1].set_ylim(min(method_aucs) - 0.001, max(method_aucs) + 0.002)
axes[1].axhline(BASELINE, color="gray", linestyle="--", label=f"exp021 baseline ({BASELINE})")
axes[1].set_title("앙상블 방법별 OOF AUC 비교")
axes[1].set_ylabel("OOF AUC")
axes[1].legend()
for i, (name, auc) in enumerate(zip(method_names, method_aucs)):
    axes[1].text(i, auc + 0.0003, f"{auc:.5f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(f"result_plots_exp{EXP_NO:03d}.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"그래프 저장: result_plots_exp{EXP_NO:03d}.png")
