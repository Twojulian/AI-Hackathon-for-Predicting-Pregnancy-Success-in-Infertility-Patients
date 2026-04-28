# ════════════════════════════════════════════════════════════
# exp029 — Stacking (Base: LGB + CAT + XGB → Meta: Ridge)
#   ① Base 모델: exp024 Optuna 최적 파라미터 그대로
#   ② 피처: exp024 기준 (102개, 신규 5종 제거)
#   ③ Meta 모델: Ridge Regression (과적합 방지)
#   ④ Stacking 방식: OOF 예측값 → Ridge 입력
#      (옵션) USE_ORIGINAL_FEATURES = True 로 바꾸면
#             OOF 예측값 + 원본 피처 같이 Meta에 넣기
#
# 기준선: exp024 앙상블 OOF AUC 0.74025
# ════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score,
                             recall_score, precision_score, accuracy_score)
from sklearn.preprocessing import LabelEncoder
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

os.system('apt-get install -y fonts-nanum -qq')
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")

SEED     = 42
N_FOLDS  = 5
TARGET   = "임신 성공 여부"
EXP_NO   = 29
AUTHOR   = "SYJ"
BASELINE = 0.74025  # exp024 앙상블 OOF AUC

# ════════════════════════════════════════════════════════════
# ★ 옵션: 메타 모델에 원본 피처도 같이 넣을지 여부
#   False → OOF 예측값 3개만 (순수 Stacking)
#   True  → OOF 예측값 3개 + 원본 피처 전체 (Feature-weighted Stacking)
# ════════════════════════════════════════════════════════════
USE_ORIGINAL_FEATURES = False


# ════════════════════════════════════════════════════════════
# exp024 Optuna 최적 파라미터
# ════════════════════════════════════════════════════════════

BEST_LGB_PARAMS = {
    "n_estimators": 687,
    "learning_rate": 0.06728712035169694,
    "num_leaves": 272,
    "max_depth": 3,
    "min_child_samples": 62,
    "subsample": 0.7104765923920849,
    "colsample_bytree": 0.5598759562284701,
    "reg_alpha": 7.8646544609636635,
    "reg_lambda": 3.5299647925886912,
}
BEST_CAT_PARAMS = {
    "iterations": 2040,
    "learning_rate": 0.011818212764898741,
    "depth": 6,
    "l2_leaf_reg": 7.461945627619095,
    "min_data_in_leaf": 10,
    "subsample": 0.9466415028340966,
    "colsample_bylevel": 0.9498475991985358,
}
BEST_XGB_PARAMS = {
    "n_estimators": 644,
    "learning_rate": 0.019039448776941068,
    "max_depth": 5,
    "min_child_weight": 46,
    "subsample": 0.8977566603325766,
    "colsample_bytree": 0.5960202930510033,
    "reg_alpha": 0.0022796242865722625,
    "reg_lambda": 0.08546935668674432,
}


# ════════════════════════════════════════════════════════════
# 설정값 — exp024 기준 (신규 5종 파생 피처 없음)
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
DI_ZERO_COLS = [
    "총 생성 배아 수", "미세주입에서 생성된 배아 수", "저장된 배아 수",
    "미세주입된 난자 수", "수집된 신선 난자 수",
]
LGB_DROP_COLS = ["총 생성 배아 수", "수집된 신선 난자 수"]

MALE_COLS   = ["남성 주 불임 원인", "남성 부 불임 원인", "불임 원인 - 남성 요인"]
FEMALE_COLS = ["여성 주 불임 원인", "여성 부 불임 원인", "불임 원인 - 난관 질환",
               "불임 원인 - 배란 장애", "불임 원인 - 자궁내막증", "불임 원인 - 자궁경부 문제"]
COUPLE_COLS = ["부부 주 불임 원인", "부부 부 불임 원인"]
SPERM_COLS  = ["불임 원인 - 정자 농도", "불임 원인 - 정자 운동성",
               "불임 원인 - 정자 형태", "불임 원인 - 정자 면역학적 요인"]

label_encoders = {}


# ════════════════════════════════════════════════════════════
# 전처리 — exp024 기준
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

    if "시술 유형" in df.columns:
        di_mask = df["시술 유형"] == "DI"
        for col in DI_ZERO_COLS:
            if col in df.columns:
                df.loc[di_mask, col] = 0
                median_val = df.loc[~di_mask, col].median() if is_train else 0
                df.loc[~di_mask, col] = df.loc[~di_mask, col].fillna(median_val)
        df["is_DI"] = di_mask.astype(int)

    num_cols = [c for c in df.select_dtypes(include="number").columns if c != TARGET]
    df[num_cols] = df[num_cols].fillna(0)
    df[df.select_dtypes(include="object").columns] = \
        df.select_dtypes(include="object").fillna("Unknown")

    for col in COUNT_COLS:
        if col in df.columns:
            df[col] = df[col].apply(convert_count)

    if "시술 당시 나이" in df.columns:
        df["시술 당시 나이"] = df["시술 당시 나이"].map(AGE_MAP).fillna(36)
    for col in ["난자 기증자 나이", "정자 기증자 나이"]:
        if col in df.columns:
            df[col] = df[col].map(DONOR_AGE_MAP).fillna(0)

    eps = 1e-6

    if "IVF 시술 횟수" in df.columns and "DI 시술 횟수" in df.columns:
        df["IVF_DI_시술_합산"] = df["IVF 시술 횟수"] + df["DI 시술 횟수"]
        df["IVF_시술_비율"]    = df["IVF 시술 횟수"] / (df["IVF_DI_시술_합산"] + eps)
    if "IVF 임신 횟수" in df.columns and "DI 임신 횟수" in df.columns:
        df["IVF_DI_임신_합산"] = df["IVF 임신 횟수"] + df["DI 임신 횟수"]
        df["IVF_임신_비율"]    = df["IVF 임신 횟수"] / (df["IVF_DI_임신_합산"] + eps)
    if "IVF 출산 횟수" in df.columns and "DI 출산 횟수" in df.columns:
        df["IVF_DI_출산_합산"] = df["IVF 출산 횟수"] + df["DI 출산 횟수"]
        df["IVF_출산_비율"]    = df["IVF 출산 횟수"] / (df["IVF_DI_출산_합산"] + eps)
    if "IVF_DI_시술_합산" in df.columns and "IVF_DI_임신_합산" in df.columns:
        df["시술_대비_임신_비율"] = df["IVF_DI_임신_합산"] / (df["IVF_DI_시술_합산"] + eps)

    cause_exist = [c for c in CAUSE_COLS if c in df.columns]
    if cause_exist:
        df["불임_원인_개수"] = df[cause_exist].sum(axis=1)

    if all(c in df.columns for c in ["동결 배아 사용 여부", "신선 배아 사용 여부", "기증 배아 사용 여부"]):
        df["배아_사용_조합"] = (
            df["동결 배아 사용 여부"].fillna(0).astype(int).astype(str) +
            df["신선 배아 사용 여부"].fillna(0).astype(int).astype(str) +
            df["기증 배아 사용 여부"].fillna(0).astype(int).astype(str)
        )

    if "임신 시도 또는 마지막 임신 경과 연수" in df.columns:
        df["임신 시도 또는 마지막 임신 경과 연수"] = pd.to_numeric(
            df["임신 시도 또는 마지막 임신 경과 연수"], errors="coerce"
        )
        df["임신시도_결측여부"] = df["임신 시도 또는 마지막 임신 경과 연수"].isnull().astype(int)
        df["임신 시도 또는 마지막 임신 경과 연수"] = df["임신 시도 또는 마지막 임신 경과 연수"].fillna(
            df["임신 시도 또는 마지막 임신 경과 연수"].median() if is_train else 0
        )

    if "총 임신 횟수" in df.columns:
        df["이전_임신_여부"] = (df["총 임신 횟수"] > 0).astype(int)
    if "총 출산 횟수" in df.columns:
        df["이전_출산_여부"] = (df["총 출산 횟수"] > 0).astype(int)
    if "총 임신 횟수" in df.columns and "총 시술 횟수" in df.columns:
        df["전체_임신_성공률"] = df["총 임신 횟수"] / (df["총 시술 횟수"] + eps)
    if "수집된 신선 난자 수" in df.columns and "총 생성 배아 수" in df.columns:
        df["난자_배아_전환율"] = df["총 생성 배아 수"] / (df["수집된 신선 난자 수"] + eps)
    if "시술 당시 나이" in df.columns:
        df["고령_여부"] = (df["시술 당시 나이"] >= 38).astype(int)
    if "총 시술 횟수" in df.columns:
        df["첫_시술_여부"] = (df["총 시술 횟수"] == 1).astype(int)
    if "배아 이식 경과일" in df.columns:
        df["배아_이식_경과일_구간"] = pd.cut(
            df["배아 이식 경과일"], bins=[-1, 2, 5, 9999], labels=[0, 1, 2]
        ).astype(int)
        df["조기_이식_여부"] = (df["배아 이식 경과일"] <= 3).astype(int)
    if "난자 채취 경과일" in df.columns and "배아 이식 경과일" in df.columns:
        df["채취_이식_경과일_차이"] = df["배아 이식 경과일"] - df["난자 채취 경과일"]
    if "난자 혼합 경과일" in df.columns and "배아 이식 경과일" in df.columns:
        df["혼합_이식_경과일_차이"] = df["배아 이식 경과일"] - df["난자 혼합 경과일"]
    if "난자 출처" in df.columns:
        df["기증_난자_여부"] = (df["난자 출처"] == "기증 제공").astype(int)
    if "정자 출처" in df.columns:
        df["기증_정자_여부"] = df["정자 출처"].isin(["기증 제공", "배우자 및 기증 제공"]).astype(int)
    if "총 생성 배아 수" in df.columns and "혼합된 난자 수" in df.columns:
        df["수정률"] = df["총 생성 배아 수"] / (df["혼합된 난자 수"] + eps)
    if "이식된 배아 수" in df.columns and "총 생성 배아 수" in df.columns:
        df["이식률"] = df["이식된 배아 수"] / (df["총 생성 배아 수"] + eps)
    if "저장된 배아 수" in df.columns and "총 생성 배아 수" in df.columns:
        df["저장률"] = df["저장된 배아 수"] / (df["총 생성 배아 수"] + eps)
    if "미세주입된 난자 수" in df.columns and "혼합된 난자 수" in df.columns:
        df["ICSI_비율"] = df["미세주입된 난자 수"] / (df["혼합된 난자 수"] + eps)
    if "총 생성 배아 수" in df.columns and "수집된 신선 난자 수" in df.columns:
        df["배아_생성_효율"] = df["총 생성 배아 수"] / (df["수집된 신선 난자 수"] + eps)
    if "이식된 배아 수" in df.columns and "총 생성 배아 수" in df.columns:
        df["이식_효율"] = df["이식된 배아 수"] / (df["총 생성 배아 수"] + eps)
    if "배아 이식 경과일" in df.columns and "난자 혼합 경과일" in df.columns:
        df["배아_발달일"] = df["배아 이식 경과일"] - df["난자 혼합 경과일"]
    if "수집된 신선 난자 수" in df.columns:
        df["신선_시술_여부"] = df["수집된 신선 난자 수"].notna().astype(int)

    male_exist   = [c for c in MALE_COLS   if c in df.columns]
    female_exist = [c for c in FEMALE_COLS if c in df.columns]
    couple_exist = [c for c in COUPLE_COLS if c in df.columns]
    sperm_exist  = [c for c in SPERM_COLS  if c in df.columns]
    if male_exist:   df["남성_불임_합계"] = df[male_exist].sum(axis=1)
    if female_exist: df["여성_불임_합계"] = df[female_exist].sum(axis=1)
    if couple_exist: df["부부_불임_합계"] = df[couple_exist].sum(axis=1)
    if sperm_exist:  df["정자_문제_합계"] = df[sperm_exist].sum(axis=1)

    if "시술 당시 나이" in df.columns and "총 시술 횟수" in df.columns:
        df["나이_시술횟수_상호작용"] = df["시술 당시 나이"] * df["총 시술 횟수"]
    if "저장된 신선 난자 수" in df.columns:
        df["신선_난자_저장_있음"] = (df["저장된 신선 난자 수"] > 0).astype(int)

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
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in le.classes_ else "Unknown"
                )
                df[col] = le.transform(df[col])
            else:
                df[col] = 0

    return df, ids


# ════════════════════════════════════════════════════════════
# 데이터 로드
# ════════════════════════════════════════════════════════════

DATA_DIR = "/kaggle/input/datasets/yjsheila/infertility/"

train  = pd.read_csv(f"{DATA_DIR}/train.csv")
test   = pd.read_csv(f"{DATA_DIR}/test.csv")

train_df, _       = preprocess(train, is_train=True)
test_df, test_ids = preprocess(test,  is_train=False)

X_all    = train_df.drop(TARGET, axis=1)
y_all    = train_df[TARGET]
X_submit = test_df.drop(columns=[TARGET], errors="ignore")

neg_pos_ratio = (y_all == 0).sum() / (y_all == 1).sum()

lgb_drop     = [c for c in LGB_DROP_COLS if c in X_all.columns]
X_all_lgb    = X_all.drop(columns=lgb_drop)
X_submit_lgb = X_submit.drop(columns=lgb_drop)

print(f"전체 피처 수  : {X_all.shape[1]}")
print(f"LGB 피처 수   : {X_all_lgb.shape[1]}")
print(f"scale_pos_weight: {neg_pos_ratio:.4f}")


# ════════════════════════════════════════════════════════════
# 파라미터 공통 항목 추가
# ════════════════════════════════════════════════════════════

BEST_LGB_PARAMS.update({
    "scale_pos_weight": neg_pos_ratio,
    "random_state": SEED, "n_jobs": -1, "verbose": -1,
})
BEST_CAT_PARAMS.update({
    "scale_pos_weight": neg_pos_ratio,
    "random_seed": SEED, "verbose": 0,
    "eval_metric": "AUC", "thread_count": -1,
})
BEST_XGB_PARAMS.update({
    "scale_pos_weight": neg_pos_ratio,
    "random_state": SEED, "n_jobs": -1,
    "verbosity": 0, "eval_metric": "logloss",
})


# ════════════════════════════════════════════════════════════
# Stage 1 — Base 모델 OOF 예측
# ════════════════════════════════════════════════════════════

skf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_lgb  = np.zeros(len(X_all))
oof_cat  = np.zeros(len(X_all))
oof_xgb  = np.zeros(len(X_all))
test_lgb = np.zeros(len(X_submit))
test_cat = np.zeros(len(X_submit))
test_xgb = np.zeros(len(X_submit))

print(f"\n{'='*60}")
print(f"[Stage 1] Base 모델 OOF 학습 (LGB + CAT + XGB, 5-Fold)")
print(f"{'='*60}")

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_all), 1):
    X_tr,     X_val     = X_all.iloc[tr_idx],     X_all.iloc[val_idx]
    X_tr_lgb, X_val_lgb = X_all_lgb.iloc[tr_idx], X_all_lgb.iloc[val_idx]
    y_tr, y_val         = y_all.iloc[tr_idx],      y_all.iloc[val_idx]

    lgb_model = lgb.LGBMClassifier(**BEST_LGB_PARAMS)
    lgb_model.fit(X_tr_lgb, y_tr)
    oof_lgb[val_idx]  = lgb_model.predict_proba(X_val_lgb)[:, 1]
    test_lgb         += lgb_model.predict_proba(X_submit_lgb)[:, 1] / N_FOLDS

    cat_model = CatBoostClassifier(**BEST_CAT_PARAMS)
    cat_model.fit(X_tr, y_tr)
    oof_cat[val_idx]  = cat_model.predict_proba(X_val)[:, 1]
    test_cat         += cat_model.predict_proba(X_submit)[:, 1] / N_FOLDS

    xgb_model = XGBClassifier(**BEST_XGB_PARAMS)
    xgb_model.fit(X_tr, y_tr)
    oof_xgb[val_idx]  = xgb_model.predict_proba(X_val)[:, 1]
    test_xgb         += xgb_model.predict_proba(X_submit)[:, 1] / N_FOLDS

    print(f"  Fold {fold}  LGB={roc_auc_score(y_val, oof_lgb[val_idx]):.4f}"
          f"  CAT={roc_auc_score(y_val, oof_cat[val_idx]):.4f}"
          f"  XGB={roc_auc_score(y_val, oof_xgb[val_idx]):.4f}")

auc_lgb = roc_auc_score(y_all, oof_lgb)
auc_cat = roc_auc_score(y_all, oof_cat)
auc_xgb = roc_auc_score(y_all, oof_xgb)
print(f"\nBase OOF AUC  LGB={auc_lgb:.5f}  CAT={auc_cat:.5f}  XGB={auc_xgb:.5f}")


# ════════════════════════════════════════════════════════════
# Stage 2 — Ridge 메타 모델
# ════════════════════════════════════════════════════════════

# 메타 피처 구성
oof_meta  = np.stack([oof_lgb,  oof_cat,  oof_xgb],  axis=1)  # (N, 3)
test_meta = np.stack([test_lgb, test_cat, test_xgb], axis=1)  # (M, 3)

if USE_ORIGINAL_FEATURES:
    # OOF 예측값 3개 + 원본 피처 전체
    oof_meta  = np.hstack([oof_meta,  X_all.values])
    test_meta = np.hstack([test_meta, X_submit.values])
    print(f"\n[메타 피처] OOF 3개 + 원본 {X_all.shape[1]}개 = {oof_meta.shape[1]}개")
else:
    print(f"\n[메타 피처] OOF 3개만 (순수 Stacking)")

# Ridge는 확률값을 직접 못 뱉으므로 LogisticRegression(C 역할 = Ridge) 사용
# → penalty='l2' + solver='lbfgs' = Ridge Regression과 동일한 정규화 효과
print(f"\n{'='*60}")
print(f"[Stage 2] Ridge 메타 모델 학습 (5-Fold OOF)")
print(f"{'='*60}")

# C값 후보 탐색 (작을수록 강한 정규화 = Ridge 강도)
C_candidates = [0.001, 0.01, 0.1, 1.0, 10.0]
best_C, best_meta_auc = 0.01, 0

for C in C_candidates:
    oof_ridge = np.zeros(len(X_all))
    for tr_idx, val_idx in skf.split(oof_meta, y_all):
        meta_tr, meta_val = oof_meta[tr_idx], oof_meta[val_idx]
        y_tr, y_val       = y_all.iloc[tr_idx], y_all.iloc[val_idx]
        ridge = LogisticRegression(C=C, penalty="l2", solver="lbfgs",
                                   max_iter=1000, random_state=SEED)
        ridge.fit(meta_tr, y_tr)
        oof_ridge[val_idx] = ridge.predict_proba(meta_val)[:, 1]
    auc = roc_auc_score(y_all, oof_ridge)
    print(f"  C={C:<8}  OOF AUC={auc:.5f}")
    if auc > best_meta_auc:
        best_meta_auc, best_C = auc, C

print(f"\n  최적 C={best_C}  →  메타 AUC={best_meta_auc:.5f}")

# 최적 C로 최종 Ridge 학습
oof_final  = np.zeros(len(X_all))
test_final = np.zeros(len(X_submit))

for tr_idx, val_idx in skf.split(oof_meta, y_all):
    meta_tr, meta_val = oof_meta[tr_idx], oof_meta[val_idx]
    y_tr              = y_all.iloc[tr_idx]
    ridge = LogisticRegression(C=best_C, penalty="l2", solver="lbfgs",
                               max_iter=1000, random_state=SEED)
    ridge.fit(meta_tr, y_tr)
    oof_final[val_idx]  = ridge.predict_proba(meta_val)[:, 1]
    test_final         += ridge.predict_proba(test_meta)[:, 1] / N_FOLDS

final_auc = roc_auc_score(y_all, oof_final)

# Rank Average 결과도 같이 비교
def rank_normalize(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)

oof_rank  = np.stack([rankdata(oof_meta[:, i])  for i in range(3)], axis=1).mean(axis=1)
test_rank = np.stack([rankdata(test_meta[:, i]) for i in range(3)], axis=1).mean(axis=1)
rank_auc  = roc_auc_score(y_all, oof_rank)

print(f"\n{'='*65}")
print(f"  Base 개별: LGB={auc_lgb:.5f}  CAT={auc_cat:.5f}  XGB={auc_xgb:.5f}")
print(f"  기준선 (exp024 Rank Average): {BASELINE}")
print(f"{'-'*65}")
print(f"  Rank Average (비교용)  : {rank_auc:.5f}  ({rank_auc-BASELINE:+.5f} vs exp024)")
print(f"  Ridge Stacking (C={best_C}): {final_auc:.5f}  ({final_auc-BASELINE:+.5f} vs exp024)")
print(f"{'='*65}")

best_pred = test_final if final_auc >= rank_auc else rank_normalize(test_rank)
best_auc  = max(final_auc, rank_auc)
best_name = f"Ridge Stacking(C={best_C})" if final_auc >= rank_auc else "Rank Average"
print(f"\n최종 선택: {best_name}  OOF AUC={best_auc:.5f}")


# ════════════════════════════════════════════════════════════
# 제출 파일 저장
# ════════════════════════════════════════════════════════════

out_fname  = f"submission_exp{EXP_NO:03d}_{AUTHOR}.csv"
submission = pd.DataFrame({"ID": test_ids, "probability": best_pred})
submission.to_csv(out_fname, index=False)
print(f"\n제출 파일 저장 완료: {out_fname}")
print(f"  probability 범위: [{best_pred.min():.4f}, {best_pred.max():.4f}]  ← 0~1 확인")


# ════════════════════════════════════════════════════════════
# 실험 기록장
# ════════════════════════════════════════════════════════════

oof_binary = (oof_final >= np.percentile(oof_final, 70)).astype(int)

print("\n" + "="*55)
print("📋 실험 기록장 정보")
print("="*55)
print(f"실험 번호     : exp{EXP_NO:03d}")
print(f"모델명        : LGB + CAT + XGB → Ridge Stacking")
print(f"메타 모델     : LogisticRegression(L2) C={best_C} / USE_ORIGINAL_FEATURES={USE_ORIGINAL_FEATURES}")
print(f"전체 피처 수  : {X_all.shape[1]}  /  LGB 피처 수: {X_all_lgb.shape[1]}")
print(f"Base LGB AUC : {auc_lgb:.5f}")
print(f"Base CAT AUC : {auc_cat:.5f}")
print(f"Base XGB AUC : {auc_xgb:.5f}")
print(f"Ridge Stack  : {final_auc:.5f}  ({final_auc-BASELINE:+.5f} vs exp024)")
print(f"Rank Average : {rank_auc:.5f}  ({rank_auc-BASELINE:+.5f} vs exp024)")
print(f"최종 선택    : {best_name}  AUC={best_auc:.5f}")
print(f"F1 Score     : {f1_score(y_all, oof_binary, average='macro'):.4f}")
print(f"검증 방법     : Stratified {N_FOLDS}-Fold OOF (Base + Meta 모두)")
print(f"클래스 불균형 : scale_pos_weight={neg_pos_ratio:.4f}")
print(f"파일명        : {out_fname}")
print(f"특이사항      :")
print(f"  - 피처 구성: exp024 기준 102개 (신규 5종 없음)")
print(f"  - Base: exp024 Optuna 파라미터 그대로")
print(f"  - Meta: Ridge(L2) C 탐색 {C_candidates}")
print(f"  - USE_ORIGINAL_FEATURES={USE_ORIGINAL_FEATURES}")
print(f"인사이트      : exp024 Rank Average 대비 Stacking 효과 확인 목적")
print("="*55)
