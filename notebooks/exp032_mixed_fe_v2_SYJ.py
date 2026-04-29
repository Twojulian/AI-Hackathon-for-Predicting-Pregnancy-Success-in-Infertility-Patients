# ════════════════════════════════════════════════════════════
# exp032 — Mixed FE v2 (팀원 방식 적용)
#   ① LGB : FE-v1 (exp024 기준 피처)
#   ② CAT : FE-v2 + fold-wise Target Encoding
#   ③ XGB : FE-v2 + fold-wise TE + Interaction TE (ITE)
#   ④ TE/ITE를 fold 안에서 계산 → 리크 없음 (핵심 수정!)
#   ⑤ 앙상블: Rank Average + Optuna 가중치 비교
#
# 기준선: exp024 앙상블 OOF AUC 0.74025
# ════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score,
                            recall_score, precision_score, accuracy_score)
from sklearn.preprocessing import LabelEncoder
from scipy.stats import rankdata
import optuna
import os, matplotlib.pyplot as plt, matplotlib.font_manager as fm

os.system('apt-get install -y fonts-nanum -qq')
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED     = 42
N_FOLDS  = 5
TARGET   = "임신 성공 여부"
EXP_NO   = 32
AUTHOR   = "SYJ"
BASELINE = 0.74025  # exp024 앙상블 OOF AUC

# Optuna 앙상블 가중치 탐색 횟수 (빠름 — 1~2분)
OPTUNA_WEIGHT_TRIALS = 300

# TE / ITE 설정
TE_COLS = [
    "시술 시기 코드", "시술 유형", "특정 시술 유형",
    "배란 유도 유형", "난자 출처", "정자 출처",
    "시술 당시 나이", "총 시술 횟수", "총 임신 횟수",
]
TE_ALPHA = 10

ITE_PAIRS = [
    ("시술 당시 나이", "시술 유형"),
    ("시술 당시 나이", "특정 시술 유형"),
    ("시술 당시 나이", "난자 출처"),
    ("시술 당시 나이", "배란 유도 유형"),
    ("시술 유형",     "총 시술 횟수"),
    ("난자 출처",     "시술 유형"),
]
ITE_ALPHA = 20


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
    "is_unbalance": True,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}
BEST_CAT_PARAMS = {
    "iterations": 2000,
    "loss_function": "Logloss", "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
    "random_seed": SEED, "verbose": 0,
    "thread_count": -1, "early_stopping_rounds": 100,
    "learning_rate": 0.018758723768855998,
    "depth": 6,
    "l2_leaf_reg": 9.189608434163782,
    "min_data_in_leaf": 19,
    "subsample": 0.8170921295501483,
    "colsample_bylevel": 0.6936810336930781,
}
BEST_XGB_PARAMS = {
    "n_estimators": 2000,
    "eval_metric": "auc", "tree_method": "hist",
    "random_state": SEED, "n_jobs": -1,
    "verbosity": 0, "early_stopping_rounds": 100,
    "learning_rate": 0.05520069867907647,
    "max_depth": 4,
    "min_child_weight": 59,
    "subsample": 0.7663066457187595,
    "colsample_bytree": 0.6581836436885355,
    "reg_alpha": 8.692038126211928,
    "reg_lambda": 0.23932562420374562,
}


# ════════════════════════════════════════════════════════════
# 설정값 — exp024 기준
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
# 전처리
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


def preprocess_base(df, is_train=True):
    """공통 전처리 — LabelEncoding 전 단계"""
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

    for col in [c for c in LOG_COLS if c in df.columns]:
        df[col] = np.log1p(df[col])

    return df, ids


def add_fe_v1(df):
    """FE-v1 — LGB용 (exp024 기준 파생 피처)"""
    df = df.copy()
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
        df["임신 시도 또는 마지막 임신 경과 연수"] = df["임신 시도 또는 마지막 임신 경과 연수"].fillna(0)

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

    return df


def add_fe_v2(df):
    """FE-v2 — CAT/XGB용 (v1 + 4개 추가)"""
    df = add_fe_v1(df)
    eps = 1e-6
    if "총 임신 횟수" in df.columns and "총 시술 횟수" in df.columns:
        df["임신_성공률"] = df["총 임신 횟수"] / (df["총 시술 횟수"] + eps)
        df["시술_실패_횟수"] = (df["총 시술 횟수"] - df["총 임신 횟수"]).clip(lower=0)
    if "수집된 신선 난자 수" in df.columns:
        egg = df["수집된 신선 난자 수"]
        df["최적_난자수"] = ((egg >= 5) & (egg <= 15)).astype(int)
    if "시술 당시 나이" in df.columns and "이식된 배아 수" in df.columns:
        df["나이_이식배아수"] = df["시술 당시 나이"] * df["이식된 배아 수"].fillna(0)
    return df


def apply_label_encoding(df, is_train=True):
    """LabelEncoding — LGB용"""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        if col == TARGET:
            continue
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
    return df


# ════════════════════════════════════════════════════════════
# 데이터 로드 + 기본 전처리
# ════════════════════════════════════════════════════════════

DATA_DIR = "/kaggle/input/datasets/yjsheila/infertility/"

train  = pd.read_csv(f"{DATA_DIR}/train.csv")
test   = pd.read_csv(f"{DATA_DIR}/test.csv")

train_base, _,        = preprocess_base(train, is_train=True)
test_base,  test_ids  = preprocess_base(test,  is_train=False)

y_all         = train_base[TARGET]
neg_pos_ratio = (y_all == 0).sum() / (y_all == 1).sum()
skf           = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# ── LGB용: FE-v1 + LabelEncoding
train_v1 = add_fe_v1(train_base.drop(columns=[TARGET]))
test_v1  = add_fe_v1(test_base)
X_lgb_train = apply_label_encoding(train_v1, is_train=True)
X_lgb_test  = apply_label_encoding(test_v1,  is_train=False)
lgb_drop     = [c for c in LGB_DROP_COLS if c in X_lgb_train.columns]
X_lgb_train  = X_lgb_train.drop(columns=lgb_drop)
X_lgb_test   = X_lgb_test.drop(columns=lgb_drop)

# ── CAT/XGB용: FE-v2 베이스 (TE/ITE는 fold 안에서 적용)
train_v2 = add_fe_v2(train_base.drop(columns=[TARGET]))
test_v2  = add_fe_v2(test_base)

# TE/ITE 대상 컬럼 실제 존재 여부 필터
_te_cols   = [c for c in TE_COLS  if c in train_v2.columns]
_ite_pairs = [(c1, c2) for c1, c2 in ITE_PAIRS
              if c1 in train_v2.columns and c2 in train_v2.columns]

# CAT/XGB에서 남은 범주형 컬럼 LabelEncoding (TE 안 된 컬럼)
cat_label_encoders = {}
train_v2_enc = train_v2.copy()
test_v2_enc  = test_v2.copy()
for col in train_v2.select_dtypes(include="object").columns:
    le = LabelEncoder()
    train_v2_enc[col] = le.fit_transform(train_v2[col].astype(str))
    cat_label_encoders[col] = le
    test_v2_enc[col] = test_v2[col].astype(str).apply(
        lambda x: x if x in le.classes_ else "Unknown"
    )
    test_v2_enc[col] = le.transform(test_v2_enc[col])

print(f"LGB  피처: {X_lgb_train.shape[1]}개 (FE-v1)")
print(f"CAT  피처: {train_v2_enc.shape[1] + len(_te_cols)}개 (FE-v2 + TE {len(_te_cols)}개)")
print(f"XGB  피처: {train_v2_enc.shape[1] + len(_te_cols) + len(_ite_pairs)}개 (FE-v2 + TE + ITE {len(_ite_pairs)}개)")


# ════════════════════════════════════════════════════════════
# 파라미터 공통 항목 추가
# ════════════════════════════════════════════════════════════

BEST_XGB_PARAMS.update({"scale_pos_weight": neg_pos_ratio})


# ════════════════════════════════════════════════════════════
# OOF 학습 — 모델별 다른 피처셋
# ★ TE/ITE는 fold 안에서 train 데이터로만 계산 (리크 없음)
# ════════════════════════════════════════════════════════════

oof_lgb  = np.zeros(len(y_all))
oof_cat  = np.zeros(len(y_all))
oof_xgb  = np.zeros(len(y_all))
test_lgb = np.zeros(len(X_lgb_test))
test_cat = np.zeros(len(X_lgb_test))
test_xgb = np.zeros(len(X_lgb_test))

print(f"\n{'='*60}")
print(f"exp032 OOF 학습 (Mixed FE, 5-Fold)")
print(f"{'='*60}")

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_lgb_train, y_all), 1):
    y_tr, y_val   = y_all.iloc[tr_idx], y_all.iloc[val_idx]
    global_mean   = y_tr.mean()

    # ── LGB: FE-v1 ───────────────────────────────────────────
    lgb_model = lgb.LGBMClassifier(**BEST_LGB_PARAMS)
    lgb_model.fit(X_lgb_train.iloc[tr_idx], y_tr)
    oof_lgb[val_idx]  = lgb_model.predict_proba(X_lgb_train.iloc[val_idx])[:, 1]
    test_lgb         += lgb_model.predict_proba(X_lgb_test)[:, 1] / N_FOLDS

    # ── CAT: FE-v2 + fold-wise TE ────────────────────────────
    X_tr_cat  = train_v2_enc.iloc[tr_idx].copy()
    X_val_cat = train_v2_enc.iloc[val_idx].copy()
    X_te_cat  = test_v2_enc.copy()

    for col in _te_cols:
        grp      = y_tr.groupby(X_tr_cat[col])
        smoothed = (grp.sum() + TE_ALPHA * global_mean) / (grp.count() + TE_ALPHA)
        te_col   = f"{col}_te"
        X_tr_cat[te_col]  = X_tr_cat[col].map(smoothed).fillna(global_mean)
        X_val_cat[te_col] = X_val_cat[col].map(smoothed).fillna(global_mean)
        X_te_cat[te_col]  = X_te_cat[col].map(smoothed).fillna(global_mean)

    cat_model = CatBoostClassifier(**BEST_CAT_PARAMS)
    cat_model.fit(X_tr_cat, y_tr, eval_set=Pool(X_val_cat, y_val), use_best_model=True)
    oof_cat[val_idx]  = cat_model.predict_proba(X_val_cat)[:, 1]
    test_cat         += cat_model.predict_proba(X_te_cat)[:, 1] / N_FOLDS

    # ── XGB: FE-v2 + TE + ITE ────────────────────────────────
    X_tr_xgb  = X_tr_cat.copy()
    X_val_xgb = X_val_cat.copy()
    X_te_xgb  = X_te_cat.copy()

    for col1, col2 in _ite_pairs:
        key_tr  = X_tr_xgb[col1].astype(str)  + "_" + X_tr_xgb[col2].astype(str)
        key_val = X_val_xgb[col1].astype(str) + "_" + X_val_xgb[col2].astype(str)
        key_te  = X_te_xgb[col1].astype(str)  + "_" + X_te_xgb[col2].astype(str)
        grp      = y_tr.groupby(key_tr)
        smoothed = (grp.sum() + ITE_ALPHA * global_mean) / (grp.count() + ITE_ALPHA)
        ite_col  = f"{col1}_{col2}_ite"
        X_tr_xgb[ite_col]  = key_tr.map(smoothed).fillna(global_mean)
        X_val_xgb[ite_col] = key_val.map(smoothed).fillna(global_mean)
        X_te_xgb[ite_col]  = key_te.map(smoothed).fillna(global_mean)

    xgb_model = XGBClassifier(**BEST_XGB_PARAMS)
    xgb_model.fit(X_tr_xgb, y_tr, eval_set=[(X_val_xgb, y_val)], verbose=False)
    oof_xgb[val_idx]  = xgb_model.predict_proba(X_val_xgb)[:, 1]
    test_xgb         += xgb_model.predict_proba(X_te_xgb)[:, 1] / N_FOLDS

    print(f"  Fold {fold}  LGB={roc_auc_score(y_val, oof_lgb[val_idx]):.4f}"
          f"  CAT={roc_auc_score(y_val, oof_cat[val_idx]):.4f}"
          f"  XGB={roc_auc_score(y_val, oof_xgb[val_idx]):.4f}")

auc_lgb = roc_auc_score(y_all, oof_lgb)
auc_cat = roc_auc_score(y_all, oof_cat)
auc_xgb = roc_auc_score(y_all, oof_xgb)
print(f"\nOOF AUC  LGB={auc_lgb:.5f}  CAT={auc_cat:.5f}  XGB={auc_xgb:.5f}")


# ════════════════════════════════════════════════════════════
# 앙상블 비교 + Optuna 가중치 탐색
# ════════════════════════════════════════════════════════════

oofs  = np.stack([oof_lgb,  oof_cat,  oof_xgb],  axis=1)
tests = np.stack([test_lgb, test_cat, test_xgb], axis=1)
aucs  = np.array([auc_lgb, auc_cat, auc_xgb])

def rank_avg(arr):
    return np.stack([rankdata(arr[:, i]) for i in range(arr.shape[1])], axis=1).mean(axis=1)

def rank_normalize(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)

# Optuna 앙상블 가중치 탐색
print(f"\n[Optuna 앙상블 가중치 탐색] {OPTUNA_WEIGHT_TRIALS}회...")
def optuna_weight_obj(trial):
    w = np.array([
        trial.suggest_float("w_lgb", 0.0, 1.0),
        trial.suggest_float("w_cat", 0.0, 1.0),
        trial.suggest_float("w_xgb", 0.0, 1.0),
    ])
    w = w / w.sum()
    return roc_auc_score(y_all, (oofs * w).sum(axis=1))

study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(optuna_weight_obj, n_trials=OPTUNA_WEIGHT_TRIALS, show_progress_bar=False)
best_w = np.array([study.best_params["w_lgb"],
                    study.best_params["w_cat"],
                    study.best_params["w_xgb"]])
best_w = best_w / best_w.sum()
print(f"  최적 가중치: LGB={best_w[0]:.3f}  CAT={best_w[1]:.3f}  XGB={best_w[2]:.3f}")

w_auc = aucs / aucs.sum()
results = {
    "Simple Average":  (roc_auc_score(y_all, oofs.mean(axis=1)),
                        oofs.mean(axis=1), tests.mean(axis=1)),
    "AUC-weighted":    (roc_auc_score(y_all, (oofs * w_auc).sum(axis=1)),
                        (oofs * w_auc).sum(axis=1), (tests * w_auc).sum(axis=1)),
    "Rank Average":    (roc_auc_score(y_all, rank_avg(oofs)),
                        rank_avg(oofs), rank_normalize(rank_avg(tests))),
    "Optuna Weights":  (roc_auc_score(y_all, (oofs * best_w).sum(axis=1)),
                        (oofs * best_w).sum(axis=1), (tests * best_w).sum(axis=1)),
}

print(f"\n{'='*65}")
print(f"  개별: LGB={auc_lgb:.5f}  CAT={auc_cat:.5f}  XGB={auc_xgb:.5f}")
print(f"  기준선 (exp024): {BASELINE}")
print(f"{'-'*65}")
best_method, best_auc, best_oof_pred, best_test = "", 0, None, None
for method, (auc, oof_pred, test_pred) in results.items():
    diff = auc - BASELINE
    flag = " ← best" if auc == max(r[0] for r in results.values()) else ""
    print(f"  {method:20s}: {auc:.5f}  ({diff:+.5f} vs exp024){flag}")
    if auc > best_auc:
        best_auc, best_method, best_oof_pred, best_test = auc, method, oof_pred, test_pred
print(f"{'='*65}")
print(f"\n최적 앙상블: {best_method}  OOF AUC={best_auc:.5f}")


# ════════════════════════════════════════════════════════════
# 제출 파일 저장
# ════════════════════════════════════════════════════════════

out_fname  = f"submission_exp{EXP_NO:03d}_{AUTHOR}.csv"
submission = pd.DataFrame({"ID": test_ids, "probability": best_test})
submission.to_csv(out_fname, index=False)
print(f"\n제출 파일 저장 완료: {out_fname}")
print(f"  probability 범위: [{best_test.min():.4f}, {best_test.max():.4f}]  ← 0~1 확인")


# ════════════════════════════════════════════════════════════
# 실험 기록장
# ════════════════════════════════════════════════════════════

oof_binary = (best_oof_pred >= np.percentile(best_oof_pred, 70)).astype(int)

print("\n" + "="*55)
print("📋 실험 기록장 정보")
print("="*55)
print(f"실험 번호     : exp{EXP_NO:03d}")
print(f"모델명        : LGB(FE-v1) + CAT(FE-v2+TE) + XGB(FE-v2+TE+ITE)")
print(f"앙상블 방법   : {best_method}")
print(f"LGB 피처 수  : {X_lgb_train.shape[1]}개 (FE-v1)")
print(f"CAT 피처 수  : {train_v2_enc.shape[1] + len(_te_cols)}개 (FE-v2 + TE {len(_te_cols)}개)")
print(f"XGB 피처 수  : {train_v2_enc.shape[1] + len(_te_cols) + len(_ite_pairs)}개 (FE-v2 + TE + ITE {len(_ite_pairs)}개)")
print(f"TE 컬럼      : {_te_cols}")
print(f"ITE 쌍       : {_ite_pairs}")
print(f"LGB OOF AUC : {auc_lgb:.5f}")
print(f"CAT OOF AUC : {auc_cat:.5f}")
print(f"XGB OOF AUC : {auc_xgb:.5f}")
print(f"앙상블 AUC  : {best_auc:.5f}  ({best_auc - BASELINE:+.5f} vs exp024)")
print(f"Optuna 가중치: LGB={best_w[0]:.3f}  CAT={best_w[1]:.3f}  XGB={best_w[2]:.3f}")
print(f"F1 Score    : {f1_score(y_all, oof_binary, average='macro'):.4f}")
print(f"Recall      : {recall_score(y_all, oof_binary, average='macro'):.4f}")
print(f"Precision   : {precision_score(y_all, oof_binary, average='macro'):.4f}")
print(f"Accuracy    : {accuracy_score(y_all, oof_binary):.4f}")
print(f"검증 방법    : Stratified {N_FOLDS}-Fold OOF")
print(f"클래스 불균형: scale_pos_weight={neg_pos_ratio:.4f}")
print(f"파일명       : {out_fname}")
print(f"특이사항     :")
print(f"  - TE/ITE fold 안에서 계산 → 리크 없음 (exp031 수정)")
print(f"  - ITE: 두 컬럼 조합 문자열에 TE 적용 (단순 곱셈 아님)")
print(f"  - FE-v2: v1 + 임신_성공률/시술_실패_횟수/최적_난자수/나이_이식배아수")
print(f"  - Optuna 앙상블 가중치 {OPTUNA_WEIGHT_TRIALS}회 탐색 추가")
print("="*55)
