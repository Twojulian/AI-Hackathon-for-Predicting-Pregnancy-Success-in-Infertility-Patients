# ════════════════════════════════════════════════════════════
# exp027 — 피처 선별 (importance 80위 이내만 사용)
#          전처리: exp026과 동일
#          모델: exp024 Optuna 튜닝된 LGB / CAT / XGB 앙상블
#          변경점: importance 81위 이하 피처 30개 제거
#          - 신규 피처 6개 제거 (시술횟수_구간, 시술유형_BLASTOCYST 등)
#          - 기존 피처 24개 제거 (불임_원인_개수, 정자_문제_합계 등)
# 기준선: exp026 (OOF 0.74016)
# ════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score,
                             recall_score, precision_score, accuracy_score)
from sklearn.preprocessing import LabelEncoder
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
EXP_NO   = 27_2
AUTHOR   = "SYJ"
BASELINE = 0.74021  # exp027 OOF AUC


# ════════════════════════════════════════════════════════════
# Best Params (exp024 Optuna 결과)
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
# exp026 importance 기준 71위 이하 제거 목록
# ════════════════════════════════════════════════════════════

DROP_BY_IMPORTANCE = [
    # 신규 피처 (6개)
    "시술횟수_구간",               # 86위
    "시술유형_BLASTOCYST",         # 88위
    "시술유형_IUI",                # 96위
    "시술유형_FER",                # 103위
    "시술유형_GIFT",               # 105위
    "시술유형_ICI",                # 111위
    # 기존 피처 (24개)
    "불임_원인_개수",               # 82위
    "부부 부 불임 원인",            # 83위
    "IVF_DI_임신_합산",            # 84위
    "불임 원인 - 자궁내막증",        # 85위
    "남성 부 불임 원인",            # 87위
    "여성 주 불임 원인",            # 89위
    "대리모 여부",                  # 90위
    "첫_시술_여부",                 # 91위
    "부부 주 불임 원인",            # 92위
    "기증 배아 사용 여부",           # 93위
    "남성 주 불임 원인",            # 94위
    "난자 혼합 경과일",             # 95위
    "불임 원인 - 정자 농도",        # 97위
    "저장된 신선 난자 수",           # 98위
    "불임 원인 - 정자 운동성",       # 99위
    "신선_난자_저장_있음",           # 100위
    "정자_문제_합계",               # 101위
    "불임 원인 - 자궁경부 문제",     # 102위
    "불임 원인 - 정자 면역학적 요인", # 104위
    "불임 원인 - 여성 요인",        # 106위
    "불임 원인 - 정자 형태",        # 107위
    "임신시도_결측여부",             # 108위
    "난자 채취 경과일",             # 109위
    "신선_시술_여부",               # 110위
        # exp028 추가 제거 71~80위 (9개)
    # "이전_임신_여부",           # 71위
    "DI 임신 횟수",             # 72위
    "착상 전 유전 진단 사용 여부", # 73위
    "동결 배아 사용 여부",        # 74위
    "IVF_DI_시술_합산",          # 75위
    # "총 시술 횟수",              # 76위
    "불명확 불임 원인",           # 77위
    "시술유형_ICSI",             # 78위 ★신규
    "시술유형_IVF",              # 79위 ★신규
    "기증자 정자와 혼합된 난자 수", # 80위
]


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
PROCEDURE_TYPES = ["IVF", "ICSI", "IUI", "FER", "BLASTOCYST", "GIFT", "ICI"]

label_encoders = {}


# ════════════════════════════════════════════════════════════
# 전처리 함수 (exp026과 동일)
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
    df[df.select_dtypes(include="object").columns] = df.select_dtypes(include="object").fillna("Unknown")

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

    # exp025 신규 피처 (전부 만들고 나중에 drop)
    if "시술 당시 나이" in df.columns and "불임_원인_개수" in df.columns:
        df["나이_불임원인_상호작용"] = df["시술 당시 나이"] * df["불임_원인_개수"]
    if "시술 당시 나이" in df.columns and "이전_임신_여부" in df.columns:
        df["나이_이전임신_상호작용"] = df["시술 당시 나이"] * df["이전_임신_여부"]
    if "총 시술 횟수" in df.columns:
        df["시술횟수_구간"] = pd.cut(
            df["총 시술 횟수"], bins=[-1, 1, 3, 999], labels=[0, 1, 2]
        ).astype(int)
    if "해동된 배아 수" in df.columns and "총 생성 배아 수" in df.columns:
        df["해동_배아_비율"] = df["해동된 배아 수"] / (df["총 생성 배아 수"] + eps)
    if "특정 시술 유형" in df.columns:
        for proc in PROCEDURE_TYPES:
            df[f"시술유형_{proc}"] = df["특정 시술 유형"].astype(str).str.contains(proc).astype(int)

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
# 데이터 로드
# ════════════════════════════════════════════════════════════

DATA_DIR = "/kaggle/input/datasets/yjsheila/infertility/"

train  = pd.read_csv(f"{DATA_DIR}/train.csv")
test   = pd.read_csv(f"{DATA_DIR}/test.csv")
sample = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

train_df, _       = preprocess(train, is_train=True)
test_df, test_ids = preprocess(test,  is_train=False)

X_all    = train_df.drop(TARGET, axis=1)
y_all    = train_df[TARGET]
X_submit = test_df.drop(columns=[TARGET], errors="ignore")

neg_pos_ratio = (y_all == 0).sum() / (y_all == 1).sum()

# ★ importance 기준 70위 이하 제거
drop_cols = [c for c in DROP_BY_IMPORTANCE if c in X_all.columns]
X_all    = X_all.drop(columns=drop_cols)
X_submit = X_submit.drop(columns=drop_cols)

lgb_drop     = [c for c in LGB_DROP_COLS if c in X_all.columns]
X_all_lgb    = X_all.drop(columns=lgb_drop)
X_submit_lgb = X_submit.drop(columns=lgb_drop)

print(f"제거된 피처 수 : {len(drop_cols)}개")
print(f"전체 피처 수   : {X_all.shape[1]}")
print(f"LGB 피처 수    : {X_all_lgb.shape[1]}")
print(f"scale_pos_weight: {neg_pos_ratio:.4f}")


# ════════════════════════════════════════════════════════════
# 파라미터에 공통 항목 추가
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
# OOF 학습 + test 예측
# ════════════════════════════════════════════════════════════

skf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_lgb  = np.zeros(len(X_all))
oof_cat  = np.zeros(len(X_all))
oof_xgb  = np.zeros(len(X_all))
test_lgb = np.zeros(len(X_submit))
test_cat = np.zeros(len(X_submit))
test_xgb = np.zeros(len(X_submit))

print(f"\n{'='*60}")
print(f"exp027 OOF 학습 (LGB + CAT + XGB, 5-Fold)")
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

    fold_lgb = roc_auc_score(y_val, oof_lgb[val_idx])
    fold_cat = roc_auc_score(y_val, oof_cat[val_idx])
    fold_xgb = roc_auc_score(y_val, oof_xgb[val_idx])
    print(f"  Fold {fold}  LGB={fold_lgb:.4f}  CAT={fold_cat:.4f}  XGB={fold_xgb:.4f}")

auc_lgb = roc_auc_score(y_all, oof_lgb)
auc_cat = roc_auc_score(y_all, oof_cat)
auc_xgb = roc_auc_score(y_all, oof_xgb)

print(f"\nOOF AUC  LGB={auc_lgb:.5f}  CAT={auc_cat:.5f}  XGB={auc_xgb:.5f}")


# ════════════════════════════════════════════════════════════
# 앙상블 비교
# ════════════════════════════════════════════════════════════

oofs  = np.stack([oof_lgb,  oof_cat,  oof_xgb],  axis=1)
tests = np.stack([test_lgb, test_cat, test_xgb], axis=1)
aucs  = np.array([auc_lgb, auc_cat, auc_xgb])

results = {}
results["Simple Average"] = (roc_auc_score(y_all, oofs.mean(axis=1)), oofs.mean(axis=1), tests.mean(axis=1))

w_auc = aucs / aucs.sum()
results["AUC-weighted"]   = (roc_auc_score(y_all, (oofs * w_auc).sum(axis=1)), (oofs * w_auc).sum(axis=1), (tests * w_auc).sum(axis=1))

oof_ranks  = np.stack([rankdata(oofs[:, i])  for i in range(3)], axis=1)
test_ranks = np.stack([rankdata(tests[:, i]) for i in range(3)], axis=1)
results["Rank Average"]   = (roc_auc_score(y_all, oof_ranks.mean(axis=1)), oof_ranks.mean(axis=1), test_ranks.mean(axis=1))

print(f"\n{'='*65}")
print(f"  개별: LGB={auc_lgb:.5f}  CAT={auc_cat:.5f}  XGB={auc_xgb:.5f}")
print(f"  기준선 (exp026): {BASELINE}")
print(f"{'-'*65}")
best_method, best_auc, best_oof_pred, best_test = "", 0, None, None
for method, (auc, oof_pred, test_pred) in results.items():
    diff = auc - BASELINE
    flag = " ← best" if auc > max(aucs) else ""
    print(f"  {method:20s}: {auc:.5f}  ({diff:+.5f} vs exp026){flag}")
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


# ════════════════════════════════════════════════════════════
# 실험 기록장용 요약 출력
# ════════════════════════════════════════════════════════════

oof_binary = (best_oof_pred >= 0.5).astype(int)

print("\n" + "="*55)
print("📋 실험 기록장 정보")
print("="*55)
print(f"실험 번호     : exp{EXP_NO:03d}")
print(f"모델명        : LGB + CAT + XGB 앙상블 ({best_method})")
print(f"제거 피처 수  : {len(drop_cols)}개 (importance 71위 이하)")
print(f"전체 피처 수  : {X_all.shape[1]}  /  LGB 피처 수: {X_all_lgb.shape[1]}")
print(f"LGB OOF AUC  : {auc_lgb:.5f}")
print(f"CAT OOF AUC  : {auc_cat:.5f}")
print(f"XGB OOF AUC  : {auc_xgb:.5f}")
print(f"앙상블 AUC   : {best_auc:.5f}  ({best_auc - BASELINE:+.5f} vs exp027)")
print(f"F1 Score     : {f1_score(y_all, oof_binary, average='macro'):.4f}")
print(f"Recall       : {recall_score(y_all, oof_binary, average='macro'):.4f}")
print(f"Precision    : {precision_score(y_all, oof_binary, average='macro'):.4f}")
print(f"Accuracy     : {accuracy_score(y_all, oof_binary):.4f}")
print(f"검증 방법     : Stratified {N_FOLDS}-Fold OOF")
print(f"클래스 불균형 : scale_pos_weight={neg_pos_ratio:.4f}")
print(f"파일명        : {out_fname}")
print(f"특이사항      : exp026 importance 기준 71위 이하 40개 피처 제거")
print(f"인사이트      : 피처 수 감소로 노이즈 제거 효과 및 AUC 변화 확인")
print("="*55)