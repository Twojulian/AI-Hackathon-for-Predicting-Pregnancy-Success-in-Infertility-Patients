# ════════════════════════════════════════════════════════════
# exp031 — Mixed FE (모델마다 다른 피처셋)
#   ① LGB : exp024 기준 102개 → importance 하위 제거 (자동 계산)
#   ② CAT : exp024 기준 102개 + OOF Target Encoding
#   ③ XGB : exp024 기준 102개 + OOF Target Encoding + 상호작용 TE 피처
#   ④ 앙상블: Rank Average
#
# 기준선: exp024 앙상블 OOF AUC 0.74025
# ════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score,
                             recall_score, precision_score, accuracy_score)
from sklearn.preprocessing import LabelEncoder
from scipy.stats import rankdata
import os, matplotlib.pyplot as plt, matplotlib.font_manager as fm

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
EXP_NO   = 31
AUTHOR   = "SYJ"
BASELINE = 0.74025  # exp024 앙상블 OOF AUC

# ════════════════════════════════════════════════════════════
# ★ LGB importance 기반 피처 선택 — 상위 몇 개 쓸지
#   exp027에서 81개가 최적이었으므로 기본값 81
# ════════════════════════════════════════════════════════════
LGB_TOP_N = 81

# ════════════════════════════════════════════════════════════
# ★ XGB 상호작용 TE 피처 조합
#   중요도 높은 범주형 컬럼끼리 조합 (TE 적용 후 곱셈)
# ════════════════════════════════════════════════════════════
TE_INTERACTION_PAIRS = [
    ("시술 유형", "배아 이식 경과일"),
    ("시술 유형", "시술 당시 나이"),
    ("난자 출처", "시술 유형"),
    ("정자 출처", "시술 유형"),
]


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
# 전처리 — exp024 기준 (범주형 컬럼 목록 따로 반환)
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
    """LabelEncoder 전 단계까지 전처리. 범주형 컬럼 목록 반환."""
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
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    df[obj_cols] = df[obj_cols].fillna("Unknown")

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

    # 범주형 컬럼 목록 저장 (TE용)
    cat_cols = [c for c in df.select_dtypes(include="object").columns if c != TARGET]

    return df, ids, cat_cols


def apply_label_encoding(df, is_train=True):
    """LabelEncoder 적용 — LGB용"""
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
# ★ OOF Target Encoding
# ════════════════════════════════════════════════════════════

class TargetEncoder:
    def __init__(self, smoothing=10):
        self.smoothing    = smoothing
        self.global_mean_ = {}
        self.cat_means_   = {}

    def fit_transform_train(self, df, y, cat_cols, skf):
        df = df.copy()
        global_mean = y.mean()
        for col in cat_cols:
            if col not in df.columns:
                continue
            self.global_mean_[col] = global_mean
            encoded = np.zeros(len(df))
            for tr_idx, val_idx in skf.split(df, y):
                tr_mean = y.iloc[tr_idx].mean()
                stats   = y.iloc[tr_idx].groupby(df[col].iloc[tr_idx]).agg(["sum", "count"])
                smooth  = (stats["sum"] + self.smoothing * tr_mean) / (stats["count"] + self.smoothing)
                encoded[val_idx] = df[col].iloc[val_idx].map(smooth).fillna(tr_mean).values
            df[col] = encoded
            stats_full = y.groupby(df[col]).agg(["sum", "count"])
            self.cat_means_[col] = (
                (stats_full["sum"] + self.smoothing * global_mean) /
                (stats_full["count"] + self.smoothing)
            ).to_dict()
        return df

    def transform_test(self, df, cat_cols):
        df = df.copy()
        for col in cat_cols:
            if col not in df.columns or col not in self.cat_means_:
                continue
            df[col] = df[col].map(self.cat_means_[col]).fillna(self.global_mean_[col])
        return df


# ════════════════════════════════════════════════════════════
# 데이터 로드 + 기본 전처리
# ════════════════════════════════════════════════════════════

DATA_DIR = "/kaggle/input/datasets/yjsheila/infertility/"

train  = pd.read_csv(f"{DATA_DIR}/train.csv")
test   = pd.read_csv(f"{DATA_DIR}/test.csv")

train_base, _,        cat_cols = preprocess_base(train, is_train=True)
test_base,  test_ids, _        = preprocess_base(test,  is_train=False)

y_all = train_base[TARGET]
skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
neg_pos_ratio = (y_all == 0).sum() / (y_all == 1).sum()


# ════════════════════════════════════════════════════════════
# ① LGB 피처셋 — LabelEncoding + importance 상위 LGB_TOP_N개
# ════════════════════════════════════════════════════════════

train_le = apply_label_encoding(train_base.drop(columns=[TARGET]), is_train=True)
test_le  = apply_label_encoding(test_base, is_train=False)

# LGB 전용 drop 컬럼 제거
lgb_drop    = [c for c in LGB_DROP_COLS if c in train_le.columns]
train_le_lgb = train_le.drop(columns=lgb_drop)
test_le_lgb  = test_le.drop(columns=lgb_drop)

# 빠른 LGB로 importance 계산
print(f"[LGB 피처 선택] importance 계산 중...")
_quick_lgb = lgb.LGBMClassifier(
    n_estimators=200, learning_rate=0.1, num_leaves=63,
    random_state=SEED, n_jobs=-1, verbose=-1,
    scale_pos_weight=neg_pos_ratio
)
_quick_lgb.fit(train_le_lgb, y_all)
_imp = pd.Series(_quick_lgb.feature_importances_, index=train_le_lgb.columns)
top_lgb_feats = _imp.nlargest(LGB_TOP_N).index.tolist()

X_lgb_train = train_le_lgb[top_lgb_feats]
X_lgb_test  = test_le_lgb.reindex(columns=top_lgb_feats, fill_value=0)
print(f"  LGB 피처 수: {len(top_lgb_feats)}개 (importance 상위 {LGB_TOP_N}개)")


# ════════════════════════════════════════════════════════════
# ② CAT 피처셋 — OOF Target Encoding
# ════════════════════════════════════════════════════════════

print(f"\n[CAT 피처 준비] OOF Target Encoding 적용 중...")
te_cat = TargetEncoder(smoothing=10)
train_cat_te = te_cat.fit_transform_train(
    train_base.drop(columns=[TARGET]), y_all, cat_cols, skf
)
test_cat_te  = te_cat.transform_test(test_base, cat_cols)

# 남은 object 컬럼 LabelEncoding (TE 안 된 컬럼 혹시 있으면)
for col in train_cat_te.select_dtypes(include="object").columns:
    le = LabelEncoder()
    train_cat_te[col] = le.fit_transform(train_cat_te[col].astype(str))
    test_cat_te[col]  = test_cat_te[col].map(
        dict(zip(le.classes_, le.transform(le.classes_)))
    ).fillna(0)

X_cat_train = train_cat_te
X_cat_test  = test_cat_te.reindex(columns=X_cat_train.columns, fill_value=0)
print(f"  CAT 피처 수: {X_cat_train.shape[1]}개 (TE 적용)")


# ════════════════════════════════════════════════════════════
# ③ XGB 피처셋 — OOF Target Encoding + 상호작용 TE 피처
# ════════════════════════════════════════════════════════════

print(f"\n[XGB 피처 준비] TE + 상호작용 피처 생성 중...")
te_xgb = TargetEncoder(smoothing=10)
train_xgb_te = te_xgb.fit_transform_train(
    train_base.drop(columns=[TARGET]), y_all, cat_cols, skf
)
test_xgb_te  = te_xgb.transform_test(test_base, cat_cols)

# 남은 object 컬럼 LabelEncoding
for col in train_xgb_te.select_dtypes(include="object").columns:
    le = LabelEncoder()
    train_xgb_te[col] = le.fit_transform(train_xgb_te[col].astype(str))
    test_xgb_te[col]  = test_xgb_te[col].map(
        dict(zip(le.classes_, le.transform(le.classes_)))
    ).fillna(0)

# 상호작용 TE 피처 추가 (TE된 컬럼끼리 곱셈)
for col_a, col_b in TE_INTERACTION_PAIRS:
    if col_a in train_xgb_te.columns and col_b in train_xgb_te.columns:
        feat_name = f"TE_interact_{col_a}_{col_b}"
        train_xgb_te[feat_name] = train_xgb_te[col_a] * train_xgb_te[col_b]
        test_xgb_te[feat_name]  = test_xgb_te[col_a]  * test_xgb_te[col_b]
        print(f"  상호작용 피처 추가: {feat_name}")

X_xgb_train = train_xgb_te
X_xgb_test  = test_xgb_te.reindex(columns=X_xgb_train.columns, fill_value=0)
print(f"  XGB 피처 수: {X_xgb_train.shape[1]}개 (TE + 상호작용)")

print(f"\n피처셋 구성 완료")
print(f"  LGB: {X_lgb_train.shape[1]}개 | CAT: {X_cat_train.shape[1]}개 | XGB: {X_xgb_train.shape[1]}개")


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
# OOF 학습 + test 예측
# ════════════════════════════════════════════════════════════

oof_lgb  = np.zeros(len(y_all))
oof_cat  = np.zeros(len(y_all))
oof_xgb  = np.zeros(len(y_all))
test_lgb = np.zeros(len(X_lgb_test))
test_cat = np.zeros(len(X_cat_test))
test_xgb = np.zeros(len(X_xgb_test))

print(f"\n{'='*60}")
print(f"exp031 OOF 학습 (Mixed FE, 5-Fold)")
print(f"{'='*60}")

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_lgb_train, y_all), 1):
    y_tr, y_val = y_all.iloc[tr_idx], y_all.iloc[val_idx]

    # LGB
    lgb_model = lgb.LGBMClassifier(**BEST_LGB_PARAMS)
    lgb_model.fit(X_lgb_train.iloc[tr_idx], y_tr)
    oof_lgb[val_idx]  = lgb_model.predict_proba(X_lgb_train.iloc[val_idx])[:, 1]
    test_lgb         += lgb_model.predict_proba(X_lgb_test)[:, 1] / N_FOLDS

    # CAT
    cat_model = CatBoostClassifier(**BEST_CAT_PARAMS)
    cat_model.fit(X_cat_train.iloc[tr_idx], y_tr)
    oof_cat[val_idx]  = cat_model.predict_proba(X_cat_train.iloc[val_idx])[:, 1]
    test_cat         += cat_model.predict_proba(X_cat_test)[:, 1] / N_FOLDS

    # XGB
    xgb_model = XGBClassifier(**BEST_XGB_PARAMS)
    xgb_model.fit(X_xgb_train.iloc[tr_idx], y_tr)
    oof_xgb[val_idx]  = xgb_model.predict_proba(X_xgb_train.iloc[val_idx])[:, 1]
    test_xgb         += xgb_model.predict_proba(X_xgb_test)[:, 1] / N_FOLDS

    print(f"  Fold {fold}  LGB={roc_auc_score(y_val, oof_lgb[val_idx]):.4f}"
          f"  CAT={roc_auc_score(y_val, oof_cat[val_idx]):.4f}"
          f"  XGB={roc_auc_score(y_val, oof_xgb[val_idx]):.4f}")

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

def rank_avg(arr):
    return np.stack([rankdata(arr[:, i]) for i in range(arr.shape[1])], axis=1).mean(axis=1)

def rank_normalize(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)

results = {
    "Simple Average": (
        roc_auc_score(y_all, oofs.mean(axis=1)),
        oofs.mean(axis=1),
        tests.mean(axis=1),
    ),
    "AUC-weighted": (
        roc_auc_score(y_all, (oofs * (aucs / aucs.sum())).sum(axis=1)),
        (oofs * (aucs / aucs.sum())).sum(axis=1),
        (tests * (aucs / aucs.sum())).sum(axis=1),
    ),
    "Rank Average": (
        roc_auc_score(y_all, rank_avg(oofs)),
        rank_avg(oofs),
        rank_normalize(rank_avg(tests)),
    ),
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
print(f"모델명        : LGB + CAT + XGB (Mixed FE) 앙상블")
print(f"앙상블 방법   : {best_method}")
print(f"LGB 피처 수  : {X_lgb_train.shape[1]}개 (importance 상위 {LGB_TOP_N}개)")
print(f"CAT 피처 수  : {X_cat_train.shape[1]}개 (OOF Target Encoding)")
print(f"XGB 피처 수  : {X_xgb_train.shape[1]}개 (OOF TE + 상호작용 피처)")
print(f"LGB OOF AUC : {auc_lgb:.5f}")
print(f"CAT OOF AUC : {auc_cat:.5f}")
print(f"XGB OOF AUC : {auc_xgb:.5f}")
print(f"앙상블 AUC  : {best_auc:.5f}  ({best_auc - BASELINE:+.5f} vs exp024)")
print(f"F1 Score    : {f1_score(y_all, oof_binary, average='macro'):.4f}")
print(f"Recall      : {recall_score(y_all, oof_binary, average='macro'):.4f}")
print(f"Precision   : {precision_score(y_all, oof_binary, average='macro'):.4f}")
print(f"Accuracy    : {accuracy_score(y_all, oof_binary):.4f}")
print(f"검증 방법    : Stratified {N_FOLDS}-Fold OOF")
print(f"클래스 불균형: scale_pos_weight={neg_pos_ratio:.4f}")
print(f"파일명       : {out_fname}")
print(f"특이사항     :")
print(f"  - LGB: LabelEncoding + importance 상위 {LGB_TOP_N}개 자동 선택")
print(f"  - CAT: OOF Target Encoding (smoothing=10)")
print(f"  - XGB: OOF TE + 상호작용 피처 {len(TE_INTERACTION_PAIRS)}개")
print(f"  - 모델마다 완전히 다른 피처셋 → 앙상블 다양성 극대화")
print(f"인사이트     : Mixed FE가 exp024 대비 다양성 확보에 기여하는지 확인")
print("="*55)
