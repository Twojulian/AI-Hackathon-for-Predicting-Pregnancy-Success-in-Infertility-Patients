# ════════════════════════════════════════════════════════════
# exp028 — 완전 새로운 전처리 방향
#   ① Target Encoding  (범주형 → 타겟 평균, OOF 방식으로 리크 방지)
#   ② RobustScaler     (수치형 피처 스케일링)
#   ③ TOP_N_FEATURES   (피처 수 줄이기 실험용 파라미터 노출)
#   ④ exp024 Optuna 최적 파라미터 유지 (모델은 그대로)
#
# 기준선: exp024 앙상블 OOF AUC 0.74025
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
from sklearn.preprocessing import RobustScaler
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
EXP_NO   = 28
AUTHOR   = "SYJ"
BASELINE = 0.74025   # exp024 앙상블 OOF AUC

# ════════════════════════════════════════════════════════════
# ★ 피처 수 줄이기 실험할 때 여기만 바꾸면 됨
#   None → 전체 피처 사용
#   예) TOP_N_FEATURES = 80 → importance 상위 80개만 사용
# ════════════════════════════════════════════════════════════
TOP_N_FEATURES = None


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
# 설정값 (exp026과 동일)
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


# ════════════════════════════════════════════════════════════
# 전처리 (범주형 컬럼 목록 분리 — Target Encoding에서 사용)
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


def preprocess_base(df, is_train=True, date_medians=None):
    """
    범주형 인코딩을 제외한 기본 전처리.
    범주형 컬럼은 문자열 그대로 반환 → 이후 Target Encoding 적용.
    """
    df = df.copy()
    ids = df["ID"].copy() if "ID" in df.columns else None
    df = df.drop(columns=["ID"], errors="ignore")
    df = df.drop(columns=[c for c in HIGH_NULL_COLS if c in df.columns])

    # 경과일 결측 플래그 + 대체
    date_cols = [c for c in ["난자 채취 경과일", "난자 혼합 경과일", "배아 이식 경과일"] if c in df.columns]
    _medians = {}
    for col in date_cols:
        df[col + "_결측여부"] = df[col].isnull().astype(int)
        if is_train:
            m = df[col].median()
            _medians[col] = m
        else:
            m = date_medians.get(col, 0) if date_medians else 0
        df[col] = df[col].fillna(m)

    # DI 마스킹
    if "시술 유형" in df.columns:
        di_mask = df["시술 유형"] == "DI"
        for col in DI_ZERO_COLS:
            if col in df.columns:
                df.loc[di_mask, col] = 0
                median_val = df.loc[~di_mask, col].median() if is_train else 0
                df.loc[~di_mask, col] = df.loc[~di_mask, col].fillna(median_val)
        df["is_DI"] = di_mask.astype(int)

    # 결측 처리
    num_cols = [c for c in df.select_dtypes(include="number").columns if c != TARGET]
    df[num_cols] = df[num_cols].fillna(0)
    df[df.select_dtypes(include="object").columns] = \
        df.select_dtypes(include="object").fillna("Unknown")

    # 횟수형 변환
    for col in COUNT_COLS:
        if col in df.columns:
            df[col] = df[col].apply(convert_count)

    # 나이 매핑
    if "시술 당시 나이" in df.columns:
        df["시술 당시 나이"] = df["시술 당시 나이"].map(AGE_MAP).fillna(36)
    for col in ["난자 기증자 나이", "정자 기증자 나이"]:
        if col in df.columns:
            df[col] = df[col].map(DONOR_AGE_MAP).fillna(0)

    eps = 1e-6

    # 파생 피처 (exp026과 동일)
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

    # log1p 변환
    for col in [c for c in LOG_COLS if c in df.columns]:
        df[col] = np.log1p(df[col])

    return df, ids, _medians


# ════════════════════════════════════════════════════════════
# ★ Target Encoding (OOF 방식 — train 리크 방지)
# ════════════════════════════════════════════════════════════

class TargetEncoder:
    """
    OOF 방식 Target Encoding.
    - train: fold 밖 데이터의 타겟 평균으로 인코딩 (리크 없음)
    - test : train 전체 타겟 평균으로 인코딩
    - smoothing: 카테고리 빈도가 낮을 때 전체 평균으로 당기는 정도
    """
    def __init__(self, smoothing=10):
        self.smoothing    = smoothing
        self.global_mean_ = {}
        self.cat_means_   = {}   # {col: {category: encoded_mean}}

    def fit_transform_train(self, df, y, cat_cols, skf):
        """OOF 방식으로 train 인코딩 + 전체 평균 저장."""
        df = df.copy()
        global_mean = y.mean()

        for col in cat_cols:
            if col not in df.columns:
                continue
            self.global_mean_[col] = global_mean
            encoded = np.zeros(len(df))

            for tr_idx, val_idx in skf.split(df, y):
                tr_target_mean = y.iloc[tr_idx].mean()
                stats = y.iloc[tr_idx].groupby(df[col].iloc[tr_idx]).agg(["sum", "count"])
                smooth = (stats["sum"] + self.smoothing * tr_target_mean) / \
                         (stats["count"] + self.smoothing)
                encoded[val_idx] = df[col].iloc[val_idx].map(smooth).fillna(tr_target_mean).values

            df[col] = encoded

            # test용 전체 평균 저장
            stats_full = y.groupby(df[col]).agg(["sum", "count"])
            self.cat_means_[col] = (
                (stats_full["sum"] + self.smoothing * global_mean) /
                (stats_full["count"] + self.smoothing)
            ).to_dict()

        return df

    def transform_test(self, df, cat_cols):
        """test 데이터는 train 전체 평균으로 인코딩."""
        df = df.copy()
        for col in cat_cols:
            if col not in df.columns or col not in self.cat_means_:
                continue
            df[col] = df[col].map(self.cat_means_[col]).fillna(self.global_mean_[col])
        return df


# ════════════════════════════════════════════════════════════
# 데이터 로드
# ════════════════════════════════════════════════════════════

DATA_DIR = "/kaggle/input/datasets/yjsheila/infertility/"

train  = pd.read_csv(f"{DATA_DIR}/train.csv")
test   = pd.read_csv(f"{DATA_DIR}/test.csv")
sample = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

# 기본 전처리 (범주형 인코딩 전 단계)
train_base, _,        date_medians = preprocess_base(train, is_train=True)
test_base,  test_ids, _            = preprocess_base(test,  is_train=False,
                                                     date_medians=date_medians)

# 범주형 컬럼 추출 (Target Encoding 대상)
cat_cols = train_base.select_dtypes(include="object").columns.tolist()
cat_cols = [c for c in cat_cols if c != TARGET]
print(f"Target Encoding 대상 범주형 컬럼 수: {len(cat_cols)}")
print(f"  {cat_cols}")

y_all = train_base[TARGET]
skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# ★ Target Encoding 적용
te = TargetEncoder(smoothing=10)
train_encoded = te.fit_transform_train(
    train_base.drop(columns=[TARGET]), y_all, cat_cols, skf
)
test_encoded  = te.transform_test(test_base, cat_cols)

# 수치형 스케일링 (RobustScaler — 이상치에 강함)
num_cols_to_scale = train_encoded.select_dtypes(include="number").columns.tolist()
scaler = RobustScaler()
train_encoded[num_cols_to_scale] = scaler.fit_transform(train_encoded[num_cols_to_scale])
test_encoded[num_cols_to_scale]  = scaler.transform(
    test_encoded[num_cols_to_scale].reindex(columns=num_cols_to_scale, fill_value=0)
)

X_all    = train_encoded.copy()
X_submit = test_encoded.reindex(columns=X_all.columns, fill_value=0)

neg_pos_ratio = (y_all == 0).sum() / (y_all == 1).sum()

# LGB 전용 피처 (exp026과 동일)
lgb_drop     = [c for c in LGB_DROP_COLS if c in X_all.columns]
X_all_lgb    = X_all.drop(columns=lgb_drop)
X_submit_lgb = X_submit.drop(columns=lgb_drop)

print(f"\n전체 피처 수  : {X_all.shape[1]}")
print(f"LGB 피처 수   : {X_all_lgb.shape[1]}")
print(f"scale_pos_weight: {neg_pos_ratio:.4f}")


# ════════════════════════════════════════════════════════════
# ★ TOP_N_FEATURES 피처 선택 (설정값이 None이면 전체 사용)
# ════════════════════════════════════════════════════════════

if TOP_N_FEATURES is not None:
    print(f"\n[피처 선택] importance 기반 상위 {TOP_N_FEATURES}개 사용")
    # 빠른 LGB로 importance 계산
    _quick_lgb = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.1,
        num_leaves=63, random_state=SEED, n_jobs=-1, verbose=-1
    )
    _quick_lgb.fit(X_all_lgb, y_all)
    _imp = pd.Series(_quick_lgb.feature_importances_, index=X_all_lgb.columns)
    top_feats_lgb = _imp.nlargest(min(TOP_N_FEATURES, len(_imp))).index.tolist()

    # CAT/XGB용도 비슷하게 적용
    top_feats_all = [f for f in X_all.columns if f in top_feats_lgb or f not in X_all_lgb.columns]
    top_feats_all = _imp.reindex([f for f in top_feats_all if f in _imp.index]).nlargest(
        min(TOP_N_FEATURES, len(_imp))).index.tolist()

    X_all_lgb    = X_all_lgb[top_feats_lgb]
    X_submit_lgb = X_submit_lgb[top_feats_lgb]
    X_all        = X_all[[f for f in top_feats_all if f in X_all.columns]]
    X_submit     = X_submit[[f for f in top_feats_all if f in X_submit.columns]]
    print(f"  → 선택 후 전체: {X_all.shape[1]}개 / LGB: {X_all_lgb.shape[1]}개")
else:
    print(f"\n[피처 선택] 전체 피처 사용 ({X_all.shape[1]}개)")


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

oof_lgb  = np.zeros(len(X_all))
oof_cat  = np.zeros(len(X_all))
oof_xgb  = np.zeros(len(X_all))
test_lgb = np.zeros(len(X_submit))
test_cat = np.zeros(len(X_submit))
test_xgb = np.zeros(len(X_submit))

print(f"\n{'='*60}")
print(f"exp028 OOF 학습 (LGB + CAT + XGB, 5-Fold)")
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
# 앙상블 비교 + 제출 파일 저장
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
        rank_normalize(rank_avg(tests)),   # ★ 정규화 적용
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
print(f"모델명        : LGB + CAT + XGB 앙상블 ({best_method})")
print(f"전체 피처 수  : {X_all.shape[1]}  /  LGB 피처 수: {X_all_lgb.shape[1]}")
print(f"TOP_N_FEATURES: {TOP_N_FEATURES if TOP_N_FEATURES else '전체'}")
print(f"LGB OOF AUC  : {auc_lgb:.5f}")
print(f"CAT OOF AUC  : {auc_cat:.5f}")
print(f"XGB OOF AUC  : {auc_xgb:.5f}")
print(f"앙상블 AUC   : {best_auc:.5f}  ({best_auc - BASELINE:+.5f} vs exp024)")
print(f"F1 Score     : {f1_score(y_all, oof_binary, average='macro'):.4f}")
print(f"Recall       : {recall_score(y_all, oof_binary, average='macro'):.4f}")
print(f"Precision    : {precision_score(y_all, oof_binary, average='macro'):.4f}")
print(f"Accuracy     : {accuracy_score(y_all, oof_binary):.4f}")
print(f"검증 방법     : Stratified {N_FOLDS}-Fold OOF")
print(f"클래스 불균형 : scale_pos_weight={neg_pos_ratio:.4f}")
print(f"파일명        : {out_fname}")
print(f"특이사항      :")
print(f"  - LabelEncoder → OOF Target Encoding (smoothing=10)")
print(f"  - RobustScaler 수치형 스케일링 추가")
print(f"  - TOP_N_FEATURES={TOP_N_FEATURES if TOP_N_FEATURES else '전체'}")
print(f"  - Rank Average test 정규화 수정 (0~1 보장)")
print(f"인사이트      : exp024 대비 Target Encoding 효과 확인 목적")
print("="*55)
