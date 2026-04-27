# ════════════════════════════════════════════════════════════
# exp025 — 새 피처 엔지니어링 추가 베이스라인
#          exp024 전처리 기반 + 신규 파생 피처 5종 추가
#          - 나이 × 불임 원인 개수 교호작용
#          - 나이 × 이전 임신 여부 교호작용
#          - 시술 횟수 구간화
#          - 해동 배아 비율
#          - 특정 시술 유형 이진 변수 분리
#          모델: LGB 베이스라인 (Optuna 없음)
# 기준선: exp024 (OOF 0.73997, 제출 0.7414540333)
# ════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score,
                             recall_score, precision_score, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 한글 폰트
os.system('apt-get install -y fonts-nanum -qq')
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")

SEED    = 42
N_FOLDS = 5
TARGET  = "임신 성공 여부"
EXP_NO  = 25
AUTHOR  = "SYJ"
BASELINE = 0.73997  # exp024 OOF AUC


# ════════════════════════════════════════════════════════════
# 설정값 (exp024와 동일)
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

# 특정 시술 유형 이진 변수로 분리할 키워드
PROCEDURE_TYPES = ["IVF", "ICSI", "IUI", "FER", "BLASTOCYST", "GIFT", "ICI"]

label_encoders = {}


# ════════════════════════════════════════════════════════════
# 전처리 함수
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

    # 경과일 결측 플래그
    date_cols = [c for c in ["난자 채취 경과일", "난자 혼합 경과일", "배아 이식 경과일"] if c in df.columns]
    for col in date_cols:
        df[col + "_결측여부"] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(df[col].median() if is_train else 0)

    # DI 구조적 결측 처리
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

    # ── exp024 기존 파생 피처 ──────────────────────────────
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

    # ── exp025 신규 파생 피처 5종 ─────────────────────────
    # 1. 나이 × 불임 원인 개수 교호작용
    #    나이가 많을수록 복합 원인이 더 치명적이라는 가설
    if "시술 당시 나이" in df.columns and "불임_원인_개수" in df.columns:
        df["나이_불임원인_상호작용"] = df["시술 당시 나이"] * df["불임_원인_개수"]

    # 2. 나이 × 이전 임신 여부 교호작용
    #    고령이어도 이전 임신 경험이 있으면 예후가 다름
    if "시술 당시 나이" in df.columns and "이전_임신_여부" in df.columns:
        df["나이_이전임신_상호작용"] = df["시술 당시 나이"] * df["이전_임신_여부"]

    # 3. 시술 횟수 구간화 (비선형 패턴 포착)
    #    1회(첫 시술) / 2~3회(경험 있음) / 4회 이상(반복 실패)
    if "총 시술 횟수" in df.columns:
        df["시술횟수_구간"] = pd.cut(
            df["총 시술 횟수"],
            bins=[-1, 1, 3, 999],
            labels=[0, 1, 2]
        ).astype(int)

    # 4. 해동 배아 비율 = 해동된 배아 수 / 총 생성 배아 수
    #    동결-해동 시술 비중 반영
    if "해동된 배아 수" in df.columns and "총 생성 배아 수" in df.columns:
        df["해동_배아_비율"] = df["해동된 배아 수"] / (df["총 생성 배아 수"] + eps)

    # 5. 특정 시술 유형 이진 변수 분리
    #    LabelEncoding만 하면 순서 관계가 생기므로 주요 유형은 이진으로 분리
    if "특정 시술 유형" in df.columns:
        for proc in PROCEDURE_TYPES:
            df[f"시술유형_{proc}"] = df["특정 시술 유형"].astype(str).str.contains(proc).astype(int)

    # ── log 변환 및 인코딩 ────────────────────────────────
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

DATA_DIR = "/content/drive/MyDrive/헬스케어 3기/프로젝트/미니프로젝트3/open (1)/"

train  = pd.read_csv(f"/content/drive/MyDrive/헬스케어 3기/프로젝트/미니프로젝트3/open (1)/train.csv")
test   = pd.read_csv(f"{DATA_DIR}/test.csv")
sample = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

train_df, _       = preprocess(train, is_train=True)
test_df, test_ids = preprocess(test,  is_train=False)

X_all    = train_df.drop(TARGET, axis=1)
y_all    = train_df[TARGET]
X_submit = test_df.drop(columns=[TARGET], errors="ignore")

neg_pos_ratio = (y_all == 0).sum() / (y_all == 1).sum()

lgb_drop     = [c for c in LGB_DROP_COLS if c in X_all.columns]
X_all_lgb    = X_all.drop(columns=lgb_drop)
X_submit_lgb = X_submit.drop(columns=lgb_drop)

# 신규 피처 확인
new_features = [
    "나이_불임원인_상호작용", "나이_이전임신_상호작용",
    "시술횟수_구간", "해동_배아_비율",
] + [f"시술유형_{p}" for p in PROCEDURE_TYPES]

print(f"전체 피처 수  : {X_all.shape[1]}")
print(f"LGB 피처 수   : {X_all_lgb.shape[1]}")
print(f"scale_pos_weight: {neg_pos_ratio:.4f}")
print(f"\n신규 피처 ({len(new_features)}개):")
for f in new_features:
    exist = "✅" if f in X_all_lgb.columns else "❌"
    print(f"  {exist} {f}")


# ════════════════════════════════════════════════════════════
# LGB 베이스라인 (Optuna 없음)
# ════════════════════════════════════════════════════════════

lgb_params = {'n_estimators': 687, 
                   'learning_rate': 0.06728712035169694, 
                   'num_leaves': 272, 
                   'max_depth': 3, 
                   'min_child_samples': 62, 
                   'subsample': 0.7104765923920849, 
                   'colsample_bytree': 0.5598759562284701, 
                   'reg_alpha': 7.8646544609636635, 
                   'reg_lambda': 3.5299647925886912,
                   "scale_pos_weight": neg_pos_ratio,  
                   "random_state": SEED,
                   "n_jobs": -1,
                   "verbose": -1,
                   }

skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof     = np.zeros(len(X_all_lgb))
test_pred = np.zeros(len(X_submit_lgb))

print(f"\n{'='*55}")
print(f"LGB 베이스라인 학습 (5-Fold OOF)")
print(f"{'='*55}")

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all_lgb, y_all), 1):
    X_tr, X_val = X_all_lgb.iloc[tr_idx], X_all_lgb.iloc[val_idx]
    y_tr, y_val = y_all.iloc[tr_idx],      y_all.iloc[val_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_tr, y_tr)

    oof[val_idx]  = model.predict_proba(X_val)[:, 1]
    test_pred    += model.predict_proba(X_submit_lgb)[:, 1] / N_FOLDS

    fold_auc = roc_auc_score(y_val, oof[val_idx])
    print(f"  Fold {fold}  AUC={fold_auc:.5f}")

oof_auc    = roc_auc_score(y_all, oof)
oof_binary = (oof >= 0.5).astype(int)
diff       = oof_auc - BASELINE

print(f"\n{'='*55}")
print(f"OOF AUC : {oof_auc:.5f}  ({diff:+.5f} vs exp024)")
print(f"{'='*55}")
print(f"\n[Classification Report]")
print(classification_report(y_all, oof_binary))
print(f"[Confusion Matrix]")
print(confusion_matrix(y_all, oof_binary))


# ════════════════════════════════════════════════════════════
# Feature Importance
# ════════════════════════════════════════════════════════════

feat_imp = pd.Series(model.feature_importances_, index=X_all_lgb.columns).nlargest(20)

plt.figure(figsize=(10, 7))
feat_imp.sort_values().plot(kind="barh", color="steelblue")
plt.title(f"exp{EXP_NO:03d} Feature Importance Top 20")
plt.tight_layout()
plt.savefig(f"feature_importance_exp{EXP_NO:03d}.png", dpi=150)
plt.show()

print(f"\n[Feature Importance Top 20]")
for i, (feat, val) in enumerate(feat_imp.items(), 1):
    new_mark = " ★신규" if feat in new_features else ""
    print(f"  {i:>2}. {feat:<35} {val:.0f}{new_mark}")

    # ════════════════════════════════════════════════════════════
# 제출 파일 저장
# ════════════════════════════════════════════════════════════

out_fname  = f"submission_exp{EXP_NO:03d}_{AUTHOR}.csv"
submission = pd.DataFrame({"ID": test_ids, "probability": test_pred})
submission.to_csv(out_fname, index=False)
print(f"\n제출 파일 저장 완료: {out_fname}")
print(submission.head())

# ════════════════════════════════════════════════════════════
# 실험 기록장용 요약 출력
# ════════════════════════════════════════════════════════════

print("\n" + "="*55)
print("📋 실험 기록장 정보")
print("="*55)
print(f"실험 번호     : exp{EXP_NO:03d}")
print(f"모델명        : LightGBM 베이스라인")
print(f"전체 피처 수  : {X_all.shape[1]}  /  LGB 피처 수: {X_all_lgb.shape[1]}")
print(f"신규 피처     : 나이×불임원인, 나이×이전임신, 시술횟수구간, 해동배아비율, 시술유형이진({len(PROCEDURE_TYPES)}개)")
print(f"OOF AUC      : {oof_auc:.5f}  ({diff:+.5f} vs exp024)")
print(f"F1 Score     : {f1_score(y_all, oof_binary, average='macro'):.4f}")
print(f"Recall       : {recall_score(y_all, oof_binary, average='macro'):.4f}")
print(f"Precision    : {precision_score(y_all, oof_binary, average='macro'):.4f}")
print(f"Accuracy     : {accuracy_score(y_all, oof_binary):.4f}")
print(f"검증 방법     : Stratified {N_FOLDS}-Fold OOF")
print(f"클래스 불균형 : scale_pos_weight={neg_pos_ratio:.4f}")
print(f"파일명        : {out_fname}")
print("="*55)