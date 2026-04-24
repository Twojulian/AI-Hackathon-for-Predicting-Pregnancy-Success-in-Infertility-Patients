import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib
import optuna
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
)
optuna.logging.set_verbosity(optuna.logging.WARNING)  # 불필요한 로그 억제
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

label_encoders = {}

# ────────────────────────────────────────────
# 전처리 함수 (버전 A: 결측 컬럼 삭제)
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

def preprocess(df, is_train=True):
    df = df.copy()

    ids = df["ID"].copy() if "ID" in df.columns else None
    df = df.drop(columns=["ID"], errors="ignore")
    df = df.drop(columns=[c for c in HIGH_NULL_COLS if c in df.columns])

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
# 1. 데이터 불러오기 및 전처리
# ────────────────────────────────────────────
train_raw = pd.read_csv("/content/drive/MyDrive/헬스케어 3기/프로젝트/미니프로젝트3/open (1)/train.csv")
test_raw  = pd.read_csv("/content/drive/MyDrive/헬스케어 3기/프로젝트/미니프로젝트3/open (1)/test.csv")

train_df, _       = preprocess(train_raw, is_train=True)
test_df, test_ids = preprocess(test_raw,  is_train=False)

X = train_df.drop("임신 성공 여부", axis=1)
y = train_df["임신 성공 여부"]
X_submit = test_df.drop(columns=["임신 성공 여부"], errors="ignore")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Submit: {X_submit.shape}")

# ────────────────────────────────────────────
# 2. 베이스라인 성능 (튜닝 전)
# ────────────────────────────────────────────
baseline = RandomForestClassifier(
    n_estimators=100, max_depth=10,
    class_weight="balanced", random_state=42, n_jobs=-1
)
baseline.fit(X_train, y_train)
baseline_auc = roc_auc_score(y_val, baseline.predict_proba(X_val)[:, 1])
print(f"\n[베이스라인] AUC-ROC: {baseline_auc:.4f}")

# ────────────────────────────────────────────
# 3. Optuna Objective 함수 정의
# ────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    params = {
        # 트리 개수: 많을수록 안정적이지만 느림
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),  # 범위 축소 (800 -> 500)
        # 트리 깊이: 깊을수록 복잡한 패턴 학습, 과적합 위험
        "max_depth": trial.suggest_int("max_depth", 5, 20),                     # 범위 축소 (30 -> 20)
        # 분기 시 고려할 최소 샘플 수: 클수록 과적합 방지
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        # 리프 노드의 최소 샘플 수: 클수록 단순한 모델
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        # 각 트리에서 사용할 피처 비율
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7]),
        # 클래스 불균형 처리
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
        # 부트스트랩 샘플링 여부
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
    }

    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)

    # 3-Fold CV + pruning: 첫 fold가 낮으면 나머지 fold 건너뜀
    scores = []
    for step, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        score = roc_auc_score(y_train.iloc[val_idx],
                                model.predict_proba(X_train.iloc[val_idx])[:, 1])
        scores.append(score)

        trial.report(np.mean(scores), step)           # 중간 값 보고
        if trial.should_prune():                       # 가망 없으면 즉시 중단
            raise optuna.exceptions.TrialPruned()

    return np.mean(scores)

# ────────────────────────────────────────────
# 4. Optuna 최적화 실행
# ────────────────────────────────────────────
N_TRIALS = 30  # 시간이 오래 걸리면 줄여도 됨 (최소 30 권장)

print(f"\nOptuna 튜닝 시작 ({N_TRIALS} trials)... 잠시 기다려주세요 ☕")
study = optuna.create_study(direction="maximize", study_name="RFC_tuning")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\n✅ 최적화 완료!")
print(f"Best CV AUC : {study.best_value:.4f}")
print(f"Best Params : {study.best_params}")

# ────────────────────────────────────────────
# 5. 최적 파라미터로 최종 모델 학습
# ────────────────────────────────────────────
best_model = RandomForestClassifier(
    **study.best_params, random_state=42, n_jobs=-1
)
best_model.fit(X_train, y_train)

best_preds = best_model.predict(X_val)
best_proba = best_model.predict_proba(X_val)[:, 1]
best_auc   = roc_auc_score(y_val, best_proba)

print(f"\n{'='*50}")
print(f"📊 튜닝 전후 AUC-ROC 비교")
print(f"{'='*50}")
print(f"베이스라인:  {baseline_auc:.4f}")
print(f"Optuna 튜닝: {best_auc:.4f}")
print(f"향상폭:      {best_auc - baseline_auc:+.4f}")

print(f"\n--- 튜닝 후 검증 성능 ---")
print(classification_report(y_val, best_preds, target_names=["실패(0)", "성공(1)"]))

# ────────────────────────────────────────────
# 6. 시각화: 최적화 히스토리 + 파라미터 중요도
# ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# 최적화 과정 (trial마다 AUC 변화)
plot_optimization_history(study, ax=axes[0])
axes[0].set_title("Optuna 최적화 히스토리", fontsize=13)

# 어떤 파라미터가 AUC에 영향을 많이 줬는지
plot_param_importances(study, ax=axes[1])
axes[1].set_title("하이퍼파라미터 중요도", fontsize=13)

plt.suptitle(f"Optuna 튜닝 결과  |  베이스라인 {baseline_auc:.4f} → 튜닝 {best_auc:.4f}",
                fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("optuna_tuning_result.png", dpi=150)
plt.show()
print("\n결과 그래프 저장 완료: optuna_tuning_result.png")

# ────────────────────────────────────────────
# 7. 제출 파일 생성
# ────────────────────────────────────────────
submit_proba = best_model.predict_proba(X_submit)[:, 1]
submission = pd.DataFrame({
    "ID": test_ids,
    "probability": submit_proba
})
submission.to_csv("optuna_RFC_submission.csv", index=False)
print(f"\n✅ 제출 파일 생성 완료: optuna_RFC_submission.csv")
print(submission.head())
print(f"총 {len(submission)}개 예측 완료")