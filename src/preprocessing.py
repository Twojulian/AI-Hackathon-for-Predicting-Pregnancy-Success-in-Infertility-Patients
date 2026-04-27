"""
공통 전처리 모듈

노트북에서 사용법:
    import sys; sys.path.insert(0, '..')
    from src.preprocessing import preprocess, CONST_COLS, RECODE_ZERO_COLS

처리 내용 (EDA 인사이트 기반):
    1. 상수 컬럼 제거 (전체 행이 동일한 값)
    2. 결측 → 0 재코딩 (비결측=전부 1인 컬럼 → 결측=미시행 의미)
    3. 순서형 정수 매핑
    4. 배아 생성 주요 이유 멀티핫 인코딩 (복수 선택 분리)
    5. 명목형 LabelEncoding (train만 fit, test 미지 카테고리 → len(classes))

개인 피처 엔지니어링은 preprocess() 결과에 각자 추가하세요.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── 컬럼 상수 ─────────────────────────────────────────────────────────────────

TARGET = '임신 성공 여부'
ID_COL = 'ID'

# EDA 확인: 전체 행이 동일한 값 → 예측력 없음
CONST_COLS = [
    '불임 원인 - 여성 요인',  # 전체 행(256,351) 모두 0
    '난자 채취 경과일',        # 전체 행(256,351) 모두 0
]

# EDA 확인: 비결측값이 전부 1 → 결측=미시행(0)으로 재코딩
# 근거: 값이 0인 행이 단 한 건도 없음. "시행한 경우만 1로 기록, 미시행은 기록 안 함"으로 추론.
# (PGS/PGD는 선택적 고비용 검사라 대다수 환자가 받지 않는 것이 의학적으로 자연스러움)
# TODO: 데이터 제공자 확인 불가 시 가정이므로, 컬럼을 제거한 버전과 성능 비교 권장.
RECODE_ZERO_COLS = [
    '착상 전 유전 검사 사용 여부',  # 비결측 2,718행, 전부 1
    'PGD 시술 여부',                # 비결측 2,179행, 전부 1
    'PGS 시술 여부',                # 비결측 1,929행, 전부 1
]

# 순서형 매핑
COUNT_ORDER = ['0회', '1회', '2회', '3회', '4회', '5회', '6회 이상']
AGE_ORDER = [
    '만18-34세', '만35-37세', '만38-39세', '만40-42세',
    '만43-44세', '만45-50세', '알 수 없음',
]
DONOR_AGE_ORDER = [
    '만20세 이하', '만21-25세', '만26-30세', '만31-35세',
    '만36-40세', '만41-45세', '알 수 없음',
]

ORDINAL_COLS = {
    '시술 당시 나이'        : AGE_ORDER,
    '난자 기증자 나이'       : DONOR_AGE_ORDER,
    '정자 기증자 나이'       : DONOR_AGE_ORDER,
    '총 시술 횟수'          : COUNT_ORDER,
    '클리닉 내 총 시술 횟수' : COUNT_ORDER,
    'IVF 시술 횟수'         : COUNT_ORDER,
    'DI 시술 횟수'          : COUNT_ORDER,
    '총 임신 횟수'          : COUNT_ORDER,
    'IVF 임신 횟수'         : COUNT_ORDER,
    'DI 임신 횟수'          : COUNT_ORDER,
    '총 출산 횟수'          : COUNT_ORDER,
    'IVF 출산 횟수'         : COUNT_ORDER,
    'DI 출산 횟수'          : COUNT_ORDER,
}

# 명목형 컬럼
LABEL_COLS = [
    '시술 시기 코드', '시술 유형', '특정 시술 유형',
    '배란 유도 유형', '난자 출처', '정자 출처',
]

# 배아 생성 주요 이유 (복수 선택 → 멀티핫)
EMBRYO_COL = '배아 생성 주요 이유'
EMBRYO_REASONS = ['기증용', '난자 저장용', '배아 저장용', '연구용', '현재 시술용']


# ── 메인 함수 ─────────────────────────────────────────────────────────────────

def preprocess(train_df: pd.DataFrame,
               test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    공통 전처리 파이프라인. train/test를 함께 받아 처리합니다.

    Parameters
    ----------
    train_df : 원본 train DataFrame (ID, TARGET 포함)
    test_df  : 원본 test DataFrame  (ID 포함, TARGET 없음)

    Returns
    -------
    X_train, X_test : ID·TARGET 제거 후 수치형만 남긴 DataFrame
                      개인 피처 엔지니어링은 반환값에 추가로 적용하세요.
    """
    tr = train_df.drop(columns=[ID_COL, TARGET] + CONST_COLS, errors='ignore').copy()
    te = test_df.drop(columns=[ID_COL] + CONST_COLS, errors='ignore').copy()

    # 1. 결측 → 0 재코딩 (미시행 의미)
    for col in RECODE_ZERO_COLS:
        if col in tr.columns:
            tr[col] = tr[col].fillna(0).astype(int)
            te[col] = te[col].fillna(0).astype(int)

    # 2. 순서형 정수 매핑
    for col, order in ORDINAL_COLS.items():
        if col in tr.columns:
            mapping = {v: i for i, v in enumerate(order)}
            tr[col] = tr[col].map(mapping)
            te[col] = te[col].map(mapping)

    # 3. 배아 생성 주요 이유 멀티핫 (복수 선택 분리)
    if EMBRYO_COL in tr.columns:
        for reason in EMBRYO_REASONS:
            col_name = f'배아이유_{reason}'
            tr[col_name] = tr[EMBRYO_COL].fillna('').str.contains(reason).astype(int)
            te[col_name] = te[EMBRYO_COL].fillna('').str.contains(reason).astype(int)
        tr.drop(columns=[EMBRYO_COL], inplace=True)
        te.drop(columns=[EMBRYO_COL], inplace=True)

    # 4. 명목형 LabelEncoding (train만 fit — test 사용 시 data leakage)
    for col in LABEL_COLS:
        if col in tr.columns:
            le = LabelEncoder()
            le.fit(tr[col].astype(str))
            tr[col] = le.transform(tr[col].astype(str))
            # test의 미지 카테고리 → len(classes) 로 처리 (GBDT가 별도 분기로 처리)
            classes_map = {c: i for i, c in enumerate(le.classes_)}
            te[col] = te[col].astype(str).map(classes_map).fillna(len(le.classes_)).astype(int)

    return tr, te