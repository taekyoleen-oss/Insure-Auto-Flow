"""
통계 분석 모듈화 툴 - 파이썬 분석 코드
이 파일은 insure-auto-flow 앱의 모든 데이터 분석 모듈을 파이썬으로 구현한 것입니다.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression, PoissonRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
import statsmodels.api as sm
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 데이터 로딩 모듈
# ============================================================================

def load_data(file_path: str):
    """
    CSV 파일에서 데이터를 로드합니다.
    
    Parameters:
    -----------
    file_path : str
        로드할 CSV 파일의 경로
    
    Returns:
    --------
    pd.DataFrame
        로드된 데이터프레임
    """
    print(f"데이터 로드 중: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"데이터 로드 완료. Shape: {df.shape}")
        print("\n데이터 미리보기:")
        print(df.head())
        print("\n데이터 정보:")
        print(df.info())
        return df
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다: {file_path}")
        return None
    except Exception as e:
        print(f"오류 발생: {e}")
        return None


def load_xol_data(file_path: str):
    """
    재보험 클레임 데이터를 로드합니다.
    
    Parameters:
    -----------
    file_path : str
        로드할 CSV 파일의 경로
    
    Returns:
    --------
    pd.DataFrame
        로드된 데이터프레임
    """
    print(f"XoL 클레임 데이터 로드 중: {file_path}")
    df = pd.read_csv(file_path)
    print("XoL 데이터 로드 완료.")
    return df


# ============================================================================
# 통계 분석 모듈
# ============================================================================

def analyze_statistics(df: pd.DataFrame):
    """
    데이터프레임에 대한 기술 통계량과 상관관계 분석을 수행합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        분석할 데이터프레임
    
    Returns:
    --------
    tuple
        (기술통계량, 상관관계 행렬)
    """
    print("=" * 60)
    print("기술 통계량 분석")
    print("=" * 60)
    
    # 기술 통계량
    desc_stats = df.describe()
    print(desc_stats)
    
    # 결측치 정보
    print("\n결측치 정보:")
    print(df.isnull().sum())
    
    # 상관관계 분석
    print("\n" + "=" * 60)
    print("상관관계 행렬")
    print("=" * 60)
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 0:
        corr_matrix = numeric_df.corr()
        print(corr_matrix)
        
        # 상관관계 히트맵 시각화
        if len(numeric_df.columns) > 1:
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", center=0)
            plt.title('상관관계 히트맵')
            plt.tight_layout()
            plt.show()
    else:
        print("수치형 컬럼이 없어 상관관계 분석을 수행할 수 없습니다.")
        corr_matrix = None
    
    return desc_stats, corr_matrix


# ============================================================================
# 데이터 선택 모듈
# ============================================================================

def select_data(df: pd.DataFrame, columns: list):
    """
    지정된 컬럼만 선택합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        원본 데이터프레임
    columns : list
        선택할 컬럼 이름 리스트
    
    Returns:
    --------
    pd.DataFrame
        선택된 컬럼만 포함하는 데이터프레임
    """
    print(f"컬럼 선택: {columns}")
    selected_df = df[columns].copy()
    print(f"선택 완료. Shape: {selected_df.shape}")
    return selected_df


# ============================================================================
# 결측치 처리 모듈
# ============================================================================

def handle_missing_values(df: pd.DataFrame, method: str = 'remove_row', 
                         strategy: str = 'mean', columns: list = None,
                         n_neighbors: int = 5):
    """
    결측치를 처리합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        처리할 데이터프레임
    method : str
        처리 방법: 'remove_row', 'impute', 'knn'
    strategy : str
        대체 전략: 'mean', 'median', 'mode' (method='impute'일 때 사용)
    columns : list
        처리할 컬럼 리스트 (None이면 모든 컬럼)
    n_neighbors : int
        KNN 방법 사용 시 이웃 수
    
    Returns:
    --------
    pd.DataFrame
        결측치가 처리된 데이터프레임
    """
    print(f"결측치 처리 방법: {method}")
    df_processed = df.copy()
    
    if method == 'remove_row':
        original_shape = df_processed.shape
        df_processed = df_processed.dropna()
        print(f"행 제거 완료. {original_shape[0]} -> {df_processed.shape[0]} 행")
    
    elif method == 'impute':
        cols_to_impute = columns if columns else df_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in cols_to_impute:
            if col not in df_processed.columns:
                continue
            if df_processed[col].isnull().any():
                if strategy == 'mean':
                    fill_value = df_processed[col].mean()
                elif strategy == 'median':
                    fill_value = df_processed[col].median()
                elif strategy == 'mode':
                    fill_value = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 0
                else:
                    fill_value = df_processed[col].mean()
                
                df_processed[col].fillna(fill_value, inplace=True)
                print(f"컬럼 '{col}' 결측치를 {strategy} 값({fill_value:.2f})으로 대체")
    
    elif method == 'knn':
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
            print(f"KNN 방법으로 결측치 처리 완료 (n_neighbors={n_neighbors})")
        else:
            print("경고: 수치형 컬럼이 없어 KNN 방법을 사용할 수 없습니다.")
    
    return df_processed


# ============================================================================
# 범주형 인코딩 모듈
# ============================================================================

def encode_categorical(df: pd.DataFrame, method: str = 'label', 
                      columns: list = None, drop: str = 'first',
                      handle_unknown: str = 'ignore', ordinal_mapping: dict = None):
    """
    범주형 변수를 인코딩합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        처리할 데이터프레임
    method : str
        인코딩 방법: 'label', 'one_hot', 'ordinal'
    columns : list
        인코딩할 컬럼 리스트 (None이면 모든 범주형 컬럼)
    drop : str
        One-hot 인코딩 시 제거할 더미 변수: 'first', 'if_binary', None
    handle_unknown : str
        알 수 없는 값 처리: 'error', 'ignore'
    ordinal_mapping : dict
        Ordinal 인코딩 시 사용할 매핑 딕셔너리
    
    Returns:
    --------
    pd.DataFrame
        인코딩된 데이터프레임
    """
    print(f"범주형 인코딩 방법: {method}")
    df_encoded = df.copy()
    
    if columns is None:
        columns = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if method == 'label':
        for col in columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                print(f"컬럼 '{col}'에 Label Encoding 적용")
    
    elif method == 'one_hot':
        for col in columns:
            if col in df_encoded.columns:
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=(drop == 'first'))
                df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
                print(f"컬럼 '{col}'에 One-Hot Encoding 적용")
    
    elif method == 'ordinal':
        for col in columns:
            if col in df_encoded.columns:
                if ordinal_mapping and col in ordinal_mapping:
                    mapping = {val: idx for idx, val in enumerate(ordinal_mapping[col])}
                    df_encoded[col] = df_encoded[col].map(mapping)
                    if handle_unknown == 'ignore':
                        df_encoded[col].fillna(-1, inplace=True)
                else:
                    # 알파벳 순서로 매핑
                    unique_vals = sorted(df_encoded[col].unique())
                    mapping = {val: idx for idx, val in enumerate(unique_vals)}
                    df_encoded[col] = df_encoded[col].map(mapping)
                print(f"컬럼 '{col}'에 Ordinal Encoding 적용")
    
    return df_encoded


# ============================================================================
# 데이터 정규화 모듈
# ============================================================================

def normalize_data(df: pd.DataFrame, method: str = 'MinMax', columns: list = None):
    """
    데이터를 정규화합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        정규화할 데이터프레임
    method : str
        정규화 방법: 'MinMax', 'StandardScaler', 'RobustScaler'
    columns : list
        정규화할 컬럼 리스트 (None이면 모든 수치형 컬럼)
    
    Returns:
    --------
    pd.DataFrame
        정규화된 데이터프레임
    """
    print(f"데이터 정규화 방법: {method}")
    df_normalized = df.copy()
    
    if columns is None:
        columns = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
    
    if method == 'MinMax':
        scaler = MinMaxScaler()
    elif method == 'StandardScaler':
        scaler = StandardScaler()
    elif method == 'RobustScaler':
        scaler = RobustScaler()
    else:
        print(f"알 수 없는 정규화 방법: {method}. MinMax를 사용합니다.")
        scaler = MinMaxScaler()
    
    df_normalized[columns] = scaler.fit_transform(df_normalized[columns])
    print(f"정규화 완료. 컬럼: {columns}")
    
    return df_normalized


# ============================================================================
# 데이터 변환 모듈
# ============================================================================

def transform_data(df: pd.DataFrame, transformations: dict):
    """
    수치형 컬럼에 수학적 변환을 적용합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        변환할 데이터프레임
    transformations : dict
        {컬럼명: 변환방법} 형태의 딕셔너리
        변환방법: 'Log', 'Square Root', 'Min-Log', 'Min-Square Root'
    
    Returns:
    --------
    pd.DataFrame
        변환된 데이터프레임
    """
    print("데이터 변환 적용 중...")
    df_transformed = df.copy()
    
    for col, method in transformations.items():
        if method == 'None' or col not in df_transformed.columns:
            continue
        
        if not pd.api.types.is_numeric_dtype(df_transformed[col]):
            print(f"경고: 컬럼 '{col}'은 수치형이 아니므로 변환할 수 없습니다.")
            continue
        
        new_col_name = f"{col}_{method.lower().replace(' ', '_').replace('-', '_')}"
        print(f"  - 컬럼 '{col}'에 '{method}' 변환 적용 -> '{new_col_name}'")
        
        if method == 'Log':
            df_transformed[new_col_name] = np.log(df_transformed[col].apply(lambda x: x if x > 0 else np.nan))
            df_transformed[new_col_name].fillna(0, inplace=True)
        elif method == 'Square Root':
            df_transformed[new_col_name] = np.sqrt(df_transformed[col].apply(lambda x: x if x >= 0 else np.nan))
            df_transformed[new_col_name].fillna(0, inplace=True)
        elif method == 'Min-Log':
            min_val = df_transformed[col].min()
            df_transformed[new_col_name] = np.log((df_transformed[col] - min_val) + 1)
        elif method == 'Min-Square Root':
            min_val = df_transformed[col].min()
            df_transformed[new_col_name] = np.sqrt((df_transformed[col] - min_val) + 1)
    
    print("데이터 변환 완료.")
    return df_transformed


# ============================================================================
# 데이터 리샘플링 모듈
# ============================================================================

def resample_data(df: pd.DataFrame, method: str = 'SMOTE', target_column: str = None):
    """
    클래스 불균형을 처리하기 위해 데이터를 리샘플링합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        리샘플링할 데이터프레임
    method : str
        리샘플링 방법: 'SMOTE', 'NearMiss'
    target_column : str
        타겟 컬럼 이름
    
    Returns:
    --------
    pd.DataFrame
        리샘플링된 데이터프레임
    """
    if target_column is None:
        print("경고: 타겟 컬럼이 지정되지 않았습니다.")
        return df
    
    print(f"리샘플링 방법: {method}, 타겟 컬럼: {target_column}")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    if method == 'SMOTE':
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print(f"SMOTE 적용 완료. {len(X)} -> {len(X_resampled)} 샘플")
    elif method == 'NearMiss':
        near_miss = NearMiss(version=1)
        X_resampled, y_resampled = near_miss.fit_resample(X, y)
        print(f"NearMiss 적용 완료. {len(X)} -> {len(X_resampled)} 샘플")
    else:
        print(f"알 수 없는 리샘플링 방법: {method}")
        return df
    
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[target_column] = y_resampled
    
    return df_resampled


# ============================================================================
# 데이터 분할 모듈
# ============================================================================

def split_data(df: pd.DataFrame, train_size: float = 0.7, random_state: int = 42,
               shuffle: bool = True, stratify: bool = False, stratify_column: str = None):
    """
    데이터를 훈련 세트와 테스트 세트로 분할합니다.
    sklearn의 train_test_split 함수를 사용합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        분할할 데이터프레임
    train_size : float
        훈련 세트 비율 (0.0 ~ 1.0)
    random_state : int
        랜덤 시드
    shuffle : bool
        셔플 여부
    stratify : bool
        계층화 여부
    stratify_column : str
        계층화 기준 컬럼
    
    Returns:
    --------
    tuple
        (훈련 데이터프레임, 테스트 데이터프레임)
    """
    print("데이터 분할 중...")
    print(f"  훈련 세트 비율: {train_size}")
    print(f"  랜덤 시드: {random_state}")
    print(f"  셔플: {shuffle}")
    print(f"  계층화: {stratify}")
    
    # stratify 배열 준비
    stratify_array = None
    if stratify and stratify_column and stratify_column != 'None':
        if stratify_column in df.columns:
            stratify_array = df[stratify_column]
            print(f"  계층화 기준 컬럼: {stratify_column}")
        else:
            print(f"  경고: 계층화 컬럼 '{stratify_column}'을 찾을 수 없습니다. 계층화 없이 진행합니다.")
    elif stratify:
        print("  경고: 계층화가 True이지만 계층화 컬럼이 지정되지 않았습니다. 계층화 없이 진행합니다.")
    
    # sklearn의 train_test_split 사용
    train_df, test_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify_array
    )
    
    print(f"분할 완료. 훈련 세트: {len(train_df)} 행, 테스트 세트: {len(test_df)} 행")
    
    return train_df, test_df


# ============================================================================
# 머신러닝 모델 생성 모듈
# ============================================================================

def create_linear_model(model_type: str = 'LinearRegression', fit_intercept: bool = True,
                        alpha: float = 1.0, l1_ratio: float = 0.5):
    """
    선형 회귀 모델을 생성합니다.
    
    Parameters:
    -----------
    model_type : str
        모델 타입: 'LinearRegression', 'Lasso', 'Ridge', 'ElasticNet'
    fit_intercept : bool
        절편 포함 여부
    alpha : float
        정규화 강도 (Lasso, Ridge, ElasticNet)
    l1_ratio : float
        L1 정규화 비율 (ElasticNet)
    
    Returns:
    --------
    모델 객체
    """
    print(f"{model_type} 모델 생성 중...")
    
    if model_type == 'LinearRegression':
        model = LinearRegression(fit_intercept=fit_intercept)
    elif model_type == 'Lasso':
        model = Lasso(alpha=alpha, fit_intercept=fit_intercept, random_state=42)
    elif model_type == 'Ridge':
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept, random_state=42)
    elif model_type == 'ElasticNet':
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, random_state=42)
    else:
        print(f"알 수 없는 모델 타입: {model_type}. LinearRegression을 사용합니다.")
        model = LinearRegression(fit_intercept=fit_intercept)
    
    print("모델 생성 완료.")
    return model


def create_logistic_regression(penalty: str = 'l2', C: float = 1.0, 
                               solver: str = 'lbfgs', max_iter: int = 100):
    """
    로지스틱 회귀 모델을 생성합니다.
    
    Parameters:
    -----------
    penalty : str
        정규화 방법: 'l1', 'l2', 'elasticnet', None
    C : float
        정규화 강도의 역수
    solver : str
        최적화 알고리즘
    max_iter : int
        최대 반복 횟수
    
    Returns:
    --------
    LogisticRegression 모델 객체
    """
    print("로지스틱 회귀 모델 생성 중...")
    model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=max_iter, random_state=42)
    print("모델 생성 완료.")
    return model


def create_poisson_regression(alpha: float = 1.0, max_iter: int = 100):
    """
    포아송 회귀 모델을 생성합니다 (sklearn 버전 - 레거시).
    
    Parameters:
    -----------
    alpha : float
        정규화 강도
    max_iter : int
        최대 반복 횟수
    
    Returns:
    --------
    PoissonRegressor 모델 객체
    """
    print("포아송 회귀 모델 생성 중...")
    model = PoissonRegressor(alpha=alpha, max_iter=max_iter)
    print("모델 생성 완료.")
    return model


def fit_count_regression_statsmodels(df: pd.DataFrame, distribution_type: str, feature_columns: list, label_column: str, 
                                     max_iter: int = 100, disp: float = 1.0):
    """
    statsmodels를 사용하여 포아송, 음이항, Quasi-Poisson 회귀 모델을 피팅합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터
    distribution_type : str
        분포 타입: 'Poisson', 'NegativeBinomial', 'QuasiPoisson'
    feature_columns : list
        특성 컬럼 리스트
    label_column : str
        레이블 컬럼 이름
    max_iter : int
        최대 반복 횟수
    disp : float
        음이항 회귀의 dispersion 파라미터 (distribution_type='NegativeBinomial'일 때 사용)
    
    Returns:
    --------
    dict
        모델 결과 딕셔너리 (results 객체, summary 텍스트, 통계량 포함)
    """
    print(f"{distribution_type} 회귀 모델 피팅 중...")
    
    X = df[feature_columns].copy()
    y = df[label_column].copy()
    
    # 결측치 제거
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    if len(X) == 0:
        raise ValueError("유효한 데이터가 없습니다. 결측치를 확인하세요.")
    
    X = sm.add_constant(X, prepend=True)
    
    try:
        if distribution_type == 'Poisson':
            model = sm.Poisson(y, X)
            results = model.fit(maxiter=max_iter)
        elif distribution_type == 'NegativeBinomial':
            model = sm.NegativeBinomial(y, X, loglike_method='nb2')
            results = model.fit(maxiter=max_iter, disp=disp)
        elif distribution_type == 'QuasiPoisson':
            # Quasi-Poisson은 GLM을 사용하여 구현
            model = sm.GLM(y, X, family=sm.families.Poisson())
            results = model.fit(maxiter=max_iter)
            # Quasi-Poisson은 분산을 과분산 파라미터로 조정
            # 잔차를 사용하여 과분산 추정
            mu = results.mu
            pearson_resid = (y - mu) / np.sqrt(mu)
            phi = np.sum(pearson_resid**2) / (len(y) - len(feature_columns) - 1)
            results.scale = phi  # 과분산 파라미터 설정
        else:
            raise ValueError(f"지원하지 않는 분포 타입: {distribution_type}")
        
        # 모델 요약 텍스트 생성
        summary_text = str(results.summary())
        print(f"\n--- {distribution_type} 회귀 모델 결과 ---")
        print(summary_text)
        
        # 통계량 추출
        metrics = {}
        metrics['Log Likelihood'] = results.llf if hasattr(results, 'llf') else None
        metrics['AIC'] = results.aic if hasattr(results, 'aic') else None
        metrics['BIC'] = results.bic if hasattr(results, 'bic') else None
        metrics['Deviance'] = results.deviance if hasattr(results, 'deviance') else None
        metrics['Pearson chi2'] = results.pearson_chi2 if hasattr(results, 'pearson_chi2') else None
        
        # 음이항 회귀의 경우 dispersion 파라미터 추가
        if distribution_type == 'NegativeBinomial':
            if hasattr(results, 'params'):
                # alpha 파라미터 찾기 (음이항 분포의 shape 파라미터)
                params_dict = results.params.to_dict() if hasattr(results.params, 'to_dict') else {}
                # statsmodels의 NegativeBinomial은 alpha를 별도로 저장하지 않을 수 있음
                # 대신 모델의 속성에서 확인
                if hasattr(model, 'alpha'):
                    metrics['Dispersion (alpha)'] = model.alpha
                elif hasattr(results, 'alpha'):
                    metrics['Dispersion (alpha)'] = results.alpha
        
        # Quasi-Poisson의 경우 과분산 파라미터 추가
        if distribution_type == 'QuasiPoisson':
            if hasattr(results, 'scale'):
                metrics['Dispersion (phi)'] = results.scale
        
        # 계수 정보 추출
        coefficients = {}
        if hasattr(results, 'params'):
            params = results.params
            if hasattr(params, 'to_dict'):
                params_dict = params.to_dict()
            else:
                params_dict = {name: params.iloc[i] if hasattr(params, 'iloc') else params[i] 
                               for i, name in enumerate(results.model.exog_names)}
            
            # 표준 오차, z/t 통계량, p-value, 신뢰구간 추출
            if hasattr(results, 'bse'):
                bse = results.bse
                if hasattr(bse, 'to_dict'):
                    bse_dict = bse.to_dict()
                else:
                    bse_dict = {name: bse.iloc[i] if hasattr(bse, 'iloc') else bse[i] 
                               for i, name in enumerate(results.model.exog_names)}
            else:
                bse_dict = {name: 0.0 for name in params_dict.keys()}
            
            if hasattr(results, 'tvalues'):
                tvalues = results.tvalues
            elif hasattr(results, 'zvalues'):
                tvalues = results.zvalues
            else:
                tvalues = None
            
            if hasattr(results, 'pvalues'):
                pvalues = results.pvalues
            else:
                pvalues = None
            
            # 신뢰구간 추출
            conf_int = None
            if hasattr(results, 'conf_int'):
                conf_int = results.conf_int()
            
            for param_name in params_dict.keys():
                coef_value = params_dict[param_name]
                std_err = bse_dict.get(param_name, 0.0)
                z_value = tvalues[param_name] if tvalues is not None and param_name in tvalues.index else 0.0
                p_value = pvalues[param_name] if pvalues is not None and param_name in pvalues.index else 1.0
                
                conf_lower = conf_int.loc[param_name, 0] if conf_int is not None and param_name in conf_int.index else 0.0
                conf_upper = conf_int.loc[param_name, 1] if conf_int is not None and param_name in conf_int.index else 0.0
                
                coefficients[param_name] = {
                    'coef': float(coef_value),
                    'std err': float(std_err),
                    'z': float(z_value),
                    'P>|z|': float(p_value),
                    '[0.025': float(conf_lower),
                    '0.975]': float(conf_upper)
                }
        
        return {
            'results': results,
            'summary_text': summary_text,
            'metrics': metrics,
            'coefficients': coefficients,
            'distribution_type': distribution_type
        }
        
    except Exception as e:
        print(f"모델 피팅 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise


def create_decision_tree(model_purpose: str = 'classification', criterion: str = 'gini',
                        max_depth: int = None, min_samples_split: int = 2, min_samples_leaf: int = 1,
                        class_weight: str = None):
    """
    의사결정나무 모델을 생성합니다.
    
    Parameters:
    -----------
    model_purpose : str
        모델 목적: 'classification', 'regression'
    criterion : str
        분할 기준: 'gini', 'entropy' (분류), 'mse', 'mae' (회귀)
    max_depth : int
        최대 깊이
    min_samples_split : int
        분할을 위한 최소 샘플 수
    min_samples_leaf : int
        리프 노드의 최소 샘플 수
    class_weight : str or None
        클래스 가중치: None, 'balanced'
    
    Returns:
    --------
    DecisionTree 모델 객체
    """
    print(f"의사결정나무 모델 생성 중 ({model_purpose})...")
    
    if model_purpose == 'classification':
        model = DecisionTreeClassifier(
            criterion=criterion.lower(),
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=42
        )
    else:
        criterion_reg = 'squared_error' if criterion == 'mse' else 'absolute_error'
        model = DecisionTreeRegressor(
            criterion=criterion_reg,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
    
    print("모델 생성 완료.")
    return model


def create_random_forest(model_purpose: str = 'classification', n_estimators: int = 100,
                        criterion: str = 'gini', max_depth: int = None):
    """
    랜덤 포레스트 모델을 생성합니다.
    
    Parameters:
    -----------
    model_purpose : str
        모델 목적: 'classification', 'regression'
    n_estimators : int
        트리 개수
    criterion : str
        분할 기준
    max_depth : int
        최대 깊이
    
    Returns:
    --------
    RandomForest 모델 객체
    """
    print(f"랜덤 포레스트 모델 생성 중 ({model_purpose})...")
    
    if model_purpose == 'classification':
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion.lower(),
            max_depth=max_depth,
            random_state=42
        )
    else:
        criterion_reg = 'squared_error' if criterion == 'mse' else 'absolute_error'
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion=criterion_reg,
            max_depth=max_depth,
            random_state=42
        )
    
    print("모델 생성 완료.")
    return model


def create_neural_network(model_purpose: str = 'classification', hidden_layer_sizes: str = '100',
                         activation: str = 'relu', max_iter: int = 200):
    """
    신경망 모델을 생성합니다.
    
    Parameters:
    -----------
    model_purpose : str
        모델 목적: 'classification', 'regression'
    hidden_layer_sizes : str
        은닉층 크기 (예: "100" 또는 "100,50")
    activation : str
        활성화 함수: 'relu', 'tanh', 'logistic'
    max_iter : int
        최대 반복 횟수
    
    Returns:
    --------
    Neural Network 모델 객체
    """
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    
    print(f"신경망 모델 생성 중 ({model_purpose})...")
    
    # Parse hidden_layer_sizes (e.g., "100" -> (100,), "100,50" -> (100, 50))
    if isinstance(hidden_layer_sizes, str):
        hidden_layers = tuple(int(x.strip()) for x in hidden_layer_sizes.split(','))
    else:
        hidden_layers = (100,) if hidden_layer_sizes is None else (hidden_layer_sizes,)
    
    if model_purpose == 'classification':
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            max_iter=max_iter,
            random_state=random_state
        )
    else:
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            max_iter=max_iter,
            random_state=random_state
        )
    
    print("모델 생성 완료.")
    return model


def create_svm(model_purpose: str = 'classification', C: float = 1.0,
              kernel: str = 'rbf', gamma: str = 'scale'):
    """
    서포트 벡터 머신 모델을 생성합니다.
    
    Parameters:
    -----------
    model_purpose : str
        모델 목적: 'classification', 'regression'
    C : float
        정규화 파라미터
    kernel : str
        커널 타입: 'linear', 'poly', 'rbf', 'sigmoid'
    gamma : str or float
        커널 계수
    
    Returns:
    --------
    SVM 모델 객체
    """
    print(f"SVM 모델 생성 중 ({model_purpose})...")
    
    if model_purpose == 'classification':
        model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)
    else:
        model = SVR(C=C, kernel=kernel, gamma=gamma)
    
    print("모델 생성 완료.")
    return model


def create_knn(model_purpose: str = 'classification', n_neighbors: int = 5,
               weights: str = 'uniform', algorithm: str = 'auto'):
    """
    K-최근접 이웃 모델을 생성합니다.
    
    Parameters:
    -----------
    model_purpose : str
        모델 목적: 'classification', 'regression'
    n_neighbors : int
        이웃 수
    weights : str
        가중치 방법: 'uniform', 'distance'
    algorithm : str
        알고리즘: 'auto', 'ball_tree', 'kd_tree', 'brute'
    
    Returns:
    --------
    KNN 모델 객체
    """
    print(f"KNN 모델 생성 중 ({model_purpose})...")
    
    if model_purpose == 'classification':
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
    else:
        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
    
    print("모델 생성 완료.")
    return model


def create_naive_bayes(var_smoothing: float = 1e-9):
    """
    나이브 베이즈 모델을 생성합니다.
    
    Parameters:
    -----------
    var_smoothing : float
        분산 스무딩 파라미터
    
    Returns:
    --------
    GaussianNB 모델 객체
    """
    print("나이브 베이즈 모델 생성 중...")
    model = GaussianNB(var_smoothing=var_smoothing)
    print("모델 생성 완료.")
    return model


def create_lda(solver: str = 'svd', shrinkage: float = None):
    """
    선형 판별 분석 모델을 생성합니다.
    
    Parameters:
    -----------
    solver : str
        솔버: 'svd', 'lsqr', 'eigen'
    shrinkage : float
        축소 파라미터 (None이면 자동)
    
    Returns:
    --------
    LinearDiscriminantAnalysis 모델 객체
    """
    print("LDA 모델 생성 중...")
    model = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
    print("모델 생성 완료.")
    return model


# ============================================================================
# 모델 훈련 및 평가 모듈
# ============================================================================

def train_model(model, df: pd.DataFrame, feature_columns: list, label_column: str):
    """
    모델을 훈련시킵니다.
    
    Parameters:
    -----------
    model : 모델 객체
        훈련할 모델
    df : pd.DataFrame
        훈련 데이터
    feature_columns : list
        특성 컬럼 리스트
    label_column : str
        레이블 컬럼 이름
    
    Returns:
    --------
    훈련된 모델 객체
    """
    print("모델 훈련 시작...")
    
    X = df[feature_columns]
    y = df[label_column]
    
    print(f"훈련 데이터: {X.shape[0]} 샘플, {X.shape[1]} 특성")
    print(f"특성: {feature_columns}")
    print(f"레이블: {label_column}")
    
    model.fit(X, y)
    
    print("모델 훈련 완료.")
    return model


def score_model(model, df: pd.DataFrame, feature_columns: list):
    """
    모델로 예측을 수행합니다.
    
    Parameters:
    -----------
    model : 모델 객체
        훈련된 모델
    df : pd.DataFrame
        예측할 데이터
    feature_columns : list
        특성 컬럼 리스트
    
    Returns:
    --------
    pd.DataFrame
        예측 결과가 추가된 데이터프레임
    """
    print("모델 예측 수행 중...")
    
    features_to_use = [col for col in feature_columns if col in df.columns]
    X_score = df[features_to_use]
    
    predictions = model.predict(X_score)
    
    scored_df = df.copy()
    scored_df['Predict'] = predictions
    
    print("예측 완료. 'Predict' 컬럼이 추가되었습니다.")
    print(scored_df.head())
    
    return scored_df


def evaluate_model(model, df: pd.DataFrame, label_column: str, prediction_column: str = 'Predict',
                   model_type: str = 'regression'):
    """
    모델의 성능을 평가합니다.
    
    Parameters:
    -----------
    model : 모델 객체
        평가할 모델
    df : pd.DataFrame
        평가 데이터
    label_column : str
        실제 레이블 컬럼
    prediction_column : str
        예측값 컬럼
    model_type : str
        모델 타입: 'classification', 'regression'
    
    Returns:
    --------
    dict
        평가 지표 딕셔너리
    """
    print("모델 평가 중...")
    
    y_true = df[label_column]
    y_pred = df[prediction_column]
    
    metrics = {}
    
    if model_type == 'classification':
        accuracy = accuracy_score(y_true, y_pred)
        metrics['accuracy'] = accuracy
        print(f"정확도: {accuracy:.4f}")
        
        print("\n분류 리포트:")
        print(classification_report(y_true, y_pred))
        
        print("\n혼동 행렬:")
        print(confusion_matrix(y_true, y_pred))
    
    else:  # regression
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        metrics['mse'] = mse
        metrics['rmse'] = rmse
        metrics['r2'] = r2
        
        print(f"평균 제곱 오차 (MSE): {mse:.4f}")
        print(f"평균 제곱근 오차 (RMSE): {rmse:.4f}")
        print(f"결정 계수 (R²): {r2:.4f}")
    
    return metrics


# ============================================================================
# 비지도 학습 모듈
# ============================================================================

def kmeans_clustering(df: pd.DataFrame, n_clusters: int = 3, feature_columns: list = None,
                     init: str = 'k-means++', n_init: int = 10, max_iter: int = 300):
    """
    K-Means 클러스터링을 수행합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        클러스터링할 데이터
    n_clusters : int
        클러스터 개수
    feature_columns : list
        사용할 특성 컬럼 리스트
    init : str
        초기화 방법
    n_init : int
        초기화 횟수
    max_iter : int
        최대 반복 횟수
    
    Returns:
    --------
    tuple
        (클러스터 할당이 추가된 데이터프레임, 모델 객체)
    """
    print(f"K-Means 클러스터링 수행 중 (클러스터 수: {n_clusters})...")
    
    if feature_columns is None:
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    X = df[feature_columns]
    
    kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    df_clustered = df.copy()
    df_clustered['cluster'] = clusters
    
    print(f"클러스터링 완료. 클러스터 중심:")
    print(kmeans.cluster_centers_)
    
    return df_clustered, kmeans


def hierarchical_clustering(df: pd.DataFrame, n_clusters: int = 3, feature_columns: list = None,
                           affinity: str = 'euclidean', linkage: str = 'ward'):
    """
    계층적 클러스터링을 수행합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        클러스터링할 데이터
    n_clusters : int
        클러스터 개수
    feature_columns : list
        사용할 특성 컬럼 리스트
    affinity : str
        거리 측정 방법
    linkage : str
        연결 방법
    
    Returns:
    --------
    pd.DataFrame
        클러스터 할당이 추가된 데이터프레임
    """
    print(f"계층적 클러스터링 수행 중 (클러스터 수: {n_clusters})...")
    
    if feature_columns is None:
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    X = df[feature_columns]
    
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        affinity=affinity,
        linkage=linkage
    )
    clusters = clustering.fit_predict(X)
    
    df_clustered = df.copy()
    df_clustered['cluster'] = clusters
    
    print("계층적 클러스터링 완료.")
    
    return df_clustered


def dbscan_clustering(df: pd.DataFrame, eps: float = 0.5, min_samples: int = 5,
                     feature_columns: list = None):
    """
    DBSCAN 클러스터링을 수행합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        클러스터링할 데이터
    eps : float
        이웃 반경
    min_samples : int
        최소 샘플 수
    feature_columns : list
        사용할 특성 컬럼 리스트
    
    Returns:
    --------
    tuple
        (클러스터 할당이 추가된 데이터프레임, 클러스터 개수, 노이즈 개수)
    """
    print(f"DBSCAN 클러스터링 수행 중 (eps={eps}, min_samples={min_samples})...")
    
    if feature_columns is None:
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    X = df[feature_columns]
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X)
    
    df_clustered = df.copy()
    df_clustered['cluster'] = clusters
    
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    
    print(f"DBSCAN 클러스터링 완료. 클러스터 수: {n_clusters}, 노이즈: {n_noise}")
    
    return df_clustered, n_clusters, n_noise


def pca_transform(df: pd.DataFrame, n_components: int = 2, feature_columns: list = None):
    """
    주성분 분석(PCA)을 수행합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        변환할 데이터
    n_components : int
        주성분 개수
    feature_columns : list
        사용할 특성 컬럼 리스트
    
    Returns:
    --------
    tuple
        (변환된 데이터프레임, 설명된 분산 비율, PCA 모델)
    """
    print(f"PCA 수행 중 (주성분 수: {n_components})...")
    
    if feature_columns is None:
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    X = df[feature_columns]
    
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X)
    
    # 변환된 데이터를 데이터프레임으로 변환
    transformed_columns = [f'PC{i+1}' for i in range(n_components)]
    df_transformed = pd.DataFrame(X_transformed, columns=transformed_columns, index=df.index)
    
    explained_variance_ratio = pca.explained_variance_ratio_
    
    print(f"PCA 완료. 설명된 분산 비율: {explained_variance_ratio}")
    print(f"총 설명된 분산: {sum(explained_variance_ratio):.4f}")
    
    return df_transformed, explained_variance_ratio, pca


# ============================================================================
# 통계 모델 모듈 (statsmodels)
# ============================================================================

def run_stats_model(df: pd.DataFrame, model_type: str, feature_columns: list, label_column: str):
    """
    statsmodels를 사용하여 통계 모델을 피팅합니다.
    Count regression 모델(Poisson, NegativeBinomial, QuasiPoisson)은 fit_count_regression_statsmodels를 사용합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터
    model_type : str
        모델 타입: 'OLS', 'Logistic', 'Poisson', 'NegativeBinomial', 'QuasiPoisson', 'Gamma', 'Tweedie'
    feature_columns : list
        특성 컬럼 리스트
    label_column : str
        레이블 컬럼 이름
    
    Returns:
    --------
    모델 결과 객체
    """
    # Count regression 모델의 경우 fit_count_regression_statsmodels 사용
    if model_type in ['Poisson', 'NegativeBinomial', 'QuasiPoisson']:
        max_iter = 100
        disp = 1.0
        model_results = fit_count_regression_statsmodels(
            df, model_type, feature_columns, label_column, max_iter, disp
        )
        
        # 통계량 출력
        print("\n=== 모델 통계량 ===")
        for key, value in model_results['metrics'].items():
            if value is not None:
                print(f"{key}: {value:.6f}")
        
        print("\n=== 계수 정보 ===")
        for param_name, coef_info in model_results['coefficients'].items():
            print(f"{param_name}:")
            print(f"  계수: {coef_info['coef']:.6f}")
            print(f"  표준 오차: {coef_info['std err']:.6f}")
            print(f"  z-통계량: {coef_info['z']:.6f}")
            print(f"  p-value: {coef_info['P>|z|']:.6f}")
            print(f"  신뢰구간: [{coef_info['[0.025']:.6f}, {coef_info['0.975]']:.6f}]")
        
        return model_results['results']
    
    # 다른 모델의 경우 기존 방식 사용
    print(f"{model_type} 모델 피팅 중...")
    
    X = df[feature_columns]
    y = df[label_column]
    X = sm.add_constant(X, prepend=True)
    
    if model_type == 'OLS':
        model = sm.OLS(y, X)
    elif model_type == 'Logistic':
        model = sm.Logit(y, X)
    elif model_type == 'Gamma':
        model = sm.GLM(y, X, family=sm.families.Gamma())
    elif model_type == 'Tweedie':
        model = sm.GLM(y, X, family=sm.families.Tweedie(var_power=1.5))
    else:
        print(f"오류: 알 수 없는 모델 타입 '{model_type}'")
        return None
    
    try:
        results = model.fit()
        print(f"\n--- {model_type} 모델 결과 ---")
        print(results.summary())
        return results
    except Exception as e:
        print(f"모델 피팅 중 오류 발생: {e}")
        return None


def predict_with_statsmodel(results, df: pd.DataFrame, feature_columns: list):
    """
    피팅된 statsmodels 모델로 예측을 수행합니다.
    
    Parameters:
    -----------
    results : 모델 결과 객체
        피팅된 모델 결과
    df : pd.DataFrame
        예측할 데이터
    feature_columns : list
        특성 컬럼 리스트
    
    Returns:
    --------
    pd.DataFrame
        예측 결과가 추가된 데이터프레임
    """
    print("statsmodels 모델로 예측 수행 중...")
    
    X = df[feature_columns]
    X = sm.add_constant(X, prepend=True, has_constant='add')
    
    # 모델의 특성 순서에 맞춰 정렬
    required_cols = results.model.exog_names
    X_aligned = X.reindex(columns=required_cols).fillna(0)
    
    predictions = results.predict(X_aligned)
    
    predict_df = df.copy()
    predict_df['Predict'] = predictions
    
    print("예측 완료. 'Predict' 컬럼이 추가되었습니다.")
    print(predict_df.head())
    
    return predict_df


def dispersion_checker(df: pd.DataFrame, feature_columns: list, label_column: str, max_iter: int = 100):
    """
    과대산포를 측정하고 적합한 모델을 추천합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터
    feature_columns : list
        특성 컬럼 리스트
    label_column : str
        레이블 컬럼 이름
    max_iter : int
        최대 반복 횟수
    
    Returns:
    --------
    dict
        분석 결과 딕셔너리
    """
    print("=== 과대산포 검사 (Diversion Checker) ===\n")
    
    # 1. 포아송 모델 적합
    print("1. 포아송 모델 적합 중...")
    poisson_result = fit_count_regression_statsmodels(
        df, 'Poisson', feature_columns, label_column, max_iter, 1.0
    )
    poisson_results = poisson_result['results']
    
    # 2. Dispersion φ 계산
    print("\n2. Dispersion φ 계산 중...")
    y = df[label_column].copy()
    mask = ~(df[feature_columns].isnull().any(axis=1) | y.isnull())
    y = y[mask]
    mu = poisson_results.mu
    pearson_resid = (y - mu) / np.sqrt(mu)
    phi = np.sum(pearson_resid**2) / (len(y) - len(feature_columns) - 1)
    
    print(f"Dispersion φ = {phi:.6f}")
    
    # 3. 모델 추천
    print("\n3. 모델 추천:")
    if phi < 1.2:
        recommendation = "Poisson"
        print(f"φ < 1.2 → Poisson 모델 추천")
    elif 1.2 <= phi < 2:
        recommendation = "QuasiPoisson"
        print(f"1.2 ≤ φ < 2 → Quasi-Poisson 모델 추천")
    else:
        recommendation = "NegativeBinomial"
        print(f"φ ≥ 2 → Negative Binomial 모델 추천")
    
    # 4. 포아송 vs 음이항 AIC 비교
    print("\n4. 포아송 vs 음이항 AIC 비교 (보조 기준):")
    poisson_aic = poisson_result['metrics'].get('AIC', None)
    print(f"Poisson AIC: {poisson_aic:.6f}" if poisson_aic else "Poisson AIC: N/A")
    
    print("음이항 모델 적합 중...")
    nb_result = fit_count_regression_statsmodels(
        df, 'NegativeBinomial', feature_columns, label_column, max_iter, 1.0
    )
    nb_aic = nb_result['metrics'].get('AIC', None)
    print(f"Negative Binomial AIC: {nb_aic:.6f}" if nb_aic else "Negative Binomial AIC: N/A")
    
    aic_comparison = None
    if poisson_aic is not None and nb_aic is not None:
        if nb_aic < poisson_aic:
            aic_comparison = "Negative Binomial이 더 낮은 AIC를 가짐 (더 나은 적합도)"
        else:
            aic_comparison = "Poisson이 더 낮은 AIC를 가짐 (더 나은 적합도)"
        print(f"AIC 비교: {aic_comparison}")
    
    # 5. Cameron–Trivedi test
    print("\n5. Cameron–Trivedi test (최종 확인):")
    # Cameron–Trivedi test: (y - mu)^2 - y를 종속변수로 하는 회귀
    X = df[feature_columns].copy()
    X = X[mask]
    X = sm.add_constant(X, prepend=True)
    
    # 테스트 통계량 계산
    test_stat = (y - mu)**2 - y
    ct_model = sm.OLS(test_stat, X)
    ct_results = ct_model.fit()
    
    # 상수항의 계수와 p-value 확인
    const_coef = ct_results.params.get('const', ct_results.params.iloc[0] if len(ct_results.params) > 0 else 0)
    const_pvalue = ct_results.pvalues.get('const', ct_results.pvalues.iloc[0] if len(ct_results.pvalues) > 0 else 1.0)
    
    print(f"Cameron–Trivedi test 통계량 (상수항 계수): {const_coef:.6f}")
    print(f"Cameron–Trivedi test p-value: {const_pvalue:.6f}")
    
    if const_pvalue < 0.05:
        ct_conclusion = "과대산포가 통계적으로 유의함 (p < 0.05)"
        print(f"결론: {ct_conclusion}")
    else:
        ct_conclusion = "과대산포가 통계적으로 유의하지 않음 (p ≥ 0.05)"
        print(f"결론: {ct_conclusion}")
    
    # 최종 추천
    print("\n=== 최종 추천 ===")
    print(f"추천 모델: {recommendation}")
    if aic_comparison:
        print(f"AIC 비교: {aic_comparison}")
    print(f"Cameron–Trivedi test: {ct_conclusion}")
    
    return {
        'phi': phi,
        'recommendation': recommendation,
        'poisson_aic': poisson_aic,
        'negative_binomial_aic': nb_aic,
        'aic_comparison': aic_comparison,
        'cameron_trivedi_coef': const_coef,
        'cameron_trivedi_pvalue': const_pvalue,
        'cameron_trivedi_conclusion': ct_conclusion,
        'methods_used': [
            '1. 포아송 모델 적합',
            '2. Dispersion φ 계산',
            '3. φ 기준 모델 추천',
            '4. 포아송 vs 음이항 AIC 비교',
            '5. Cameron–Trivedi test'
        ],
        'results': {
            'phi': phi,
            'phi_interpretation': f"φ = {phi:.6f}",
            'recommendation': recommendation,
            'poisson_aic': poisson_aic,
            'negative_binomial_aic': nb_aic,
            'cameron_trivedi_coef': const_coef,
            'cameron_trivedi_pvalue': const_pvalue,
            'cameron_trivedi_conclusion': ct_conclusion
        }
    }


# ============================================================================
# 재보험 분석 모듈
# ============================================================================

def fit_loss_distribution(df: pd.DataFrame, loss_column: str, dist_type: str = 'Pareto'):
    """
    손실 데이터에 통계 분포를 피팅합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터
    loss_column : str
        손실 컬럼 이름
    dist_type : str
        분포 타입: 'Pareto', 'Lognormal'
    
    Returns:
    --------
    tuple
        피팅된 분포 파라미터
    """
    print(f"{dist_type} 분포 피팅 중 (컬럼: {loss_column})...")
    loss_data = df[loss_column].dropna()
    
    if dist_type.lower() == 'pareto':
        params = stats.pareto.fit(loss_data, floc=0)
        print(f"Pareto 파라미터 (shape, loc, scale): {params}")
    elif dist_type.lower() == 'lognormal':
        params = stats.lognorm.fit(loss_data, floc=0)
        print(f"Lognormal 파라미터 (shape, loc, scale): {params}")
    else:
        print(f"오류: 지원하지 않는 분포 타입 '{dist_type}'")
        return None
    
    return params


def generate_exposure_curve(dist_type: str, params: tuple, total_loss: float):
    """
    피팅된 분포로부터 노출 곡선을 생성합니다.
    
    Parameters:
    -----------
    dist_type : str
        분포 타입
    params : tuple
        분포 파라미터
    total_loss : float
        총 손실
    
    Returns:
    --------
    list
        [(retention, loss_pct), ...] 형태의 곡선 데이터
    """
    print("노출 곡선 생성 중...")
    
    if dist_type.lower() == 'pareto':
        dist = stats.pareto(b=params[0], loc=params[1], scale=params[2])
    elif dist_type.lower() == 'lognormal':
        dist = stats.lognorm(s=params[0], loc=params[1], scale=params[2])
    else:
        raise ValueError(f"지원하지 않는 분포: {dist_type}")
    
    max_retention = total_loss * 2
    retention_points = np.linspace(0, max_retention, 100)
    loss_percentages = 1 - dist.cdf(retention_points)
    
    curve_data = list(zip(retention_points, loss_percentages))
    
    print("노출 곡선 생성 완료.")
    return curve_data


def price_xol_layer(curve_data: list, total_loss: float, retention: float, 
                   limit: float, loading_factor: float = 1.5):
    """
    노출 곡선을 사용하여 XoL 레이어의 가격을 책정합니다.
    
    Parameters:
    -----------
    curve_data : list
        노출 곡선 데이터
    total_loss : float
        총 손실
    retention : float
        자기부담금
    limit : float
        한도
    loading_factor : float
        로딩 팩터
    
    Returns:
    --------
    tuple
        (최종 보험료, 예상 레이어 손실, Rate on Line)
    """
    print(f"레이어 가격 책정: {limit:,.0f} xs {retention:,.0f}")
    
    retentions, loss_pcts = zip(*curve_data)
    
    pct_at_retention = np.interp(retention, retentions, loss_pcts)
    pct_at_limit_plus_retention = np.interp(retention + limit, retentions, loss_pcts)
    
    layer_loss_pct = pct_at_retention - pct_at_limit_plus_retention
    expected_layer_loss = total_loss * layer_loss_pct
    rate_on_line = (expected_layer_loss / limit) * 100 if limit > 0 else 0
    final_premium = expected_layer_loss * loading_factor
    
    print(f"  - 예상 레이어 손실: {expected_layer_loss:,.2f}")
    print(f"  - Rate on Line (RoL): {rate_on_line:.2f}%")
    print(f"  - 최종 보험료 (로딩 팩터 {loading_factor}): {final_premium:,.2f}")
    
    return final_premium, expected_layer_loss, rate_on_line


def apply_loss_threshold(df: pd.DataFrame, threshold: float, loss_column: str):
    """
    지정된 임계값 이상의 클레임만 필터링합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터
    threshold : float
        임계값
    loss_column : str
        손실 컬럼 이름
    
    Returns:
    --------
    pd.DataFrame
        필터링된 데이터프레임
    """
    print(f"임계값 {threshold:,.0f} 적용 중 (컬럼: {loss_column})...")
    original_rows = len(df)
    filtered_df = df[df[loss_column] >= threshold].copy()
    retained_rows = len(filtered_df)
    print(f"{original_rows}개 중 {retained_rows}개 클레임 유지")
    return filtered_df


def calculate_ceded_loss(df: pd.DataFrame, deductible: float, limit: float, loss_column: str):
    """
    각 클레임에 대한 재보험 인출 손실을 계산합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터
    deductible : float
        자기부담금
    limit : float
        한도
    loss_column : str
        손실 컬럼 이름
    
    Returns:
    --------
    pd.DataFrame
        인출 손실이 추가된 데이터프레임
    """
    print(f"인출 손실 계산 중: 레이어 {limit:,.0f} xs {deductible:,.0f}...")
    
    df['ceded_loss'] = df[loss_column].apply(
        lambda loss: min(limit, max(0, loss - deductible))
    )
    
    print("'ceded_loss' 컬럼이 추가되었습니다.")
    return df


def price_xol_contract(df: pd.DataFrame, contract: dict, volatility_loading: float,
                       year_column: str, ceded_loss_column: str):
    """
    경험 기반 방법으로 XoL 계약의 가격을 책정합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터
    contract : dict
        계약 조건 딕셔너리
    volatility_loading : float
        변동성 로딩 (%)
    year_column : str
        연도 컬럼 이름
    ceded_loss_column : str
        인출 손실 컬럼 이름
    
    Returns:
    --------
    float
        최종 보험료
    """
    print("경험 기반 XoL 계약 가격 책정 중...")
    
    yearly_ceded_losses = df.groupby(year_column)[ceded_loss_column].sum()
    print("\n연도별 인출 손실:")
    print(yearly_ceded_losses)
    
    expected_loss = yearly_ceded_losses.mean()
    loss_volatility = yearly_ceded_losses.std()
    
    volatility_margin = loss_volatility * (volatility_loading / 100)
    pure_premium = expected_loss + volatility_margin
    
    expense_ratio = contract.get('expense_ratio', 0.3)
    gross_premium = pure_premium / (1 - expense_ratio)
    
    print(f"\n--- 가격 책정 요약 ---")
    print(f"평균 연도별 인출 손실 (예상 손실): {expected_loss:,.2f}")
    print(f"연도별 손실 표준편차 (변동성): {loss_volatility:,.2f}")
    print(f"변동성 마진 ({volatility_loading}%): {volatility_margin:,.2f}")
    print(f"순 보험료 (손실 + 변동성): {pure_premium:,.2f}")
    print(f"총 보험료 ({expense_ratio*100:.1f}% 비용 로딩): {gross_premium:,.2f}")
    
    return gross_premium


# ============================================================================
# 실행 예제
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("통계 분석 모듈화 툴 - 파이썬 분석 코드")
    print("=" * 60)
    
    # 예제 1: 기본 데이터 분석 파이프라인
    print("\n" + "=" * 60)
    print("예제 1: 기본 데이터 분석 파이프라인")
    print("=" * 60)
    
    # 데이터 로드 (예제용 - 실제 파일 경로로 변경하세요)
    # df = load_data('your_data.csv')
    
    # 통계 분석
    # desc_stats, corr_matrix = analyze_statistics(df)
    
    # 결측치 처리
    # df_clean = handle_missing_values(df, method='impute', strategy='mean')
    
    # 데이터 분할
    # train_df, test_df = split_data(df_clean, train_size=0.7, random_state=42)
    
    # 모델 생성 및 훈련
    # model = create_linear_model('LinearRegression')
    # model = train_model(model, train_df, feature_columns=['feature1', 'feature2'], label_column='target')
    
    # 예측 및 평가
    # scored_df = score_model(model, test_df, feature_columns=['feature1', 'feature2'])
    # metrics = evaluate_model(model, scored_df, label_column='target', model_type='regression')
    
    print("\n예제 코드는 주석 처리되어 있습니다.")
    print("실제 사용 시 주석을 해제하고 파일 경로와 컬럼 이름을 수정하세요.")
    
    # 예제 2: 재보험 분석 파이프라인
    print("\n" + "=" * 60)
    print("예제 2: 재보험 분석 파이프라인")
    print("=" * 60)
    
    # XoL 데이터 로드
    # xol_df = load_xol_data('xol_claims.csv')
    
    # 임계값 적용
    # filtered_df = apply_loss_threshold(xol_df, threshold=100000, loss_column='loss')
    
    # 손실 분포 피팅
    # params = fit_loss_distribution(filtered_df, loss_column='loss', dist_type='Pareto')
    
    # 노출 곡선 생성
    # total_loss = filtered_df['loss'].sum()
    # curve = generate_exposure_curve('Pareto', params, total_loss)
    
    # 레이어 가격 책정
    # premium, expected_loss, rol = price_xol_layer(curve, total_loss, retention=1000000, limit=5000000)
    
    print("\n모든 모듈이 준비되었습니다!")
    print("각 함수를 개별적으로 호출하거나 파이프라인으로 연결하여 사용할 수 있습니다.")

