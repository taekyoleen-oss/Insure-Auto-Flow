import { CanvasModule, Connection } from './types';

const replacePlaceholders = (template: string, params: Record<string, any>): string => {
  let code = template;
  for (const key in params) {
    const placeholder = new RegExp(`{${key}}`, 'g');
    let value = params[key];
    // Stringify only if it's not already a string that looks like code
    if (value === null) {
        value = 'None';
    } else if (typeof value !== 'string' || !isNaN(Number(value))) {
        value = JSON.stringify(value);
    } else {
        value = `'${value}'`; // Wrap strings in quotes for Python
    }
    code = code.replace(placeholder, value);
  }
  return code;
};

const templates: Record<string, string> = {
    LoadData: `
import pandas as pd

# CSV 파일을 불러와서 DataFrame으로 반환합니다.
# Parameters from UI
file_path = {source}

# Execution
dataframe = pd.read_csv(file_path)
`,

    Statistics: `
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_statistics(df: pd.DataFrame):
    """
    데이터프레임에 대한 기술 통계량과 상관관계 분석을 수행합니다.
    """
    print("=" * 60)
    print("기술 통계량 분석")
    print("=" * 60)
    
    # 기술 통계량
    desc_stats = df.describe()
    print(desc_stats)
    
    # 결측치 정보
    print("\\n결측치 정보:")
    print(df.isnull().sum())
    
    # 상관관계 분석
    print("\\n" + "=" * 60)
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

# Assuming 'dataframe' is passed from the previous step
# Execution
# descriptive_statistics, correlation_matrix = analyze_statistics(dataframe)
`,

    SelectData: `
import pandas as pd

def select_data(df: pd.DataFrame, columns: list):
    """
    지정된 컬럼만 선택합니다.
    """
    print(f"컬럼 선택: {columns}")
    selected_df = df[columns].copy()
    print(f"선택 완료. Shape: {selected_df.shape}")
    return selected_df

# Assuming 'dataframe' is passed from the previous step
# Parameters from UI
# columnSelections is a dict: {column_name: {selected: bool, type: str}}
column_selections = {columnSelections}
selected_columns = [col for col, sel in column_selections.items() if sel.get('selected', True)]

# Execution
# selected_data = select_data(dataframe, selected_columns)
`,
    HandleMissingValues: `
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

def handle_missing_values(df: pd.DataFrame, method: str = 'remove_row', 
                         strategy: str = 'mean', columns: list = None,
                         n_neighbors: int = 5):
    """
    결측치를 처리합니다.
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

# Assuming 'dataframe' is passed from the previous step
# Parameters from UI
p_method = {method}
p_strategy = {strategy}
p_columns = {columns}
p_n_neighbors = {n_neighbors}

# Execution
# cleaned_data = handle_missing_values(dataframe, p_method, p_strategy, p_columns, p_n_neighbors)
`,
    EncodeCategorical: `
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_categorical(df: pd.DataFrame, method: str = 'label', 
                      columns: list = None, drop: str = 'first',
                      handle_unknown: str = 'ignore', ordinal_mapping: dict = None):
    """
    범주형 변수를 인코딩합니다.
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

# Assuming 'dataframe' is passed from the previous step
# Parameters from UI
p_method = {method}
p_columns = {columns}
p_drop = {drop}
p_handle_unknown = {handle_unknown}
p_ordinal_mapping = {ordinal_mapping}

# Execution
# encoded_data = encode_categorical(dataframe, p_method, p_columns, p_drop, p_handle_unknown, p_ordinal_mapping)
`,
    NormalizeData: `
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import pandas as pd
import numpy as np

def normalize_data(df: pd.DataFrame, method: str = 'MinMax', columns: list = None):
    """
    데이터를 정규화합니다.
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

# Assuming 'dataframe' is passed from the previous step
# Parameters from UI
# columnSelections is a dict: {column_name: {selected: bool}}
p_method = {method}
column_selections = {columnSelections}
p_columns = [col for col, sel in column_selections.items() if sel.get('selected', False) and col in dataframe.columns]

# Execution
# normalized_data = normalize_data(dataframe, p_method, p_columns)
`,
    TransitionData: `
import pandas as pd
import numpy as np

def transform_data(df: pd.DataFrame, transformations: dict):
    """
    수치형 컬럼에 수학적 변환을 적용합니다.
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

# Assuming 'dataframe' is passed from the previous step
# Parameters from UI are captured in a dictionary
p_transformations = {transformations}

# Execution
# transformed_data = transform_data(dataframe, p_transformations)
`,
    ResampleData: `
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

def resample_data(df: pd.DataFrame, method: str = 'SMOTE', target_column: str = None):
    """
    클래스 불균형을 처리하기 위해 데이터를 리샘플링합니다.
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

# Assuming 'dataframe' is passed from the previous step
# Parameters from UI
p_method = {method}
p_target_column = {target_column}

# Execution
# resampled_data = resample_data(dataframe, p_method, p_target_column)
`,
    SplitData: `
from sklearn.model_selection import train_test_split
import pandas as pd

# sklearn의 train_test_split을 사용하여 데이터를 분할합니다.
# Assuming 'dataframe' is passed from the previous step

# DataFrame 인덱스를 명시적으로 0부터 시작하도록 리셋
# 이는 동일한 random_state로 항상 동일한 결과를 보장하기 위함입니다.
df = dataframe.copy()
df.index = range(len(df))

# Parameters from UI
p_train_size = {train_size}
p_random_state = {random_state}
p_shuffle = {shuffle} == 'True'
p_stratify = {stratify} == 'True'
p_stratify_column = {stratify_column}

# Stratify 배열 준비
stratify_array = None
if p_stratify and p_stratify_column and p_stratify_column != 'None':
    stratify_array = df[p_stratify_column]

# 데이터 분할
train_data, test_data = train_test_split(
    df,
    train_size=p_train_size,
    random_state=p_random_state,
    shuffle=p_shuffle,
    stratify=stratify_array
)
`,

    LinearRegression: `
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

# This module creates a linear regression model instance.
# The model will be trained in the 'Train Model' module.
# Parameters from UI
p_model_type = {model_type}
p_fit_intercept = {fit_intercept} == 'True'
p_alpha = {alpha}
p_l1_ratio = {l1_ratio}

# Create model instance based on model type
if p_model_type == 'LinearRegression':
    model = LinearRegression(fit_intercept=p_fit_intercept)
elif p_model_type == 'Lasso':
    model = Lasso(alpha=p_alpha, fit_intercept=p_fit_intercept, random_state=42)
elif p_model_type == 'Ridge':
    model = Ridge(alpha=p_alpha, fit_intercept=p_fit_intercept, random_state=42)
elif p_model_type == 'ElasticNet':
    model = ElasticNet(alpha=p_alpha, l1_ratio=p_l1_ratio, fit_intercept=p_fit_intercept, random_state=42)
else:
    print(f"Unknown model type: {p_model_type}. Defaulting to LinearRegression.")
    model = LinearRegression(fit_intercept=p_fit_intercept)

print(f"{p_model_type} model instance created successfully.")
print(f"  Fit Intercept: {p_fit_intercept}")
if p_model_type in ['Lasso', 'Ridge', 'ElasticNet']:
    print(f"  Alpha: {p_alpha}")
if p_model_type == 'ElasticNet':
    print(f"  L1 Ratio: {p_l1_ratio}")

# Note: The model is not fitted here. It will be fitted in the 'Train Model' module.
# model variable contains the model instance ready for training.
`,

    DecisionTree: `
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def create_decision_tree(model_purpose: str = 'classification', criterion: str = 'gini',
                        max_depth: int = None, min_samples_split: int = 2, min_samples_leaf: int = 1):
    """
    의사결정나무 모델을 생성합니다.
    """
    print(f"의사결정나무 모델 생성 중 ({model_purpose})...")
    
    if model_purpose == 'classification':
        model = DecisionTreeClassifier(
            criterion=criterion.lower(),
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
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

# Parameters from UI
p_model_purpose = {model_purpose}
p_criterion = {criterion}
p_max_depth = {max_depth}
p_min_samples_split = {min_samples_split}
p_min_samples_leaf = {min_samples_leaf}

# Execution
# decision_tree_model = create_decision_tree(p_model_purpose, p_criterion, p_max_depth, p_min_samples_split, p_min_samples_leaf)
`,

    LogisticTradition: `
from sklearn.linear_model import LogisticRegression

def create_logistic_regression_model():
    """
    Creates a Logistic Regression model using scikit-learn.
    This model uses solvers like 'lbfgs' to find the optimal coefficients.
    It can handle regularization and is suitable for binary and multiclass classification.
    """
    print("Creating Logistic Regression model (sklearn.linear_model.LogisticRegression)...")

    # The parameters (e.g., penalty, C, solver) are hardcoded here for simplicity,
    # but could be exposed in the UI in a real application.
    model = LogisticRegression(random_state=42)
    
    print("Model created successfully.")
    return model

# This module defines the intent to use a scikit-learn LogisticRegression model.
# The actual training happens when this model is connected to a 'Train Model' module.
print("sklearn.linear_model.LogisticRegression model configured.")
`,

    TrainModel: `
import pandas as pd

# This module trains a model using the provided data.
# The model instance comes from a model definition module (e.g., LinearRegression module).
# Parameters from UI
p_feature_columns = {feature_columns}
p_label_column = {label_column}

# Assuming 'model' (from LinearRegression module) and 'dataframe' (from data source) are available
# Extract features and label from dataframe
X_train = dataframe[p_feature_columns]
y_train = dataframe[p_label_column]

# Train the model
trained_model = model.fit(X_train, y_train)

# The trained_model is now ready for use in Score Model or Evaluate Model modules
`,
    ScoreModel: `
import pandas as pd

# This module applies a trained model to a second dataset to generate predictions.
# Parameters from UI (if needed for feature selection)
# Note: Feature columns are typically inferred from the trained model

# Assuming 'trained_model' (from TrainModel module) and 'second_data' (second dataset) are available
# Extract feature columns from the trained model (sklearn models store feature names)
if hasattr(trained_model, 'feature_names_in_'):
    feature_columns = list(trained_model.feature_names_in_)
else:
    # Fallback: use all numeric columns except the label column
    # This assumes the second_data has the same structure as training data
    feature_columns = second_data.select_dtypes(include=['number']).columns.tolist()

# Prepare the second dataset features
X_second = second_data[feature_columns]

# Apply model.predict() to the second data
predictions = trained_model.predict(X_second)

# Add predictions to the second dataset
scored_data = second_data.copy()
scored_data['Predict'] = predictions

# The scored_data now contains the original data plus predictions
`,
    EvaluateModel: `
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np

# This module evaluates model performance using the scored data from ScoreModel.
# The scored data should contain both actual values (label_column) and predictions (prediction_column).
# Parameters from UI
p_label_column = {label_column}
p_prediction_column = {prediction_column}
p_model_type = {model_type}

# Assuming 'scored_data' (from ScoreModel module) is available
# Extract actual values and predictions
y_true = scored_data[p_label_column]
y_pred = scored_data[p_prediction_column]

# Calculate evaluation metrics based on model type
if p_model_type == 'classification':
    # Classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    evaluation_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    print("Classification Evaluation Metrics:")
    print(f"  Accuracy: {accuracy:.6f}")
    print(f"  Precision: {precision:.6f}")
    print(f"  Recall: {recall:.6f}")
    print(f"  F1-Score: {f1:.6f}")
    
else:  # regression
    # Regression metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    evaluation_metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    print("Regression Evaluation Metrics:")
    print(f"  Mean Squared Error (MSE): {mse:.6f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"  Mean Absolute Error (MAE): {mae:.6f}")
    print(f"  R-squared (R²): {r2:.6f}")

# evaluation_metrics contains all calculated statistics
`,
    OLSModel: `
import statsmodels.api as sm

# This module defines an OLS (Ordinary Least Squares) regression model.
# The model instance will be created in the 'Result Model' module using this definition.
# Parameters from UI: None (OLS has no configurable parameters)

print("OLS Model definition created successfully.")
print("This model will be instantiated and fitted using statsmodels.OLS in the Result Model module.")
`,
    LogisticModel: `
import statsmodels.api as sm

# This module defines a Logistic regression model.
# The model instance will be created in the 'Result Model' module using this definition.
# Parameters from UI: None (Logistic has no configurable parameters)

print("Logistic Model definition created successfully.")
print("This model will be instantiated and fitted using statsmodels.Logit in the Result Model module.")
`,
    PoissonModel: `
import statsmodels.api as sm

# This module defines a Poisson regression model.
# The model instance will be created in the 'Result Model' module using this definition.
# Parameters from UI
p_max_iter = {max_iter}

print(f"Poisson Model definition created successfully (max_iter={p_max_iter}).")
print("This model will be instantiated and fitted using statsmodels.Poisson in the Result Model module.")
`,
    QuasiPoissonModel: `
import statsmodels.api as sm

# This module defines a Quasi-Poisson regression model.
# The model instance will be created in the 'Result Model' module using this definition.
# Parameters from UI
p_max_iter = {max_iter}

print(f"Quasi-Poisson Model definition created successfully (max_iter={p_max_iter}).")
print("This model will be instantiated and fitted using statsmodels.GLM with Poisson family in the Result Model module.")
`,
    NegativeBinomialModel: `
import statsmodels.api as sm

# This module defines a Negative Binomial regression model.
# The model instance will be created in the 'Result Model' module using this definition.
# Parameters from UI
p_max_iter = {max_iter}
p_disp = {disp}

print(f"Negative Binomial Model definition created successfully (max_iter={p_max_iter}, disp={p_disp}).")
print("This model will be instantiated and fitted using statsmodels.NegativeBinomial in the Result Model module.")
`,
    StatModels: `
import statsmodels.api as sm

# This module configures advanced statistical models (Gamma, Tweedie) from the statsmodels library.
# The model instance will be created in the 'Result Model' module using this definition.
# Parameters from UI
selected_model_type = {model}

print(f"Stat Models definition created successfully (model type: {selected_model_type}).")
print("This model will be instantiated and fitted in the Result Model module.")
`,

    ResultModel: `
import pandas as pd
import numpy as np
import statsmodels.api as sm

def run_stats_model(df: pd.DataFrame, model_type: str, feature_columns: list, label_column: str, max_iter: int = 100, disp: float = 1.0):
    """
    statsmodels를 사용하여 통계 모델을 피팅합니다.
    """
    print(f"{model_type} 모델 피팅 중...")
    
    X = df[feature_columns]
    y = df[label_column]
    X = sm.add_constant(X, prepend=True)
    
    if model_type == 'OLS':
        model = sm.OLS(y, X)
    elif model_type == 'Logit' or model_type == 'Logistic':
        model = sm.Logit(y, X)
    elif model_type == 'Poisson':
        model = sm.Poisson(y, X)
        results = model.fit(maxiter=max_iter)
        print(f"\\n--- {model_type} 모델 결과 ---")
        print(results.summary())
        return results
    elif model_type == 'QuasiPoisson':
        model = sm.GLM(y, X, family=sm.families.Poisson())
        results = model.fit(maxiter=max_iter)
        # Quasi-Poisson은 분산을 과분산 파라미터로 조정
        mu = results.mu
        pearson_resid = (y - mu) / np.sqrt(mu)
        phi = np.sum(pearson_resid**2) / (len(y) - len(feature_columns) - 1)
        results.scale = phi
        print(f"\\n--- {model_type} 모델 결과 ---")
        print(results.summary())
        return results
    elif model_type == 'NegativeBinomial':
        model = sm.NegativeBinomial(y, X, loglike_method='nb2')
        results = model.fit(maxiter=max_iter, disp=disp)
        print(f"\\n--- {model_type} 모델 결과 ---")
        print(results.summary())
        return results
    elif model_type == 'Gamma':
        model = sm.GLM(y, X, family=sm.families.Gamma())
    elif model_type == 'Tweedie':
        model = sm.GLM(y, X, family=sm.families.Tweedie(var_power=1.5))
    else:
        print(f"오류: 알 수 없는 모델 타입 '{model_type}'")
        return None
    
    try:
        results = model.fit()
        print(f"\\n--- {model_type} 모델 결과 ---")
        print(results.summary())
        return results
    except Exception as e:
        print(f"모델 피팅 중 오류 발생: {e}")
        return None

# Assuming 'dataframe' is passed from a data module.
# The 'model_type' would be passed from the connected model definition module.
# Parameters from UI
p_feature_columns = {feature_columns}
p_label_column = {label_column}
# p_model_type = 'OLS'  # This would be set dynamically based on model definition output

# Execution
# model_results = run_stats_model(
#     dataframe,
#     p_model_type,
#     p_feature_columns,
#     p_label_column
# )
`,
    PredictModel: `
import pandas as pd
import statsmodels.api as sm

def predict_with_statsmodel(results, df: pd.DataFrame):
    """
    Applies a fitted statsmodels result object to a new dataset to generate predictions.
    """
    print("Generating predictions with the fitted statsmodels model...")
    
    # Ensure the 'const' column is present for the intercept
    df_with_const = sm.add_constant(df, prepend=True, has_constant='raise')
    
    # Ensure columns in the prediction data match the model's exog names
    # and are in the same order.
    required_cols = results.model.exog_names
    df_aligned = df_with_const.reindex(columns=required_cols).fillna(0)

    predictions = results.predict(df_aligned)
    
    predict_df = df.copy()
    predict_df['Predict'] = predictions
    
    print("Prediction complete. 'Predict' column added.")
    print(predict_df.head())
    
    return predict_df

# Assuming 'model_results' (from ResultModel) and a dataframe 'data_to_predict' are available
#
# Execution
# predicted_data = predict_with_statsmodel(model_results, data_to_predict)
`,
    FitLossDistribution: `
from scipy import stats
import pandas as pd

def fit_loss_distribution(df: pd.DataFrame, loss_column: str, dist_type: str = 'Pareto'):
    """
    손실 데이터에 통계 분포를 피팅합니다.
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

# Assuming 'dataframe' is passed from a previous step
# Parameters from UI
p_loss_column = {loss_column}
p_dist_type = {distribution_type}

# Execution
# fitted_params = fit_loss_distribution(dataframe, p_loss_column, p_dist_type)
`,

    GenerateExposureCurve: `
import numpy as np
from scipy import stats

def generate_exposure_curve(dist_type: str, params: tuple, total_loss: float):
    """
    피팅된 분포로부터 노출 곡선을 생성합니다.
    """
    print("노출 곡선 생성 중...")
    
    if dist_type.lower() == 'pareto':
        dist = stats.pareto(b=params[0], loc=params[1], scale=params[2])
    elif dist_type.lower() == 'lognormal':
        dist = stats.lognorm(s=params[0], loc=params[1], scale=params[2])
    else:
        raise ValueError(f"지원하지 않는 분포: {dist_type}")
    
    max_retention = total_loss * 2  # Go beyond total loss for a full curve
    retention_points = np.linspace(0, max_retention, 100)
    loss_percentages = 1 - dist.cdf(retention_points)
    
    curve_data = list(zip(retention_points, loss_percentages))
    
    print("노출 곡선 생성 완료.")
    return curve_data

# Assuming 'fitted_params' and 'total_loss' are available
# p_dist_type = 'Pareto'  # From FitLossDistribution module
# p_total_loss = 50000000  # From input data
#
# Execution
# exposure_curve = generate_exposure_curve(p_dist_type, fitted_params, p_total_loss)
`,

    PriceXoLLayer: `
import numpy as np

def price_xol_layer(curve_data: list, total_loss: float, retention: float, 
                   limit: float, loading_factor: float = 1.5):
    """
    노출 곡선을 사용하여 XoL 레이어의 가격을 책정합니다.
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

# Assuming 'exposure_curve' and 'total_loss' are available
# Parameters from UI
p_retention = {retention}
p_limit = {limit}
p_loading_factor = {loading_factor}
# p_total_loss = 50000000  # from GenerateExposureCurve step

# Execution
# premium, _, _ = price_xol_layer(exposure_curve, p_total_loss, p_retention, p_limit, p_loading_factor)
`,
    XolLoading: `
import pandas as pd

# This is identical to the standard LoadData module but conceptually used for XoL data.
def load_xol_data(file_path: str):
    """
    Loads claims data from a CSV file, expecting columns like 'year', 'loss'.
    """
    print(f"Loading XoL claims data from: {file_path}")
    df = pd.read_csv(file_path)
    print("XoL data loaded successfully.")
    return df

# Parameters from UI
p_file_path = {source}

# Execution
# xol_dataframe = load_xol_data(p_file_path)
`,

    ApplyThreshold: `
import pandas as pd

def apply_loss_threshold(df: pd.DataFrame, threshold: float, loss_col: str):
    """
    Filters out claims that are below the specified threshold.
    """
    print(f"Applying threshold of {threshold:,.0f} to column '{loss_col}'...")
    original_rows = len(df)
    filtered_df = df[df[loss_col] >= threshold].copy()
    retained_rows = len(filtered_df)
    print(f"Retained {retained_rows} of {original_rows} claims.")
    return filtered_df

# Assuming 'xol_dataframe' is passed from the previous step
# Parameters from UI
p_threshold = {threshold}
p_loss_column = {loss_column}

# Execution
# large_claims_df = apply_loss_threshold(xol_dataframe, p_threshold, p_loss_column)
`,

    DefineXolContract: `
# This module defines the parameters for an Excess of Loss (XoL) reinsurance contract.
# These parameters are then used by downstream modules.

# Parameters from UI
p_deductible = {deductible}  # Also known as retention
p_limit = {limit}
p_reinstatements = {reinstatements}
p_agg_deductible = {aggDeductible}
p_expense_ratio = {expenseRatio}

contract_terms = {
    'deductible': p_deductible,
    'limit': p_limit,
    'reinstatements': p_reinstatements,
    'agg_deductible': p_agg_deductible,
    'expense_ratio': p_expense_ratio,
}

print("XoL Contract terms defined:")
print(contract_terms)
`,

    CalculateCededLoss: `
import pandas as pd

def calculate_ceded_loss(df: pd.DataFrame, deductible: float, limit: float, loss_col: str):
    """
    Calculates the ceded loss for each claim based on the contract's deductible and limit.
    """
    print(f"Calculating ceded loss for layer {limit:,.0f} xs {deductible:,.0f}...")
    
    # Ceded loss is the portion of the loss above the deductible, up to the limit.
    df['ceded_loss'] = df[loss_col].apply(
        lambda loss: min(limit, max(0, loss - deductible))
    )
    
    print("'ceded_loss' column added to the dataframe.")
    return df

# Assuming 'large_claims_df' and 'contract_terms' are passed from previous steps
# Parameters from UI
p_loss_column = {loss_column}
# contract_deductible = contract_terms['deductible']
# contract_limit = contract_terms['limit']

# Execution
# ceded_df = calculate_ceded_loss(large_claims_df, contract_deductible, contract_limit, p_loss_column)
`,

    PriceXolContract: `
import pandas as pd
import numpy as np

def price_xol_contract(df: pd.DataFrame, contract: dict, volatility_loading: float,
                       year_column: str, ceded_loss_column: str):
    """
    경험 기반 방법으로 XoL 계약의 가격을 책정합니다.
    """
    print("경험 기반 XoL 계약 가격 책정 중...")
    
    yearly_ceded_losses = df.groupby(year_column)[ceded_loss_column].sum()
    print("\\n연도별 인출 손실:")
    print(yearly_ceded_losses)
    
    expected_loss = yearly_ceded_losses.mean()
    loss_volatility = yearly_ceded_losses.std()
    
    volatility_margin = loss_volatility * (volatility_loading / 100)
    pure_premium = expected_loss + volatility_margin
    
    expense_ratio = contract.get('expense_ratio', 0.3)
    gross_premium = pure_premium / (1 - expense_ratio)
    
    print(f"\\n--- 가격 책정 요약 ---")
    print(f"평균 연도별 인출 손실 (예상 손실): {expected_loss:,.2f}")
    print(f"연도별 손실 표준편차 (변동성): {loss_volatility:,.2f}")
    print(f"변동성 마진 ({volatility_loading}%): {volatility_margin:,.2f}")
    print(f"순 보험료 (손실 + 변동성): {pure_premium:,.2f}")
    print(f"총 보험료 ({expense_ratio*100:.1f}% 비용 로딩): {gross_premium:,.2f}")
    
    return gross_premium

# Assuming 'ceded_df' and 'contract_terms' are passed from previous steps
# Parameters from UI
p_volatility_loading = {volatility_loading}
p_year_column = {year_column}
p_ceded_loss_column = {ceded_loss_column}

# Execution
# final_price = price_xol_contract(
#     ceded_df, 
#     contract_terms, 
#     p_volatility_loading, 
#     p_year_column, 
#     p_ceded_loss_column
# )
`,
    DiversionChecker: `
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Parameters from UI
p_feature_columns = {feature_columns}
p_label_column = {label_column}
p_max_iter = {max_iter}

# Assuming 'dataframe' is passed from a data module.
# Extract features and label
X = dataframe[p_feature_columns]
y = dataframe[p_label_column]

# 과대산포 검사 (Diversion Checker) 실행
# 이 모듈은 dispersion_checker 함수를 사용하여 과대산포를 측정하고
# 적합한 회귀 모델을 추천합니다.

print("=== 과대산포 검사 (Diversion Checker) ===")
print("이 모듈은 다음을 수행합니다:")
print("1. 포아송 모델 적합")
print("2. Dispersion φ 계산")
print("3. φ 기준 모델 추천")
print("4. 포아송 vs 음이항 AIC 비교")
print("5. Cameron–Trivedi test")
print("\\n실제 실행은 'Run' 버튼을 클릭하면 수행됩니다.")
`,
    KNN: `
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

def create_knn_model(model_purpose: str = 'classification', n_neighbors: int = 3,
                     weights: str = 'uniform', algorithm: str = 'auto', metric: str = 'minkowski'):
    """
    Creates a K-Nearest Neighbors model using scikit-learn.
    This model uses k nearest neighbors to make predictions.
    It can handle both classification and regression tasks.
    """
    print(f"Creating K-Nearest Neighbors model ({model_purpose})...")
    
    if model_purpose == 'classification':
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            metric=metric
        )
    else:
        model = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            metric=metric
        )
    
    print("Model created successfully.")
    return model

# Parameters from UI
p_model_purpose = {model_purpose}
p_n_neighbors = {n_neighbors}
p_weights = {weights}
p_algorithm = {algorithm}
p_metric = {metric}

# Execution
# knn_model = create_knn_model(p_model_purpose, p_n_neighbors, p_weights, p_algorithm, p_metric)

# This module defines the intent to use a scikit-learn KNeighborsClassifier/KNeighborsRegressor model.
# The actual training happens when this model is connected to a 'Train Model' module.
print(f"sklearn.neighbors.KNeighbors{'Classifier' if p_model_purpose == 'classification' else 'Regressor'} model configured.")
`,
};

export const getModuleCode = (
    module: CanvasModule | null,
    allModules?: CanvasModule[],
    connections?: Connection[]
): string => {
    if (!module) {
        return "# Select a module to view its Python code.";
    }
    
    // EvaluateStat의 경우 generateEvaluateStatCode 사용
    if (module.type === "EvaluateStat") {
        return generateEvaluateStatCode(module);
    }
    
    // ResultModel의 경우 연결된 모델 타입에 따라 코드 생성
    if (module.type === "ResultModel" && allModules && connections) {
        const modelInputConnection = connections.find(
            (c) => c.to.moduleId === module.id && c.to.portName === "model_in"
        );
        
        if (modelInputConnection) {
            const modelSourceModule = allModules.find(
                (m) => m.id === modelInputConnection.from.moduleId
            );
            
            if (modelSourceModule) {
                // 모델 타입 확인
                let modelType: string | null = null;
                
                if (modelSourceModule.type === "OLSModel") {
                    modelType = "OLS";
                } else if (modelSourceModule.type === "LogisticModel") {
                    modelType = "Logit";
                } else if (modelSourceModule.type === "PoissonModel") {
                    modelType = "Poisson";
                } else if (modelSourceModule.type === "QuasiPoissonModel") {
                    modelType = "QuasiPoisson";
                } else if (modelSourceModule.type === "NegativeBinomialModel") {
                    modelType = "NegativeBinomial";
                } else if (modelSourceModule.type === "StatModels") {
                    modelType = modelSourceModule.parameters.model || "Gamma";
                } else if (modelSourceModule.outputData?.type === "ModelDefinitionOutput") {
                    modelType = modelSourceModule.outputData.modelType;
                }
                
                if (modelType) {
                    return generateResultModelCode(module, modelType, modelSourceModule.parameters);
                }
            }
        }
    }
    
    const template = templates[module.type] || `# Code for ${module.name} is not available.`;
    return replacePlaceholders(template.trim(), module.parameters);
};

/**
 * ResultModel의 모델 타입에 맞는 코드를 생성합니다
 */
function generateResultModelCode(
    module: CanvasModule,
    modelType: string,
    modelParams: Record<string, any>
): string {
    const { feature_columns, label_column } = module.parameters;
    const max_iter = modelParams.max_iter || 100;
    const disp = modelParams.disp || 1.0;
    
    let code = `import pandas as pd
import numpy as np
import statsmodels.api as sm

# Parameters from UI (Result Model)
p_feature_columns = ${JSON.stringify(feature_columns || [])}
p_label_column = ${label_column ? `'${label_column}'` : 'None'}

# Assuming 'dataframe' is passed from a data module and 'model_definition' is passed from the model definition module.
# Extract features and label
X = dataframe[p_feature_columns]
y = dataframe[p_label_column]
X = sm.add_constant(X, prepend=True)

# Create model instance based on the connected model definition module
`;

    // 모델 타입에 따라 코드 생성 (연결된 모델 정의 모듈의 타입에 따라)
    if (modelType === "OLS") {
        code += `# OLS 모델 인스턴스 생성 및 피팅
# (모델 정의는 OLS Model 모듈에서 제공됨)
model = sm.OLS(y, X)
results = model.fit()

print("\\n--- OLS 모델 결과 ---")
print(results.summary())

model_results = results
`;
    } else if (modelType === "Logit" || modelType === "Logistic") {
        code += `# Logistic 모델 인스턴스 생성 및 피팅
# (모델 정의는 Logistic Model 모듈에서 제공됨)
model = sm.Logit(y, X)
results = model.fit()

print("\\n--- Logistic 모델 결과 ---")
print(results.summary())

model_results = results
`;
    } else if (modelType === "Poisson") {
        code += `# Poisson 모델 인스턴스 생성 및 피팅
# (모델 정의는 Poisson Model 모듈에서 제공됨, max_iter=${max_iter})
model = sm.Poisson(y, X)
results = model.fit(maxiter=${max_iter})

print("\\n--- Poisson 모델 결과 ---")
print(results.summary())

model_results = results
`;
    } else if (modelType === "QuasiPoisson") {
        code += `# Quasi-Poisson 모델 인스턴스 생성 및 피팅
# (모델 정의는 Quasi-Poisson Model 모듈에서 제공됨, max_iter=${max_iter})
model = sm.GLM(y, X, family=sm.families.Poisson())
results = model.fit(maxiter=${max_iter})

# Quasi-Poisson은 분산을 과분산 파라미터로 조정
mu = results.mu
pearson_resid = (y - mu) / np.sqrt(mu)
phi = np.sum(pearson_resid**2) / (len(y) - len(p_feature_columns) - 1)
results.scale = phi

print("\\n--- Quasi-Poisson 모델 결과 ---")
print(results.summary())

model_results = results
`;
    } else if (modelType === "NegativeBinomial") {
        code += `# Negative Binomial 모델 인스턴스 생성 및 피팅
# (모델 정의는 Negative Binomial Model 모듈에서 제공됨, max_iter=${max_iter}, disp=${disp})
model = sm.NegativeBinomial(y, X, loglike_method='nb2')
results = model.fit(maxiter=${max_iter}, disp=${disp})

print("\\n--- Negative Binomial 모델 결과 ---")
print(results.summary())

model_results = results
`;
    } else if (modelType === "Gamma") {
        code += `# Gamma 모델 인스턴스 생성 및 피팅
# (모델 정의는 Stat Models 모듈에서 제공됨, model type: Gamma)
model = sm.GLM(y, X, family=sm.families.Gamma())
results = model.fit()

print("\\n--- Gamma 모델 결과 ---")
print(results.summary())

model_results = results
`;
    } else if (modelType === "Tweedie") {
        code += `# Tweedie 모델 인스턴스 생성 및 피팅
# (모델 정의는 Stat Models 모듈에서 제공됨, model type: Tweedie)
model = sm.GLM(y, X, family=sm.families.Tweedie(var_power=1.5))
results = model.fit()

print("\\n--- Tweedie 모델 결과 ---")
print(results.summary())

model_results = results
`;
    } else {
        code += `# 알 수 없는 모델 타입: ${modelType}
print(f"오류: 알 수 없는 모델 타입 '${modelType}'")
model_results = None
`;
    }
    
    return code;
}

/**
 * EvaluateStat 모듈의 코드를 생성합니다
 */
function generateEvaluateStatCode(
    module: CanvasModule
): string {
    const { label_column, prediction_column, model_type } = module.parameters;
    
    let code = `import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Parameters from UI
p_label_column = ${label_column ? `'${label_column}'` : 'None'}
p_prediction_column = ${prediction_column ? `'${prediction_column}'` : 'None'}
p_model_type = ${model_type ? `'${model_type}'` : 'None'}

# Assuming 'dataframe' is passed from a data module.
# Extract actual and predicted values
y_true = dataframe[p_label_column].values
y_pred = dataframe[p_prediction_column].values

# 기본 통계량 계산 (전통적인 방법)
print("=" * 60)
print("통계 모델 평가 (Evaluate Stat)")
print("=" * 60)

# 기본 회귀 메트릭
mse = float(mean_squared_error(y_true, y_pred))
rmse = float(np.sqrt(mse))
mae = float(mean_absolute_error(y_true, y_pred))
r2 = float(r2_score(y_true, y_pred))

print(f"\\n--- 기본 통계량 ---")
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"R-squared: {r2:.6f}")

# 잔차 계산
residuals = (y_true - y_pred).tolist()
residuals_array = np.array(residuals)

print(f"\\n--- 잔차 통계량 ---")
print(f"Mean Residual: {np.mean(residuals_array):.6f}")
print(f"Std Residual: {np.std(residuals_array):.6f}")
print(f"Min Residual: {np.min(residuals_array):.6f}")
print(f"Max Residual: {np.max(residuals_array):.6f}")

# 모델 타입별 특수 통계량 (선택적)
`;

    if (model_type) {
        code += `
# 모델 타입: ${model_type}
if p_model_type and p_model_type != '' and p_model_type != 'None':
    print(f"\\n--- ${model_type} 모델 특수 통계량 ---")
    
    if p_model_type in ['Poisson', 'NegativeBinomial', 'QuasiPoisson']:
        # Count regression 모델 통계량
        mu = np.maximum(y_pred, 1e-10)  # 0 방지
        deviance_val = 2 * np.sum(y_true * np.log(np.maximum(y_true, 1e-10) / mu) - (y_true - mu))
        deviance = float(deviance_val)
        
        # Pearson chi2
        pearson_resid = (y_true - mu) / np.sqrt(mu)
        pearson_chi2_val = np.sum(pearson_resid ** 2)
        pearson_chi2 = float(pearson_chi2_val)
        
        # Dispersion (phi)
        n = len(y_true)
        p = 1  # 간단히 1로 가정 (실제로는 모델의 파라미터 수)
        dispersion_val = pearson_chi2_val / (n - p) if (n - p) > 0 else 1.0
        dispersion = float(dispersion_val)
        
        if p_model_type == 'Poisson':
            # Log-likelihood (Poisson)
            log_likelihood_val = np.sum(stats.poisson.logpmf(y_true, mu))
            log_likelihood = float(log_likelihood_val)
            
            # AIC, BIC (근사치)
            aic = float(-2 * log_likelihood_val + 2 * p)
            bic = float(-2 * log_likelihood_val + np.log(n) * p)
            
            print(f"Deviance: {deviance:.6f}")
            print(f"Pearson chi²: {pearson_chi2:.6f}")
            print(f"Dispersion (φ): {dispersion:.6f}")
            print(f"Log-Likelihood: {log_likelihood:.6f}")
            print(f"AIC: {aic:.6f}")
            print(f"BIC: {bic:.6f}")
        else:
            print(f"Deviance: {deviance:.6f}")
            print(f"Pearson chi²: {pearson_chi2:.6f}")
            print(f"Dispersion (φ): {dispersion:.6f}")
    
    elif p_model_type in ['Logistic', 'Logit']:
        # Logistic regression 통계량
        y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
        y_true_clipped = np.clip(y_true, 1e-10, 1 - 1e-10)
        deviance_val = -2 * np.sum(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        deviance = float(deviance_val)
        
        # Pearson chi2
        pearson_resid = (y_true - y_pred) / np.sqrt(y_pred * (1 - y_pred) + 1e-10)
        pearson_chi2_val = np.sum(pearson_resid ** 2)
        pearson_chi2 = float(pearson_chi2_val)
        
        # Log-likelihood
        log_likelihood_val = np.sum(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        log_likelihood = float(log_likelihood_val)
        
        n = len(y_true)
        p = 1
        aic = float(-2 * log_likelihood_val + 2 * p)
        bic = float(-2 * log_likelihood_val + np.log(n) * p)
        
        print(f"Deviance: {deviance:.6f}")
        print(f"Pearson chi²: {pearson_chi2:.6f}")
        print(f"Log-Likelihood: {log_likelihood:.6f}")
        print(f"AIC: {aic:.6f}")
        print(f"BIC: {bic:.6f}")
    
    elif p_model_type == 'OLS':
        # OLS 통계량
        deviance_val = np.sum((y_true - y_pred) ** 2)
        deviance = float(deviance_val)
        
        # Log-likelihood (normal distribution)
        n = len(y_true)
        sigma2 = deviance_val / n if n > 0 else 1.0
        log_likelihood_val = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
        log_likelihood = float(log_likelihood_val)
        
        p = 1
        aic = float(-2 * log_likelihood_val + 2 * p)
        bic = float(-2 * log_likelihood_val + np.log(n) * p)
        
        print(f"Deviance (Residual Sum of Squares): {deviance:.6f}")
        print(f"Log-Likelihood: {log_likelihood:.6f}")
        print(f"AIC: {aic:.6f}")
        print(f"BIC: {bic:.6f}")

print("\\n" + "=" * 60)
print("평가 완료")
print("=" * 60)
`;
    } else {
        code += `
print("\\n모델 타입이 지정되지 않아 기본 통계량만 계산되었습니다.")
print("모델 타입을 지정하면 추가 통계량(Deviance, AIC, BIC 등)을 계산할 수 있습니다.")

print("\\n" + "=" * 60)
print("평가 완료")
print("=" * 60)
`;
    }
    
    return code;
}