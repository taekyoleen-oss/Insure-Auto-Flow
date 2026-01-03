"""
앱의 Code 탭에 표시되는 SplitData 코드를 실행하여 검증
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# 앱의 Code 탭에서 생성되는 코드 (실제 실행 가능한 형태)
def split_data(df: pd.DataFrame, train_size: float, random_state: int, shuffle: bool, stratify: bool, stratify_col: str):
    """
    Splits data into training and testing sets using sklearn's train_test_split.
    """
    print("Splitting data...")
    print(f"  Train Size: {train_size}")
    print(f"  Random State: {random_state}")
    print(f"  Shuffle: {shuffle}")
    print(f"  Stratify: {stratify}")

    stratify_array = None
    if stratify and stratify_col and stratify_col != 'None':
        print(f"  Stratifying by column: {stratify_col}")
        stratify_array = df[stratify_col]
    elif stratify:
        print("  Warning: Stratify is True, but no stratify column was selected. Proceeding without stratification.")

    train_df, test_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify_array
    )
    
    print("Data split completed.")
    print(f"  Training set size: {len(train_df)} rows")
    print(f"  Testing set size: {len(test_df)} rows")
    
    return train_df, test_df

# Parameters from UI (앱에서 생성되는 형태)
p_train_size = 0.7
p_random_state = 42
p_shuffle = 'True' == 'True'  # True
p_stratify = 'False' == 'True'  # False
p_stratify_column = None

# Execution (주석 해제하여 실행)
def main():
    import os
    
    csv_path = os.path.join(os.path.dirname(__file__), 'compactiv.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'utils', 'Test', 'compactiv.csv')
    
    if not os.path.exists(csv_path):
        print(f"오류: 파일을 찾을 수 없습니다.")
        return
    
    dataframe = pd.read_csv(csv_path)
    print("=" * 80)
    print("앱의 Code 탭 코드 실행 검증")
    print("=" * 80)
    print(f"\n데이터 Shape: {dataframe.shape}")
    
    # 앱의 Code 탭 코드 실행
    train_data, test_data = split_data(
        dataframe,
        p_train_size,
        p_random_state,
        p_shuffle,
        p_stratify,
        p_stratify_column
    )
    
    print("\n" + "=" * 80)
    print("결과 요약")
    print("=" * 80)
    print(f"훈련 데이터: {len(train_data)} 행")
    print(f"테스트 데이터: {len(test_data)} 행")
    print(f"\n훈련 데이터 첫 5개 행 인덱스: {train_data.index[:5].tolist()}")
    print(f"테스트 데이터 첫 5개 행 인덱스: {test_data.index[:5].tolist()}")
    
    # Linear Regression으로 검증
    print("\n" + "=" * 80)
    print("Linear Regression으로 검증")
    print("=" * 80)
    
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
    target_column = dataframe.columns[-1]
    feature_columns = [col for col in dataframe.columns if col != target_column]
    
    X_train = train_data[feature_columns]
    y_train = train_data[target_column]
    X_test = test_data[feature_columns]
    y_test = test_data[target_column]
    
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    print(f"테스트 R²: {r2:.6f}")
    print(f"절편: {model.intercept_:.6f}")

if __name__ == "__main__":
    main()




























































