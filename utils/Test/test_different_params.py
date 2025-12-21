"""
다양한 파라미터로 테스트하여 앱의 결과 0.6857과 일치하는지 확인
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def app_split_data(df, train_size=0.7, random_state=42, shuffle=True):
    """앱의 SplitData 로직 재현"""
    input_rows = df.values.tolist()
    indices = list(range(len(input_rows)))
    
    if shuffle:
        seed = int(random_state)
        
        def random():
            nonlocal seed
            seed = (seed * 9301 + 49297) % 233280
            return seed / 233280
        
        for i in range(len(indices) - 1, 0, -1):
            j = int(random() * (i + 1))
            indices[i], indices[j] = indices[j], indices[i]
    
    train_count = int(len(input_rows) * float(train_size))
    train_indices = indices[:train_count]
    test_indices = indices[train_count:]
    
    train_rows = [input_rows[i] for i in train_indices]
    test_rows = [input_rows[i] for i in test_indices]
    
    train_df = pd.DataFrame(train_rows, columns=df.columns)
    test_df = pd.DataFrame(test_rows, columns=df.columns)
    
    return train_df, test_df

def test_model(train_df, test_df, feature_columns, target_column):
    """모델 훈련 및 평가"""
    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    X_test = test_df[feature_columns]
    y_test = test_df[target_column]
    
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return r2, rmse, model.intercept_

def main():
    csv_path = r'utils/Test/compactiv.csv'
    df = pd.read_csv(csv_path)
    
    target_column = df.columns[-1]
    feature_columns = [col for col in df.columns if col != target_column]
    
    app_target_r2 = 0.6857
    
    print("=" * 80)
    print("다양한 파라미터로 테스트")
    print("=" * 80)
    
    # 다양한 train_size 테스트
    print("\n" + "-" * 80)
    print("다양한 train_size로 테스트 (random_state=42)")
    print("-" * 80)
    
    for train_size in [0.6, 0.65, 0.7, 0.75, 0.8]:
        train_df, test_df = app_split_data(df, train_size=train_size, random_state=42, shuffle=True)
        r2, rmse, intercept = test_model(train_df, test_df, feature_columns, target_column)
        diff = abs(r2 - app_target_r2)
        print(f"train_size={train_size:.2f}: R²={r2:.6f}, RMSE={rmse:.6f}, 차이={diff:.6f}")
        if diff < 0.001:
            print(f"  *** 일치! ***")
    
    # 다양한 random_state 테스트
    print("\n" + "-" * 80)
    print("다양한 random_state로 테스트 (train_size=0.7)")
    print("-" * 80)
    
    for rs in [40, 41, 42, 43, 44]:
        train_df, test_df = app_split_data(df, train_size=0.7, random_state=rs, shuffle=True)
        r2, rmse, intercept = test_model(train_df, test_df, feature_columns, target_column)
        diff = abs(r2 - app_target_r2)
        print(f"random_state={rs}: R²={r2:.6f}, RMSE={rmse:.6f}, 차이={diff:.6f}")
        if diff < 0.001:
            print(f"  *** 일치! ***")

if __name__ == "__main__":
    main()


















































