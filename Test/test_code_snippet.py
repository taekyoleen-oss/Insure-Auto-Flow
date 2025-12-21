"""
앱의 Code 탭에 표시되는 SplitData 코드를 실행하여 검증하는 스크립트
"""

import pandas as pd
from sklearn.model_selection import train_test_split

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

def main():
    import os
    
    # CSV 파일 경로
    csv_path = os.path.join(os.path.dirname(__file__), 'compactiv.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'utils', 'Test', 'compactiv.csv')
    
    if not os.path.exists(csv_path):
        print(f"오류: 파일을 찾을 수 없습니다: {csv_path}")
        return
    
    # 데이터 로드
    print("=" * 80)
    print("앱의 Code 탭 코드 검증")
    print("=" * 80)
    print(f"\n파일: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"데이터 Shape: {df.shape}")
    
    # 앱의 Code 탭에서 생성되는 파라미터
    p_train_size = 0.7
    p_random_state = 42
    p_shuffle = 'True' == 'True'  # True
    p_stratify = 'False' == 'True'  # False
    p_stratify_column = None
    
    print(f"\n파라미터:")
    print(f"  train_size: {p_train_size}")
    print(f"  random_state: {p_random_state}")
    print(f"  shuffle: {p_shuffle}")
    print(f"  stratify: {p_stratify}")
    print(f"  stratify_column: {p_stratify_column}")
    
    # 실행
    print("\n" + "-" * 80)
    print("코드 실행")
    print("-" * 80)
    train_data, test_data = split_data(
        df,
        p_train_size,
        p_random_state,
        p_shuffle,
        p_stratify,
        p_stratify_column
    )
    
    print("\n" + "-" * 80)
    print("결과")
    print("-" * 80)
    print(f"훈련 데이터: {len(train_data)} 행")
    print(f"테스트 데이터: {len(test_data)} 행")
    print(f"\n훈련 데이터 샘플:")
    print(train_data.head())
    print(f"\n테스트 데이터 샘플:")
    print(test_data.head())
    
    # data_analysis_modules와 비교
    print("\n" + "=" * 80)
    print("data_analysis_modules.py와 비교")
    print("=" * 80)
    
    try:
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from data_analysis_modules import split_data as module_split_data
        
        train_module, test_module = module_split_data(
            df,
            train_size=p_train_size,
            random_state=p_random_state,
            shuffle=p_shuffle,
            stratify=p_stratify,
            stratify_column=p_stratify_column
        )
        
        print(f"\n모듈 결과:")
        print(f"훈련 데이터: {len(train_module)} 행")
        print(f"테스트 데이터: {len(test_module)} 행")
        
        # 첫 몇 개 행 비교
        print(f"\n첫 5개 행의 인덱스 비교:")
        print(f"Code 실행 - 훈련 데이터 인덱스: {train_data.index[:5].tolist()}")
        print(f"모듈 실행 - 훈련 데이터 인덱스: {train_module.index[:5].tolist()}")
        
        if train_data.index.equals(train_module.index) and test_data.index.equals(test_module.index):
            print("\n[OK] 완전히 동일한 결과입니다!")
        else:
            print("\n[WARNING] 인덱스가 다릅니다. 하지만 데이터는 동일할 수 있습니다.")
            # 데이터 값 비교
            train_data_sorted = train_data.sort_index()
            train_module_sorted = train_module.sort_index()
            if train_data_sorted.equals(train_module_sorted):
                print("[OK] 데이터 값은 동일합니다 (인덱스만 다름)")
            else:
                print("[WARNING] 데이터 값도 다릅니다.")
                
    except ImportError as e:
        print(f"\n모듈 import 실패: {e}")
        print("data_analysis_modules.py와 직접 비교할 수 없습니다.")

if __name__ == "__main__":
    main()



















































