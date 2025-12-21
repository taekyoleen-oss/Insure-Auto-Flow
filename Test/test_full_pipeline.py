"""
LoadData부터 SplitData까지 전체 파이프라인 테스트
Python의 실제 실행 결과를 확인하여 앱과 비교합니다.
"""

import pandas as pd
import numpy as np
import sys
import os

# data_analysis_modules.py의 함수들을 import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_analysis_modules import load_data, split_data

def test_full_pipeline():
    """전체 파이프라인 테스트"""
    print("=" * 80)
    print("전체 파이프라인 테스트 (LoadData -> SplitData)")
    print("=" * 80)
    
    # 1. LoadData
    print("\n" + "-" * 80)
    print("1. LoadData 모듈")
    print("-" * 80)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'compactiv.csv')
    
    if not os.path.exists(csv_path):
        print(f"오류: 파일을 찾을 수 없습니다: {csv_path}")
        return None
    
    df = load_data(csv_path)
    if df is None:
        print("오류: 데이터 로드 실패")
        return None
    
    print(f"\n로드된 데이터:")
    print(f"  Shape: {df.shape}")
    print(f"  컬럼: {list(df.columns)}")
    print(f"\n처음 5행:")
    print(df.head())
    print(f"\n데이터 타입:")
    print(df.dtypes)
    
    # 2. SplitData
    print("\n" + "-" * 80)
    print("2. SplitData 모듈")
    print("-" * 80)
    train_size = 0.7
    random_state = 42
    shuffle = True
    
    print(f"\n파라미터:")
    print(f"  train_size: {train_size}")
    print(f"  random_state: {random_state}")
    print(f"  shuffle: {shuffle}")
    
    train_df, test_df = split_data(
        df, 
        train_size=train_size, 
        random_state=random_state, 
        shuffle=shuffle
    )
    
    print(f"\n분할 결과:")
    print(f"  훈련 데이터: {len(train_df)} 행 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  테스트 데이터: {len(test_df)} 행 ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  총 행 수: {len(train_df) + len(test_df)}")
    print(f"  원본 행 수: {len(df)}")
    
    # 검증
    assert len(train_df) + len(test_df) == len(df), "행 수가 일치하지 않습니다!"
    
    # 첫 번째 행의 인덱스 확인 (셔플 결과 확인)
    print(f"\n훈련 데이터 첫 5개 인덱스:")
    print(train_df.index[:5].tolist())
    print(f"\n테스트 데이터 첫 5개 인덱스:")
    print(test_df.index[:5].tolist())
    
    # 첫 번째 행의 실제 데이터 확인
    print(f"\n훈련 데이터 첫 행 (인덱스 {train_df.index[0]}):")
    print(train_df.iloc[0])
    print(f"\n테스트 데이터 첫 행 (인덱스 {test_df.index[0]}):")
    print(test_df.iloc[0])
    
    # 원본 데이터의 첫 행과 비교
    print(f"\n원본 데이터 첫 행 (인덱스 0):")
    print(df.iloc[0])
    
    # 결과를 JSON으로 저장하여 JavaScript와 비교 가능하도록
    result = {
        'total_rows': len(df),
        'train_rows': len(train_df),
        'test_rows': len(test_df),
        'train_indices': train_df.index[:100].tolist(),  # 처음 100개만
        'test_indices': test_df.index[:100].tolist(),   # 처음 100개만
        'train_first_row': train_df.iloc[0].to_dict(),
        'test_first_row': test_df.iloc[0].to_dict(),
    }
    
    import json
    output_path = os.path.join(script_dir, 'python_split_result.json')
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n결과가 저장되었습니다: {output_path}")
    
    return train_df, test_df

if __name__ == "__main__":
    test_full_pipeline()



















































