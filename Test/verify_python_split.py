"""
Python의 sklearn.train_test_split 결과를 검증하고
앱과 비교할 수 있는 기준 결과를 생성합니다.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'compactiv.csv')
    
    print("=" * 80)
    print("Python sklearn.train_test_split 검증")
    print("=" * 80)
    
    # 데이터 로드
    df = pd.read_csv(csv_path)
    print(f"\n데이터 로드 완료: {df.shape}")
    
    # 파라미터 설정
    train_size = 0.7
    random_state = 42
    shuffle = True
    
    print(f"\n파라미터:")
    print(f"  train_size: {train_size}")
    print(f"  random_state: {random_state}")
    print(f"  shuffle: {shuffle}")
    
    # 데이터 분할
    train_df, test_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=None
    )
    
    print(f"\n분할 결과:")
    print(f"  훈련 데이터: {len(train_df)} 행 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  테스트 데이터: {len(test_df)} 행 ({len(test_df)/len(df)*100:.1f}%)")
    
    # 결과 저장
    result = {
        'total_rows': len(df),
        'train_rows': len(train_df),
        'test_rows': len(test_df),
        'train_size': train_size,
        'random_state': random_state,
        'shuffle': shuffle,
        'train_indices': train_df.index.tolist(),
        'test_indices': test_df.index.tolist(),
        'train_first_10_indices': train_df.index[:10].tolist(),
        'test_first_10_indices': test_df.index[:10].tolist(),
    }
    
    output_path = os.path.join(script_dir, 'python_split_reference.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n결과 저장: {output_path}")
    print(f"\n훈련 데이터 첫 10개 인덱스: {result['train_first_10_indices']}")
    print(f"테스트 데이터 첫 10개 인덱스: {result['test_first_10_indices']}")
    
    print("\n검증 완료!")

if __name__ == "__main__":
    main()




















