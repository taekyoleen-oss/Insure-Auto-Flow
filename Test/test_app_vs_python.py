"""
앱의 JavaScript SplitData와 Python의 sklearn.train_test_split 비교
Python 결과를 기준으로 앱이 동일한 결과를 내도록 검증합니다.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import os

def test_python_split():
    """Python의 sklearn.train_test_split 결과"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'compactiv.csv')
    
    df = pd.read_csv(csv_path)
    train_size = 0.7
    random_state = 42
    shuffle = True
    
    train_df, test_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=None
    )
    
    return {
        'train_indices': train_df.index[:20].tolist(),
        'test_indices': test_df.index[:20].tolist(),
        'train_count': len(train_df),
        'test_count': len(test_df),
    }

if __name__ == "__main__":
    result = test_python_split()
    print("Python 결과:")
    print(f"  훈련 데이터 첫 20개 인덱스: {result['train_indices']}")
    print(f"  테스트 데이터 첫 20개 인덱스: {result['test_indices']}")
    print(f"  훈련 데이터 수: {result['train_count']}")
    print(f"  테스트 데이터 수: {result['test_count']}")

























































