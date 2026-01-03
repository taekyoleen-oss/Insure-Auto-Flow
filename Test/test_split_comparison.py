"""
SplitData 모듈의 Python 실행 결과 확인
sklearn.train_test_split을 사용하여 정확한 결과를 확인합니다.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import os

def test_split_data():
    """SplitData 테스트"""
    print("=" * 80)
    print("SplitData 모듈 테스트 (Python sklearn.train_test_split)")
    print("=" * 80)
    
    # 데이터 로드
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'compactiv.csv')
    
    print(f"\n1. 데이터 로드")
    print(f"   파일: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Shape: {df.shape}")
    print(f"   컬럼 수: {len(df.columns)}")
    print(f"   행 수: {len(df)}")
    
    # 파라미터 설정 (앱과 동일)
    train_size = 0.7
    random_state = 42
    shuffle = True
    
    print(f"\n2. SplitData 파라미터")
    print(f"   train_size: {train_size}")
    print(f"   random_state: {random_state}")
    print(f"   shuffle: {shuffle}")
    
    # sklearn의 train_test_split 사용 (Python 코드와 동일)
    train_df, test_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=None
    )
    
    print(f"\n3. 분할 결과")
    print(f"   훈련 데이터: {len(train_df)} 행 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   테스트 데이터: {len(test_df)} 행 ({len(test_df)/len(df)*100:.1f}%)")
    print(f"   총 행 수: {len(train_df) + len(test_df)}")
    print(f"   원본 행 수: {len(df)}")
    
    # 검증
    assert len(train_df) + len(test_df) == len(df), "행 수가 일치하지 않습니다!"
    print(f"   [OK] 행 수 검증 통과")
    
    # 인덱스 정보 (셔플 결과 확인)
    print(f"\n4. 인덱스 정보 (셔플 결과)")
    print(f"   훈련 데이터 첫 10개 인덱스: {train_df.index[:10].tolist()}")
    print(f"   테스트 데이터 첫 10개 인덱스: {test_df.index[:10].tolist()}")
    
    # 첫 번째 행의 데이터 확인
    print(f"\n5. 첫 번째 행 데이터")
    print(f"   훈련 데이터 첫 행 (원본 인덱스 {train_df.index[0]}):")
    first_train_row = train_df.iloc[0]
    print(f"     {first_train_row.name}: {dict(first_train_row)}")
    
    print(f"   테스트 데이터 첫 행 (원본 인덱스 {test_df.index[0]}):")
    first_test_row = test_df.iloc[0]
    print(f"     {first_test_row.name}: {dict(first_test_row)}")
    
    # 원본 데이터의 첫 행
    print(f"   원본 데이터 첫 행 (인덱스 0):")
    print(f"     0: {dict(df.iloc[0])}")
    
    # 결과를 JSON으로 저장
    result = {
        'total_rows': len(df),
        'train_rows': len(train_df),
        'test_rows': len(test_df),
        'train_size': train_size,
        'random_state': random_state,
        'shuffle': shuffle,
        'train_indices': train_df.index[:100].tolist(),  # 처음 100개만
        'test_indices': test_df.index[:100].tolist(),   # 처음 100개만
        'train_first_5_indices': train_df.index[:5].tolist(),
        'test_first_5_indices': test_df.index[:5].tolist(),
        'train_first_row_index': int(train_df.index[0]),
        'test_first_row_index': int(test_df.index[0]),
        'train_first_row_data': {str(k): float(v) if isinstance(v, (int, float, np.integer, np.floating)) else str(v) 
                                  for k, v in train_df.iloc[0].to_dict().items()},
        'test_first_row_data': {str(k): float(v) if isinstance(v, (int, float, np.integer, np.floating)) else str(v) 
                                 for k, v in test_df.iloc[0].to_dict().items()},
    }
    
    output_path = os.path.join(script_dir, 'python_split_result.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n6. 결과 저장")
    print(f"   파일: {output_path}")
    
    # 요약 출력
    print(f"\n" + "=" * 80)
    print("요약")
    print("=" * 80)
    print(f"훈련 데이터: {len(train_df)} 행")
    print(f"테스트 데이터: {len(test_df)} 행")
    print(f"훈련 데이터 첫 인덱스: {train_df.index[0]}")
    print(f"테스트 데이터 첫 인덱스: {test_df.index[0]}")
    
    return train_df, test_df, result

if __name__ == "__main__":
    train_df, test_df, result = test_split_data()
    print(f"\n테스트 완료!")





























































