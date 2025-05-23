import pandas as pd
import os

def explore_csv(file_path):
    """CSV 파일의 기본 정보를 출력"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(file_path, encoding='cp949')
    
    print(f"\n=== {os.path.basename(file_path)} ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"First 3 rows:\n{df.head(3)}")
    return df

# 15~18번 모듈 파일들 탐색
files = [
    "15_예비건조기.csv",
    "16_호이스트.csv", 
    "17_6호기.csv",
    "18_우측분전반2.csv"
]

for file in files:
    file_path = f"data/raw/{file}"
    if os.path.exists(file_path):
        explore_csv(file_path)
    else:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        