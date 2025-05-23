# utils/preprocessing_utils.py
import pandas as pd
import numpy as np
import os
from datetime import datetime

def load_csv_safely(file_path, encoding='utf-8-sig'):
    """안전하게 CSV 파일 로드"""
    try:
        return pd.read_csv(file_path, encoding=encoding)
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='cp949')

def clean_column_names(df):
    """컬럼명 정리 및 표준화"""
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('(', '_').str.replace(')', '_')
    return df

def convert_datetime(df, datetime_col='localtime'):
    """날짜 시간 컬럼을 datetime 타입으로 변환"""
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    return df

def add_time_features(df, datetime_col='localtime'):
    """시간 관련 파생 변수 추가"""
    df['year'] = df[datetime_col].dt.year
    df['month'] = df[datetime_col].dt.month
    df['day'] = df[datetime_col].dt.day
    df['hour'] = df[datetime_col].dt.hour
    df['minute'] = df[datetime_col].dt.minute
    df['dayofweek'] = df[datetime_col].dt.dayofweek  # 0=월요일, 6=일요일
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    return df

def remove_outliers_iqr(df, columns, multiplier=1.5):
    """IQR 방법으로 이상치 제거"""
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            # 이상치 개수 출력
            outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
            print(f"{col}: {len(outliers)} outliers removed")
            
            # 이상치 제거
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

def calculate_power_features(df):
    """전력 관련 파생 변수 계산"""
    # 평균 전압
    df['voltage_avg'] = (df['voltageR'] + df['voltageS'] + df['voltageT']) / 3
    
    # 평균 전류
    df['current_avg'] = (df['currentR'] + df['currentS'] + df['currentT']) / 3
    
    # 평균 역률
    df['powerFactor_avg'] = (df['powerFactorR'] + df['powerFactorS'] + df['powerFactorT']) / 3
    
    # 전력 밀도 (전력/전류)
    df['power_density'] = df['activePower'] / (df['current_avg'] + 1e-6)  # 0으로 나누기 방지
    
    # 전압 불균형 (최대값 - 최소값)
    voltage_cols = ['voltageR', 'voltageS', 'voltageT']
    df['voltage_imbalance'] = df[voltage_cols].max(axis=1) - df[voltage_cols].min(axis=1)
    
    # 전류 불균형
    current_cols = ['currentR', 'currentS', 'currentT']
    df['current_imbalance'] = df[current_cols].max(axis=1) - df[current_cols].min(axis=1)
    
    return df

def resample_data(df, freq='1T', datetime_col='localtime'):
    """데이터 리샘플링 (기본: 1분 단위)"""
    df_resampled = df.set_index(datetime_col).resample(freq).agg({
        'voltageR': 'mean',
        'voltageS': 'mean', 
        'voltageT': 'mean',
        'voltageRS': 'mean',
        'voltageST': 'mean',
        'voltageTR': 'mean',
        'currentR': 'mean',
        'currentS': 'mean',
        'currentT': 'mean',
        'activePower': 'mean',
        'powerFactorR': 'mean',
        'powerFactorS': 'mean',
        'powerFactorT': 'mean',
        'reactivePowerLagging': 'mean',
        'accumActiveEnergy': 'last'  # 누적값은 마지막 값 사용
    }).reset_index()
    
    return df_resampled

def save_processed_data(df, file_name, output_dir='data/processed'):
    """전처리된 데이터 저장"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, file_name)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ Saved: {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    
    return output_path

def get_data_summary(df):
    """데이터 요약 통계"""
    print("=== 데이터 요약 ===")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.1f} MB")
    print(f"Date range: {df['localtime'].min()} ~ {df['localtime'].max()}")
    print(f"Unique modules: {df['module_equipment_'].nunique()}")
    print("\n=== 수치형 데이터 통계 ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numeric_cols].describe())