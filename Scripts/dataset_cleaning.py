# 70percent_dataset - 70퍼센트 threhold에서 이상치를 제외한 데이터셋 파일
# 70percent_outlier - 70퍼센트 threhold에서 이상치 파일
# 80percent_dataset - 80퍼센트 threhold에서 이상치를 제외한 데이터셋 파일
# 80percent_outlier - 80퍼센트 threhold에서 이상치 파일

import pandas as pd
import numpy as np
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt
from colorama import Fore
sns.set(style="whitegrid")

# 한글 폰트 설정
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'DejaVu Sans'

from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

import os
import warnings
warnings.filterwarnings("ignore")

'''---------------------------------------------------------------------------------------------------------'''
def load_data(file_path, sample_size = 20000):
    df = pd.read_csv(file_path, encoding='utf-8')

    # "Unnamed: 0" 컬럼이 있으면 삭제
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    
    df.set_index('date', inplace=True)
    df.to_csv(file_path, index=False)

    step = len(df) // sample_size
    if step > 0:
        df_sample = df.iloc[::step, :].copy()
    else:
        df_sample = df.copy()

    # 시간 컬럼 변환
    df_sample['date'] = pd.to_datetime(df_sample['date'])

    print(f"원본 데이터: {len(df):,}행 → 샘플 데이터: {len(df_sample):,}행")
    return df_sample
  
'''---------------------------------------------------------------------------------------------------------'''
def resample_data(df, freq='1T', datetime_col='date'):
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

'''---------------------------------------------------------------------------------------------------------'''
# 분석 대상 컬럼
FLOAT_COLUMNS = [
    "voltageR", "voltageS", "voltageT",
    "voltageRS", "voltageST", "voltageTR",
    "currentR", "currentS", "currentT",
    "powerFactorR", "powerFactorS", "powerFactorT",
    "reactivePowerLagging", "accumActiveEnergy",
    "activePower"
]

# 날짜 컬럼 이름
DATE_COLUMN = "date"

summary_vars = [
    'voltageR', 'voltageS', 'voltageT',
    'voltageRS', 'voltageST', 'voltageTR',
    'currentR', 'currentS', 'currentT',
    'powerFactorR', 'powerFactorS', 'powerFactorT',
    'reactivePowerLagging', 'accumActiveEnergy',
    'activePower'
]

'''---------------------------------------------------------------------------------------------------------'''
# 시각화 함수 정의
def visualize_time_series(df, columns=FLOAT_COLUMNS, date_col=DATE_COLUMN):
    """시간에 따른 시계열 데이터 시각화"""
    for col in columns:
        plt.figure(figsize=(20, 8))
        sns.lineplot(data=df, x=date_col, y=col)
        plt.title(f"{col} Time Series")
        plt.xlabel("Date")
        plt.ylabel(col)
        plt.xticks(rotation=45)
        plt.grid()
        plt.show()

'''---------------------------------------------------------------------------------------------------------'''
all_stats = {var: [] for var in summary_vars}

DATA_DIR = "../half_dataset/"
for file in os.listdir(DATA_DIR):
    file_name = os.path.join(DATA_DIR, file)
    df = load_data(file_name)
    print(f"📂 {file} 데이터 로드 중...")
    df = resample_data(df, freq='1T', datetime_col='date')
    print(f"📊 {file} 모듈 통계 요약")
    print("="*60)
    
    for var in summary_vars:
        print(f"\n🔍 {var} 통계:")
        mean_val = df[var].mean()
        std_val = df[var].std()
        min_val = df[var].min()
        max_val = df[var].max()
        print(f"{file} | 평균:{mean_val:8.2f} | 표준편차:{std_val:8.2f} | 범위:[{min_val:8.2f}, {max_val:8.2f}]")
        
        # 저장
        all_stats[var].append({
            'file': file,
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val
        })
    
    print("="*60)
    visualize_time_series(df, FLOAT_COLUMNS, DATE_COLUMN)
    print("="*60)

# 최종 통계 비교 결과 출력
print("\n📋 전체 모델별 통계 비교 요약")
print("="*60)
for var in summary_vars:
    print(f"\n🔧 {var} 비교")
    for stat in all_stats[var]:
        print(f"{stat['file']:<25} | 평균:{stat['mean']:8.2f} | 표준편차:{stat['std']:8.2f} | 범위:[{stat['min']:8.2f}, {stat['max']:8.2f}]")
print("="*70)

'''---------------------------------------------------------------------------------------------------------'''
def normalize_outlier_results(outlier_results):
    """모든 이상치 시간을 datetime 객체로 변환"""
    normalized = {}
    for col, time_list in outlier_results.items():
        normalized[col] = pd.to_datetime(time_list)
    return normalized

def detect_outlier_fixed_threshold(df, column, threshold=0.7):
    """고정 임계값을 사용한 이상치 탐지"""
    outliers = df[df[column] < df[column].max()*threshold]
    return outliers, outliers.index.tolist()

def detect_outlier_iqr(df, column):
    """IQR(Interquartile Range) 방법을 사용한 이상치 탐지"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return outliers, outliers.index.tolist()

def detect_outlier_zscore(df, column, threshold=3):
    """Z-점수 방법을 사용한 이상치 탐지"""
    mean = df[column].mean()
    std = df[column].std()
    z_scores = (df[column] - mean) / std
    outliers = df[np.abs(z_scores) > threshold]
    return outliers, outliers.index.tolist()

# 이상치 시각화 및 구간 출력 함수
def plot_outlier(df, column, outlier_indices, title=None):
    plt.figure(figsize=(14, 5))
    plt.plot(df.index, df[column], label='Data', color='blue')
    plt.scatter(df.index[outlier_indices], df[column].iloc[outlier_indices], color='red', label='Outliers', zorder=5)
    plt.title(title or f"Outlier Detection for '{column}'")
    plt.xlabel("Time")
    plt.ylabel(column)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 이상치 연속 구간 추출
def group_outlier_intervals(indices):
    """연속된 이상치 인덱스를 구간으로 묶음 ([(start1, end1), (start2, end2), ...])"""
    if not indices:
        return []
    
    intervals = []
    start = prev = indices[0]

    for i in indices[1:]:
        if i == prev + 1:
            prev = i
        else:
            intervals.append((start, prev))
            start = prev = i
    intervals.append((start, prev))
    return intervals

# 통합 실행 
def analyze_and_plot_outliers(df, column, method='iqr', threshold=0.7, z_threshold=3):
    # 이상치 탐지
    if method == 'fixed':
        outliers, outlier_indices = detect_outlier_fixed_threshold(df, column, threshold)
    elif method == 'iqr':
        outliers, outlier_indices = detect_outlier_iqr(df, column)
    elif method == 'zscore':
        outliers, outlier_indices = detect_outlier_zscore(df, column, z_threshold)
    else:
        raise ValueError("method must be 'fixed', 'iqr', or 'zscore'")
    
    print(f"\n🔍 '{column}' 이상치 {len(outlier_indices)}건 발견")

    if not outliers.empty:
        print("⏰ 이상치 발생 시점:")
        print(outliers[[column]])

        # 연속 이상치 구간 계산
        intervals = group_outlier_intervals(outliers.index.to_list())
        print("\n🧭 이상치 연속 구간 (시작~끝):")
        for start, end in intervals:
            print(f" - {df.index[start]} ~ {df.index[end]}")
    else:
        intervals = []

    # 시각화
    plot_outliers(df, column, outliers.index.to_list(), title=f"{column} 이상치 탐지 ({method})")

    return outliers, intervals
'''---------------------------------------------------------------------------------------------------------'''
# 이상치 감지 모듈 상세 분석 #모듈13
def analyze_outlier(df, 
                    columns = ['voltageR', 'voltageS', 'voltageT','voltageRS', 'voltageST', 'voltageTR'],
                    method = 'fixed', # fixed, iqr, zscore
                    threshold = 0.7,
                    z_threshold =3
):
    all_results = {}

    for col in columns:
        print(f"\n📊 분석 대상: {col} (method: {method})")

        # 이상치 탐지
        if method == 'fixed':
            outliers, outlier_indices = detect_outlier_fixed_threshold(df, col, threshold)
        elif method == 'iqr':
            outliers, outlier_indices = detect_outlier_iqr(df, col)
        elif method == 'zscore':
            outliers, outlier_indices = detect_outlier_zscore(df, col, z_threshold)
        else:
            raise ValueError("method must be 'fixed', 'iqr', or 'zscore'")

        print(f"🔍 이상치 {len(outlier_indices)}건 발견")

        if not outliers.empty:
            print("⏰ 이상치 발생 시점:")
            print(outliers[[col]])

            # 연속 구간 추출
            intervals = group_outlier_intervals(outliers.index.to_list())
            print("🧭 이상치 연속 구간 (시작~끝):")
            for start, end in intervals:
                print(f" - {df.index[start]} ~ {df.index[end]}")
        else:
            intervals = []

        # 시각화
        plot_outliers(df, col, outliers.index.to_list(), title=f"{col} 이상치 탐지 ({method})")

        # 결과 저장
        all_results[col] = {
            "outliers": outliers,
            "intervals": intervals
        }

    return all_results

'''---------------------------------------------------------------------------------------------------------'''
# 이상치 감지 모듈 상세 분석 #모듈 15,17
def analyze_outlier(df, 
                    columns = ['powerFactorR', 'powerFactorS', 'powerFactorT'],
                    method = 'fixed', # fixed, iqr, zscore
                    threshold = 0.7,
                    z_threshold =3
):
    all_results = {}

    for col in columns:
        print(f"\n📊 분석 대상: {col} (method: {method})")

        # 이상치 탐지
        if method == 'fixed':
            outliers, outlier_indices = detect_outlier_fixed_threshold(df, col, threshold)
        elif method == 'iqr':
            outliers, outlier_indices = detect_outlier_iqr(df, col)
        elif method == 'zscore':
            outliers, outlier_indices = detect_outlier_zscore(df, col, z_threshold)
        else:
            raise ValueError("method must be 'fixed', 'iqr', or 'zscore'")

        print(f"🔍 이상치 {len(outlier_indices)}건 발견")

        if not outliers.empty:
            print("⏰ 이상치 발생 시점:")
            print(outliers[[col]])

            # 연속 구간 추출
            intervals = group_outlier_intervals(outliers.index.to_list())
            print("🧭 이상치 연속 구간 (시작~끝):")
            for start, end in intervals:
                print(f" - {df.index[start]} ~ {df.index[end]}")
        else:
            intervals = []

        # 시각화
        plot_outliers(df, col, outliers.index.to_list(), title=f"{col} 이상치 탐지 ({method})")

        # 결과 저장
        all_results[col] = {
            "outliers": outliers,
            "intervals": intervals
        }

    return all_results
  
'''---------------------------------------------------------------------------------------------------------'''
# 임계값 70% 이상치 제거
def remove_outlier_from_df(df, outlier_results):
    """date index 기반으로 이상치 제거"""
    outlier_results = normalize_outlier_results(outlier_results)

    for column, outlier_times in outlier_results.items():
        for outlier_time in outlier_times:
            if outlier_time in df.index:
                df.loc[outlier_time, column] = np.nan
    return df

'''---------------------------------------------------------------------------------------------------------'''
# 임계값 70% 이상치 저장
def save_outlier(outlier_results, file_name, data_dir="../70percent_outlier/"):
    all_outliers = []

    for col, result in outlier_results.items():
        temp_df = result['outliers'].copy()
        temp_df['feature'] = col
        all_outliers.append(temp_df)

    if all_outliers:
        outlier_df = pd.concat(all_outliers).sort_index()
        outlier_df.to_csv(os.path.join(data_dir, file_name), index=False)
        print(f"✅ 이상치 {len(outlier_df)}건 저장 완료: {os.path.join(data_dir, file_name)}")
    else:
        print("⚠️ 이상치 없음. 저장 생략.")

'''---------------------------------------------------------------------------------------------------------'''
# 임계값 70% 이상치 제거 데이터셋 저장
def save_cleaned_data(df, outlier_results, file_name, data_dir="../70percent_dataset/"):
    os.makedirs(data_dir, exist_ok=True)
    cleaned_df = remove_outlier_from_df(df, outlier_results)
    cleaned_df.to_csv(os.path.join(data_dir, file_name), index=False)
    print(f"✅ 이상치 제거 후 데이터 저장 완료: {os.path.join(data_dir, file_name)}")

'''---------------------------------------------------------------------------------------------------------'''
# 모듈 13, 15, 17 이상치 탐지 및 제거
# 모듈 13
df13 = pd.read_csv("../half_dataset/even_13_3호기.csv", encoding='utf-8')
outlier_result13 = analyze_outlier(df13, 
                    columns = ['voltageR', 'voltageS', 'voltageT','voltageRS', 'voltageST', 'voltageTR'],
                    method = 'iqr', # fixed, iqr, zscore
                    threshold = 0.7,
                    z_threshold =3
)

remove_outlier_from_df(df, outlier_result13)
save_outlier(outlier_result13, "even_13_3호기.csv", data_dir="../70percent_outlier/")
save_cleaned_data(df13, outlier_result13, "even_13_3호기.csv", data_dir="../70percent_dataset/")

print(outlier_result13)

'''---------------------------------------------------------------------------------------------------------'''
# 모듈 15
df15 = pd.read_csv("../half_dataset/even_15_예비건조기.csv", encoding='utf-8')
outlier_result15 = analyze_outlier(df15, 
                    columns = ['powerFactorR', 'powerFactorS', 'powerFactorT'],
                    method = 'fixed', # fixed, iqr, zscore
                    threshold = 0.7,
                    z_threshold =3
)
remove_outlier_from_df(df, outlier_result15)
save_outlier(outlier_result15, "even_15_예비건조기.csv", data_dir="../70percent_outlier/")
save_cleaned_data(df15, outlier_result15, "even_15_예비건조기.csv", data_dir="../70percent_dataset/")

# 모듈 17
df17 = pd.read_csv("../half_dataset/even_17_6호기.csv", encoding='utf-8')
outlier_result17 = analyze_outlier(df17, 
                    columns = ['powerFactorR', 'powerFactorS', 'powerFactorT'],
                    method = 'fixed', # fixed, iqr, zscore
                    threshold = 0.7,
                    z_threshold =3
)
remove_outlier_from_df(df, outlier_result17)
save_outlier(outlier_result17, "even_17_6호기.csv", data_dir="../70percent_outlier/")
save_cleaned_data(df13, outlier_result17, "even_17_6호기.csv", data_dir="../70percent_dataset/")

'''---------------------------------------------------------------------------------------------------------'''
# 임계값 80% 이상치 제거
# 80% 임계값 이상치 탐지 (fixed threshold) 함수 예시 (기존 analyze_outlier 변형)
def detect_outliers_fixed_threshold(df, columns, threshold=0.8):
    outlier_results = {}
    for col in columns:
        # 임계값: 예) voltage가 max 값의 80% 이하일 때 이상치라고 가정
        max_val = df[col].max()
        cutoff = max_val * threshold
        outliers = df[df[col] < cutoff][[col]]
        outlier_results[col] = {'outliers': outliers}
    return outlier_results
