# scripts/module15_outlier_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def analyze_module15_powerfactor():
    """모듈 15 역률 이상치 상세 분석"""
    
    print("🔍 모듈 15 powerFactorR 이상치 분석 시작...")
    
    # 데이터 로드
    file_path = "data/raw/15_예비건조기.csv"
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    df['localtime'] = pd.to_datetime(df['localtime'])
    
    print(f"전체 데이터 개수: {len(df):,}개")
    
    # 역률 통계
    pf_stats = {
        '평균': df['powerFactorR'].mean(),
        '표준편차': df['powerFactorR'].std(), 
        '최소값': df['powerFactorR'].min(),
        '최대값': df['powerFactorR'].max(),
        '25%': df['powerFactorR'].quantile(0.25),
        '50%': df['powerFactorR'].quantile(0.50),
        '75%': df['powerFactorR'].quantile(0.75)
    }
    
    print("\n📊 모듈 15 powerFactorR 기본 통계:")
    for key, value in pf_stats.items():
        print(f"{key:8}: {value:.2f}")
    
    # 이상치 탐지 (여러 기준)
    print("\n🚨 이상치 탐지 결과:")
    
    # 1. 임계값 기준 (70% 미만)
    low_pf_70 = df[df['powerFactorR'] < 70]
    print(f"역률 70% 미만: {len(low_pf_70):,}개 ({len(low_pf_70)/len(df)*100:.2f}%)")
    
    # 2. 임계값 기준 (80% 미만) 
    low_pf_80 = df[df['powerFactorR'] < 80]
    print(f"역률 80% 미만: {len(low_pf_80):,}개 ({len(low_pf_80)/len(df)*100:.2f}%)")
    
    # 3. IQR 기준 이상치
    Q1 = df['powerFactorR'].quantile(0.25)
    Q3 = df['powerFactorR'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    iqr_outliers = df[(df['powerFactorR'] < lower_bound) | (df['powerFactorR'] > upper_bound)]
    print(f"IQR 기준 이상치: {len(iqr_outliers):,}개 ({len(iqr_outliers)/len(df)*100:.2f}%)")
    print(f"IQR 범위: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    # 4. Z-score 기준 (|z| > 2)
    z_scores = np.abs((df['powerFactorR'] - df['powerFactorR'].mean()) / df['powerFactorR'].std())
    z_outliers = df[z_scores > 2]
    print(f"Z-score 기준 이상치 (|z|>2): {len(z_outliers):,}개 ({len(z_outliers)/len(df)*100:.2f}%)")
    
    # 이상치 상세 분석
    print("\n📋 역률 70% 미만 구간 상세:")
    if len(low_pf_70) > 0:
        print(f"최저값: {low_pf_70['powerFactorR'].min():.2f}%")
        print(f"평균값: {low_pf_70['powerFactorR'].mean():.2f}%")
        print(f"첫 발생: {low_pf_70['localtime'].min()}")
        print(f"마지막 발생: {low_pf_70['localtime'].max()}")
        
        # 이상치 구간의 다른 변수들도 확인
        print(f"\n이상치 구간의 다른 변수 평균:")
        print(f"전압R: {low_pf_70['voltageR'].mean():.2f}V")
        print(f"전류R: {low_pf_70['currentR'].mean():.2f}A") 
        print(f"유효전력: {low_pf_70['activePower'].mean():.2f}W")
    
    return df, low_pf_70

def plot_powerfactor_detail(df, sample_size=20000):
    """역률 상세 시각화"""
    
    # 샘플링
    step = len(df) // sample_size
    if step > 1:
        df_sample = df.iloc[::step, :].copy()
    else:
        df_sample = df.copy()
    
    print(f"시각화용 샘플: {len(df_sample):,}개")
    
    # 1. 전체 시계열 + 이상치 강조
    fig, axes = plt.subplots(3, 1, figsize=(20, 15))
    
    # 첫 번째: 전체 역률 시계열
    ax1 = axes[0]
    ax1.plot(df_sample['localtime'], df_sample['powerFactorR'], alpha=0.7, linewidth=0.8, color='blue', label='정상 구간')
    
    # 이상치 강조 (70% 미만)
    low_pf_sample = df_sample[df_sample['powerFactorR'] < 70]
    if len(low_pf_sample) > 0:
        ax1.scatter(low_pf_sample['localtime'], low_pf_sample['powerFactorR'], 
                   color='red', s=20, alpha=0.8, label='이상치 (70% 미만)', zorder=5)
    
    ax1.axhline(y=70, color='red', linestyle='--', alpha=0.8, label='70% 임계값')
    ax1.axhline(y=80, color='orange', linestyle='--', alpha=0.8, label='80% 임계값')
    ax1.set_title('모듈15 powerFactorR 전체 시계열 (이상치 강조)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('역률 (%)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 두 번째: 히스토그램
    ax2 = axes[1]
    ax2.hist(df_sample['powerFactorR'], bins=100, alpha=0.7, color='skyblue', density=True)
    ax2.axvline(df_sample['powerFactorR'].mean(), color='green', linestyle='-', linewidth=2, label=f'평균: {df_sample["powerFactorR"].mean():.1f}%')
    ax2.axvline(70, color='red', linestyle='--', linewidth=2, label='70% 임계값')
    ax2.axvline(80, color='orange', linestyle='--', linewidth=2, label='80% 임계값')
    ax2.set_title('모듈15 powerFactorR 분포', fontsize=14, fontweight='bold')
    ax2.set_xlabel('역률 (%)', fontsize=12)
    ax2.set_ylabel('밀도', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 세 번째: 박스플롯
    ax3 = axes[2]
    bp = ax3.boxplot(df_sample['powerFactorR'], patch_artist=True, vert=False)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    ax3.axvline(70, color='red', linestyle='--', linewidth=2, label='70% 임계값')
    ax3.axvline(80, color='orange', linestyle='--', linewidth=2, label='80% 임계값')
    ax3.set_title('모듈15 powerFactorR 박스플롯', fontsize=14, fontweight='bold')
    ax3.set_xlabel('역률 (%)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def find_outlier_periods(df, threshold=70, min_duration_minutes=1):
    """이상치 발생 구간 탐지"""
    
    print(f"\n🔍 역률 {threshold}% 미만 연속 구간 탐지:")
    
    # 이상치 마킹
    df['is_outlier'] = df['powerFactorR'] < threshold
    
    # 연속된 이상치 구간 찾기
    df['outlier_group'] = (df['is_outlier'] != df['is_outlier'].shift()).cumsum()
    
    outlier_periods = []
    for group_id, group in df.groupby('outlier_group'):
        if group['is_outlier'].iloc[0]:  # 이상치 그룹인 경우
            start_time = group['localtime'].min()
            end_time = group['localtime'].max()
            duration = (end_time - start_time).total_seconds() / 60  # 분 단위
            min_pf = group['powerFactorR'].min()
            
            if duration >= min_duration_minutes:
                outlier_periods.append({
                    'start': start_time,
                    'end': end_time, 
                    'duration_min': duration,
                    'min_powerfactor': min_pf,
                    'count': len(group)
                })
    
    print(f"발견된 연속 이상치 구간: {len(outlier_periods)}개")
    
    # 상위 10개 구간 출력
    outlier_periods.sort(key=lambda x: x['duration_min'], reverse=True)
    
    print("\n📋 주요 이상치 구간 (지속시간 기준):")
    for i, period in enumerate(outlier_periods[:10]):
        print(f"{i+1:2}. {period['start']} ~ {period['end']} "
              f"({period['duration_min']:.1f}분, 최저: {period['min_powerfactor']:.1f}%)")
    
    return outlier_periods

if __name__ == "__main__":
    # 모듈 15 상세 분석
    df, low_pf_data = analyze_module15_powerfactor()
    
    # 시각화
    plot_powerfactor_detail(df)
    
    # 이상치 구간 탐지
    outlier_periods = find_outlier_periods(df, threshold=70)
    
    print("\n✅ 모듈 15 powerFactorR 이상치 분석 완료!")
    print(f"💡 다음 단계: 이상치 구간의 다른 변수들(전압, 전류, 전력) 동시 분석 권장")