# scripts/powerfactor_correlation_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_all_modules():
    """모든 모듈 데이터 로드"""
    module_files = {
        15: "data/raw/15_예비건조기.csv",
        16: "data/raw/16_호이스트.csv", 
        17: "data/raw/17_6호기.csv",
        18: "data/raw/18_우측분전반2.csv"
    }
    
    all_data = {}
    
    for module_num, file_path in module_files.items():
        print(f"📂 모듈 {module_num} 데이터 로드 중...")
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        df['localtime'] = pd.to_datetime(df['localtime'])
        all_data[module_num] = df
        
    return all_data

def analyze_powerfactor_anomaly_correlation(df, threshold=70):
    """역률 이상치 시점의 다른 변수 변화 분석"""
    
    print(f"\n🔍 역률 {threshold}% 미만 이상치 시점 분석...")
    
    # 이상치 마킹
    df['is_pf_anomaly'] = df['powerFactorR'] < threshold
    
    # 정상/이상 구간 통계 비교
    normal_data = df[~df['is_pf_anomaly']]
    anomaly_data = df[df['is_pf_anomaly']]
    
    print(f"정상 데이터: {len(normal_data):,}개 ({len(normal_data)/len(df)*100:.1f}%)")
    print(f"이상 데이터: {len(anomaly_data):,}개 ({len(anomaly_data)/len(df)*100:.1f}%)")
    
    # 주요 변수들 비교
    variables = ['voltageR', 'voltageS', 'voltageT', 'currentR', 'currentS', 'currentT', 
                'activePower', 'powerFactorS', 'powerFactorT', 'reactivePowerLagging']
    
    comparison_stats = {}
    
    print(f"\n📊 정상 vs 이상 구간 변수 비교:")
    print("=" * 80)
    print(f"{'변수명':20} {'정상_평균':>12} {'이상_평균':>12} {'차이':>12} {'차이율':>10}")
    print("-" * 80)
    
    for var in variables:
        if var in df.columns:
            normal_mean = normal_data[var].mean()
            anomaly_mean = anomaly_data[var].mean()
            diff = anomaly_mean - normal_mean
            diff_rate = (diff / normal_mean * 100) if normal_mean != 0 else 0
            
            comparison_stats[var] = {
                'normal_mean': normal_mean,
                'anomaly_mean': anomaly_mean,
                'difference': diff,
                'diff_rate': diff_rate
            }
            
            print(f"{var:20} {normal_mean:12.2f} {anomaly_mean:12.2f} {diff:12.2f} {diff_rate:9.1f}%")
    
    return comparison_stats, anomaly_data

def plot_anomaly_correlation(df, sample_size=5000):
    """이상치 시점의 다변수 시각화"""
    
    # 샘플링
    step = len(df) // sample_size
    if step > 1:
        df_sample = df.iloc[::step, :].copy()
    else:
        df_sample = df.copy()
    
    df_sample['is_pf_anomaly'] = df_sample['powerFactorR'] < 70
    
    print(f"시각화용 샘플: {len(df_sample):,}개")
    
    # 1. 다변수 시계열 (역률 이상치 시점 강조)
    fig, axes = plt.subplots(4, 1, figsize=(24, 20))
    
    # 전압 3상
    ax1 = axes[0]
    ax1.plot(df_sample['localtime'], df_sample['voltageR'], alpha=0.6, label='voltageR', color='blue')
    ax1.plot(df_sample['localtime'], df_sample['voltageS'], alpha=0.6, label='voltageS', color='orange')
    ax1.plot(df_sample['localtime'], df_sample['voltageT'], alpha=0.6, label='voltageT', color='green')
    
    # 이상치 시점 강조
    anomaly_points = df_sample[df_sample['is_pf_anomaly']]
    if len(anomaly_points) > 0:
        ax1.scatter(anomaly_points['localtime'], anomaly_points['voltageR'], 
                   color='red', s=30, alpha=0.8, label='역률이상시점', zorder=5)
    
    ax1.set_title('모듈15 3상 전압 (역률 이상치 시점 강조)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('전압 (V)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 전류 3상
    ax2 = axes[1]
    ax2.plot(df_sample['localtime'], df_sample['currentR'], alpha=0.6, label='currentR', color='blue')
    ax2.plot(df_sample['localtime'], df_sample['currentS'], alpha=0.6, label='currentS', color='orange')
    ax2.plot(df_sample['localtime'], df_sample['currentT'], alpha=0.6, label='currentT', color='green')
    
    if len(anomaly_points) > 0:
        ax2.scatter(anomaly_points['localtime'], anomaly_points['currentR'], 
                   color='red', s=30, alpha=0.8, label='역률이상시점', zorder=5)
    
    ax2.set_title('모듈15 3상 전류 (역률 이상치 시점 강조)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('전류 (A)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 전력 관련
    ax3 = axes[2]
    ax3.plot(df_sample['localtime'], df_sample['activePower'], alpha=0.6, label='유효전력', color='blue')
    ax3.plot(df_sample['localtime'], df_sample['reactivePowerLagging'], alpha=0.6, label='무효전력', color='orange')
    
    if len(anomaly_points) > 0:
        ax3.scatter(anomaly_points['localtime'], anomaly_points['activePower'], 
                   color='red', s=30, alpha=0.8, label='역률이상시점', zorder=5)
    
    ax3.set_title('모듈15 전력 (역률 이상치 시점 강조)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('전력 (W)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 역률 3상
    ax4 = axes[3]
    ax4.plot(df_sample['localtime'], df_sample['powerFactorR'], alpha=0.7, label='powerFactorR', color='blue')
    ax4.plot(df_sample['localtime'], df_sample['powerFactorS'], alpha=0.7, label='powerFactorS', color='orange')
    ax4.plot(df_sample['localtime'], df_sample['powerFactorT'], alpha=0.7, label='powerFactorT', color='green')
    ax4.axhline(y=70, color='red', linestyle='--', alpha=0.8, label='70% 임계값')
    
    if len(anomaly_points) > 0:
        ax4.scatter(anomaly_points['localtime'], anomaly_points['powerFactorR'], 
                   color='red', s=30, alpha=0.8, label='역률이상시점', zorder=5)
    
    ax4.set_title('모듈15 3상 역률 (이상치 강조)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('역률 (%)', fontsize=12)
    ax4.set_xlabel('시간', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_time_pattern(df, threshold=70):
    """이상치 발생 시간대 패턴 분석"""
    
    print(f"\n🕐 역률 이상치 발생 시간대 패턴 분석...")
    
    anomaly_data = df[df['powerFactorR'] < threshold].copy()
    
    if len(anomaly_data) == 0:
        print("이상치 데이터가 없습니다.")
        return
    
    # 시간 관련 변수 추가
    anomaly_data['hour'] = anomaly_data['localtime'].dt.hour
    anomaly_data['dayofweek'] = anomaly_data['localtime'].dt.dayofweek  # 0=월요일
    anomaly_data['date'] = anomaly_data['localtime'].dt.date
    
    # 시간대별 발생 빈도
    hourly_count = anomaly_data['hour'].value_counts().sort_index()
    
    # 요일별 발생 빈도
    daily_count = anomaly_data['dayofweek'].value_counts().sort_index()
    daily_labels = ['월', '화', '수', '목', '금', '토', '일']
    
    # 날짜별 발생 빈도
    date_count = anomaly_data['date'].value_counts().sort_index()
    
    print(f"\n📅 시간대별 이상치 발생 패턴:")
    print("시간대 | 발생횟수 | 비율")
    print("-" * 25)
    for hour in range(24):
        count = hourly_count.get(hour, 0)
        ratio = count / len(anomaly_data) * 100
        print(f"{hour:2d}시   | {count:7d} | {ratio:5.1f}%")
    
    print(f"\n📅 요일별 이상치 발생 패턴:")
    print("요일 | 발생횟수 | 비율")
    print("-" * 20)
    for i, day_name in enumerate(daily_labels):
        count = daily_count.get(i, 0)
        ratio = count / len(anomaly_data) * 100
        print(f"{day_name}   | {count:7d} | {ratio:5.1f}%")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 시간대별 그래프
    ax1 = axes[0]
    bars1 = ax1.bar(hourly_count.index, hourly_count.values, alpha=0.7, color='skyblue')
    ax1.set_title('시간대별 역률 이상치 발생 빈도', fontsize=14, fontweight='bold')
    ax1.set_xlabel('시간 (시)', fontsize=12)
    ax1.set_ylabel('발생 횟수', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 최대값 표시
    max_hour = hourly_count.idxmax()
    max_count = hourly_count.max()
    ax1.annotate(f'최대: {max_hour}시\n({max_count}회)', 
                xy=(max_hour, max_count), xytext=(max_hour+2, max_count+50),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, ha='center')
    
    # 요일별 그래프
    ax2 = axes[1]
    bars2 = ax2.bar(range(7), [daily_count.get(i, 0) for i in range(7)], 
                    alpha=0.7, color='lightcoral')
    ax2.set_title('요일별 역률 이상치 발생 빈도', fontsize=14, fontweight='bold')
    ax2.set_xlabel('요일', fontsize=12)
    ax2.set_ylabel('발생 횟수', fontsize=12)
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(daily_labels)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return hourly_count, daily_count

def compare_modules_during_anomaly(all_data, reference_module=15, threshold=70):
    """모듈 15 이상치 시점에 다른 모듈들의 상태 비교"""
    
    print(f"\n🔄 모듈 {reference_module} 역률 이상치 시점의 다른 모듈 상태 분석...")
    
    ref_df = all_data[reference_module].copy()
    ref_df['is_anomaly'] = ref_df['powerFactorR'] < threshold
    
    # 이상치 시점들
    anomaly_times = ref_df[ref_df['is_anomaly']]['localtime'].values
    print(f"분석 대상 이상치 시점: {len(anomaly_times):,}개")
    
    if len(anomaly_times) == 0:
        return
    
    # 각 모듈별로 해당 시점의 데이터 확인
    results = {}
    
    for module_num, df in all_data.items():
        if module_num == reference_module:
            continue
            
        # 이상치 시점과 가장 가까운 시간의 데이터 찾기
        module_anomaly_data = []
        
        for anomaly_time in anomaly_times[:1000]:  # 샘플링 (성능상)
            # 해당 시점 ±30초 범위에서 데이터 찾기
            time_mask = (df['localtime'] >= pd.to_datetime(anomaly_time) - timedelta(seconds=30)) & \
                       (df['localtime'] <= pd.to_datetime(anomaly_time) + timedelta(seconds=30))
            
            nearby_data = df[time_mask]
            if len(nearby_data) > 0:
                # 가장 가까운 시점의 데이터
                closest_idx = (nearby_data['localtime'] - pd.to_datetime(anomaly_time)).abs().idxmin()
                module_anomaly_data.append(nearby_data.loc[closest_idx])
        
        if module_anomaly_data:
            anomaly_df = pd.DataFrame(module_anomaly_data)
            
            # 해당 모듈의 정상 상태와 비교
            normal_stats = df.describe()
            anomaly_stats = anomaly_df.describe()
            
            results[module_num] = {
                'normal_powerfactor': normal_stats.loc['mean', 'powerFactorR'],
                'anomaly_powerfactor': anomaly_stats.loc['mean', 'powerFactorR'],
                'normal_power': normal_stats.loc['mean', 'activePower'],
                'anomaly_power': anomaly_stats.loc['mean', 'activePower'],
                'normal_current': normal_stats.loc['mean', 'currentR'],
                'anomaly_current': anomaly_stats.loc['mean', 'currentR']
            }
    
    # 결과 출력
    print(f"\n📊 모듈 {reference_module} 이상치 시점의 다른 모듈 상태:")
    print("=" * 80)
    print(f"{'모듈':4} {'정상역률':>10} {'이상시역률':>12} {'정상전력':>12} {'이상시전력':>12} {'전력변화율':>10}")
    print("-" * 80)
    
    for module_num, stats in results.items():
        power_change = ((stats['anomaly_power'] - stats['normal_power']) / stats['normal_power']) * 100
        print(f"{module_num:4} {stats['normal_powerfactor']:10.1f} {stats['anomaly_powerfactor']:12.1f} "
              f"{stats['normal_power']:12.0f} {stats['anomaly_power']:12.0f} {power_change:9.1f}%")
    
    return results

if __name__ == "__main__":
    print("🔍 모듈 15 역률 이상치 다변수 상관관계 분석 시작!")
    print("=" * 60)
    
    # 1. 모든 모듈 데이터 로드
    all_data = load_all_modules()
    df_module15 = all_data[15]
    
    # 2. 역률 이상치 시점의 다른 변수 변화 분석
    comparison_stats, anomaly_data = analyze_powerfactor_anomaly_correlation(df_module15)
    
    # 3. 다변수 시계열 시각화
    print("\n📈 다변수 시계열 그래프 생성 중...")
    plot_anomaly_correlation(df_module15)
    
    # 4. 시간대 패턴 분석
    hourly_pattern, daily_pattern = analyze_time_pattern(df_module15)
    
    # 5. 다른 모듈들과의 상관관계
    module_comparison = compare_modules_during_anomaly(all_data)
    
    print("\n✅ 종합 분석 완료!")
    print("\n🎯 주요 발견사항:")
    print("1. 역률 이상치 시점의 전압/전류/전력 변화 패턴")
    print("2. 이상치 발생 시간대 특성")
    print("3. 다른 모듈들과의 동시 변화 여부")
    print("\n💡 이 결과를 바탕으로 이상치 원인 규명 및 대응방안 수립 가능!")