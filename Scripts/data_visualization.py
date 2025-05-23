# scripts/data_visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False

def load_sample_data(file_path, sample_size=20000):  # 샘플 크기도 증가
    """메모리 효율을 위해 샘플 데이터만 로드"""
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    # 시간순으로 일정한 간격으로 샘플링
    step = len(df) // sample_size
    if step > 1:
        df_sample = df.iloc[::step, :].copy()
    else:
        df_sample = df.copy()
    
    # 시간 컬럼 변환
    df_sample['localtime'] = pd.to_datetime(df_sample['localtime'])
    
    print(f"원본 데이터: {len(df):,}행 → 샘플 데이터: {len(df_sample):,}행")
    return df_sample

def plot_time_series(df, columns, title_prefix="", save_path=None):
    """시계열 그래프 그리기"""
    n_cols = len(columns)
    n_rows = (n_cols + 1) // 2  # 2열씩 배치로 변경 (더 넓게)
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(24, 8*n_rows))  # 폭을 더 넓게, 높이도 증가
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(columns):
        row = i // 2  # 2열 배치로 변경
        col_idx = i % 2
        
        ax = axes[row, col_idx]
        ax.plot(df['localtime'], df[col], alpha=0.8, linewidth=1.2, color='steelblue')  # 선 두께 증가
        ax.set_title(f'{title_prefix}{col}', fontsize=14, fontweight='bold')  # 제목 크기 증가
        ax.set_xlabel('시간', fontsize=12)
        ax.set_ylabel(col, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # x축 날짜 포맷 설정
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
    
    # 빈 subplot 숨기기
    for i in range(len(columns), n_rows * 2):  # 2열 배치로 변경
        row = i // 2
        col_idx = i % 2
        axes[row, col_idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"그래프 저장: {save_path}")
    
    plt.show()

def plot_distribution(df, columns, title_prefix="", save_path=None):
    """분포 히스토그램 그리기"""
    n_cols = len(columns)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(columns):
        row = i // 3
        col_idx = i % 3
        
        ax = axes[row, col_idx]
        
        # 히스토그램과 KDE
        ax.hist(df[col].dropna(), bins=50, alpha=0.7, density=True, color='skyblue')
        
        # KDE 라인
        try:
            df[col].dropna().plot.kde(ax=ax, color='red', linewidth=2)
        except:
            pass
        
        ax.set_title(f'{title_prefix}{col} 분포')
        ax.set_xlabel(col)
        ax.set_ylabel('밀도')
        ax.grid(True, alpha=0.3)
        
        # 통계 정보 추가
        mean_val = df[col].mean()
        std_val = df[col].std()
        ax.axvline(mean_val, color='green', linestyle='--', alpha=0.8, label=f'평균: {mean_val:.2f}')
        ax.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.8, label=f'+1σ: {mean_val+std_val:.2f}')
        ax.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.8, label=f'-1σ: {mean_val-std_val:.2f}')
        ax.legend(fontsize=8)
    
    # 빈 subplot 숨기기
    for i in range(len(columns), n_rows * 3):
        row = i // 3
        col_idx = i % 3
        axes[row, col_idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"그래프 저장: {save_path}")
    
    plt.show()

def plot_correlation_matrix(df, title_prefix="", save_path=None):
    """상관관계 히트맵"""
    # 수치형 컬럼만 선택
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                fmt='.2f',
                linewidths=0.5,
                cbar_kws={"shrink": .8})
    
    plt.title(f'{title_prefix}변수 간 상관관계')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"상관관계 히트맵 저장: {save_path}")
    
    plt.show()

def plot_boxplot(df, columns, title_prefix="", save_path=None):
    """박스플롯으로 이상치 확인"""
    n_cols = len(columns)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(columns):
        row = i // 3
        col_idx = i % 3
        
        ax = axes[row, col_idx]
        
        # 박스플롯
        bp = ax.boxplot(df[col].dropna(), patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][0].set_alpha(0.7)
        
        ax.set_title(f'{title_prefix}{col} 박스플롯')
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
        
        # 이상치 개수 표시
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        ax.text(0.5, 0.95, f'이상치: {len(outliers)}개 ({len(outliers)/len(df)*100:.1f}%)', 
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # 빈 subplot 숨기기
    for i in range(len(columns), n_rows * 3):
        row = i // 3
        col_idx = i % 3
        axes[row, col_idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"박스플롯 저장: {save_path}")
    
    plt.show()

def compare_all_modules(modules=[15, 16, 17, 18], sample_size=15000):
    """4개 모듈을 동시에 비교하는 그래프"""
    module_names = {
        15: "15_예비건조기.csv",
        16: "16_호이스트.csv", 
        17: "17_6호기.csv",
        18: "18_우측분전반2.csv"
    }
    
    module_labels = {
        15: "모듈15(예비건조기)",
        16: "모듈16(호이스트)", 
        17: "모듈17(6호기)",
        18: "모듈18(우측분전반2)"
    }
    
    colors = ['steelblue', 'darkorange', 'green', 'red']
    
    # 모든 데이터 로드
    all_data = {}
    for module in modules:
        file_path = f"data/raw/{module_names[module]}"
        print(f"📂 {module_labels[module]} 데이터 로드 중...")
        df = load_sample_data(file_path, sample_size)
        all_data[module] = df
    
    # 주요 변수별 비교 그래프
    compare_variables = {
        '전압 (voltageR)': 'voltageR',
        '전류 (currentR)': 'currentR', 
        '유효전력 (activePower)': 'activePower',
        '역률R (powerFactorR)': 'powerFactorR',
        '누적에너지 (accumActiveEnergy)': 'accumActiveEnergy'
    }
    
    for var_name, var_col in compare_variables.items():
        fig, ax = plt.subplots(1, 1, figsize=(20, 8))
        
        for i, module in enumerate(modules):
            df = all_data[module]
            ax.plot(df['localtime'], df[var_col], 
                   alpha=0.7, linewidth=1.2, 
                   color=colors[i], label=module_labels[module])
        
        ax.set_title(f'🔍 4개 모듈 {var_name} 비교', fontsize=16, fontweight='bold')
        ax.set_xlabel('시간', fontsize=12)
        ax.set_ylabel(var_name, fontsize=12)
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        
        plt.tight_layout()
        plt.show()
    
    # 전압 3상 비교 (R, S, T)
    voltage_phases = ['voltageR', 'voltageS', 'voltageT']
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    axes = axes.flatten()
    
    for i, module in enumerate(modules):
        df = all_data[module]
        ax = axes[i]
        
        for j, phase in enumerate(voltage_phases):
            ax.plot(df['localtime'], df[phase], 
                   alpha=0.8, linewidth=1.0, 
                   label=f'{phase}')
        
        ax.set_title(f'{module_labels[module]} - 3상 전압 비교', fontsize=14, fontweight='bold')
        ax.set_xlabel('시간', fontsize=11)
        ax.set_ylabel('전압 (V)', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # 전류 3상 비교 (R, S, T)
    current_phases = ['currentR', 'currentS', 'currentT']
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    axes = axes.flatten()
    
    for i, module in enumerate(modules):
        df = all_data[module]
        ax = axes[i]
        
        for j, phase in enumerate(current_phases):
            ax.plot(df['localtime'], df[phase], 
                   alpha=0.8, linewidth=1.0, 
                   label=f'{phase}')
        
        ax.set_title(f'{module_labels[module]} - 3상 전류 비교', fontsize=14, fontweight='bold')
        ax.set_xlabel('시간', fontsize=11)
        ax.set_ylabel('전류 (A)', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # 통계 요약 출력
    print("\n📊 4개 모듈 통계 요약")
    print("="*80)
    
    summary_vars = ['voltageR', 'currentR', 'activePower', 'powerFactorR']
    
    for var in summary_vars:
        print(f"\n🔍 {var} 통계:")
        print("-" * 50)
        for module in modules:
            df = all_data[module]
            mean_val = df[var].mean()
            std_val = df[var].std()
            min_val = df[var].min()
            max_val = df[var].max()
            print(f"{module_labels[module]:15} | 평균:{mean_val:8.2f} | 표준편차:{std_val:8.2f} | 범위:[{min_val:8.2f}, {max_val:8.2f}]")
    
    return all_data
    """특정 모듈 데이터 전체 분석"""
    module_names = {
        15: "15_예비건조기.csv",
        16: "16_호이스트.csv", 
        17: "17_6호기.csv",
        18: "18_우측분전반2.csv"
    }
    
    if module_num not in module_names:
        print(f"지원하지 않는 모듈 번호: {module_num}")
        return
    
    file_path = f"data/raw/{module_names[module_num]}"
    print(f"\n🔍 모듈 {module_num} 데이터 분석 시작...")
    
    # 데이터 로드
    df = load_sample_data(file_path, sample_size)
    
    # 전력 관련 주요 변수들
    voltage_cols = ['voltageR', 'voltageS', 'voltageT', 'voltageRS', 'voltageST', 'voltageTR']
    current_cols = ['currentR', 'currentS', 'currentT']
    power_cols = ['activePower', 'powerFactorR', 'powerFactorS', 'powerFactorT', 'reactivePowerLagging']
    energy_cols = ['accumActiveEnergy']
    
    # 1. 시계열 그래프
    print("📈 시계열 그래프 생성 중...")
    plot_time_series(df, voltage_cols, f"모듈{module_num} 전압 - ")
    plot_time_series(df, current_cols, f"모듈{module_num} 전류 - ")
    plot_time_series(df, power_cols, f"모듈{module_num} 전력 - ")
    plot_time_series(df, energy_cols, f"모듈{module_num} 에너지 - ")
    
    # 2. 분포 히스토그램
    print("📊 분포 히스토그램 생성 중...")
    plot_distribution(df, voltage_cols, f"모듈{module_num} 전압 - ")
    plot_distribution(df, current_cols, f"모듈{module_num} 전류 - ")
    plot_distribution(df, power_cols, f"모듈{module_num} 전력 - ")
    
    # 3. 박스플롯 (이상치 확인)
    print("📦 박스플롯 생성 중...")
    plot_boxplot(df, voltage_cols + current_cols, f"모듈{module_num} 전압/전류 - ")
    plot_boxplot(df, power_cols, f"모듈{module_num} 전력 - ")
    
    # 4. 상관관계 히트맵
    print("🔥 상관관계 분석 중...")
    plot_correlation_matrix(df, f"모듈{module_num} - ")
    
    print(f"✅ 모듈 {module_num} 분석 완료!")

if __name__ == "__main__":
    # 4개 모듈 동시 비교 그래프
    print("🎯 4개 모듈 동시 비교 분석 시작!")
    print("="*60)
    
    # 모든 모듈 비교
    all_data = compare_all_modules([15, 16, 17, 18], sample_size=15000)
    
    print("\n✅ 4개 모듈 비교 분석 완료!")
    print("\n🔍 이상치 탐지 체크 포인트:")
    print("1. 전압: 정상 범위(380V±10%) 벗어나는 구간")
    print("2. 전류: 급격한 변화나 비정상적 고전류")
    print("3. 전력: 음수값이나 급변 구간") 
    print("4. 역률: 0.8 미만 구간 (전력품질 문제)")
    print("5. 에너지: 누적 정체 구간 (장비 정지)")
    print("\n📋 각 모듈별 특성을 비교하여 이상 패턴을 확인하세요!")