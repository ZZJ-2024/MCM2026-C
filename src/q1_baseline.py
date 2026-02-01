import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt

# 导入通用工具
try:
    from src.utils import DataPreprocessor, VotingInverter, save_results
except ImportError:
    # 备用：如果直接运行此脚本
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    from src.utils import DataPreprocessor, VotingInverter, save_results

# 绘图函数 (局部定义，保证独立性)
def plot_uncertainty(df, output_folder, model_name):
    if df.empty: return
    print(f"🎨 [Drawing] 正在为 {model_name} 绘制不确定性分析图...")
    
    def get_system(season):
        if season <= 2 or season >= 28: return 'Rank System (Discrete)'
        else: return 'Percent System (Continuous)'
    df['System'] = df['Season'].apply(get_system)
    
    sns.set(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 6))
    palette = {'Rank System (Discrete)': '#FF9F40', 'Percent System (Continuous)': '#36A2EB'}
    
    try:
        ax = sns.violinplot(x='System', y='Certainty', data=df, palette=palette, inner=None, alpha=0.3)
        plt.setp(ax.collections, alpha=0.3)
        sns.boxplot(x='System', y='Certainty', data=df, palette=palette, width=0.15, boxprops={'zorder': 2}, fliersize=2)
    except: return

    plt.title(f'Uncertainty Quantification: {model_name}', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Certainty Score', fontsize=12)
    plt.ylim(-0.1, 1.1)
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_path, 'output', output_folder)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'{model_name}_Uncertainty_Analysis.svg'), format='svg', bbox_inches='tight')
    plt.close()

# === 核心函数 ===
def run_baseline():
    print("🚀 [Refining Q1] 开始运行基准模型 (Data-Rich Version)...")
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'data', '2026_MCM_Problem_C_Data.csv')
    
    if not os.path.exists(data_path):
        print(f"❌ 错误: 找不到文件 {data_path}")
        return

    processor = DataPreprocessor(data_path)
    all_results = []
    seasons = processor.get_seasons()
    consistency_stats = {'total_attempts': 0, 'success_count': 0}

    for season in seasons:
        for week in range(1, 16):
            weekly_df = processor.get_weekly_data(season, week)
            if weekly_df.empty or 'Eliminated' not in weekly_df['Result'].values: continue
            
            consistency_stats['total_attempts'] += 1
            try:
                # Baseline: Trust Factor = 0.0
                solver = VotingInverter(weekly_df, season, trust_factor=0.0)
                res = solver.solve()
                
                if res['success']:
                    consistency_stats['success_count'] += 1
                    for i, name in enumerate(solver.contestants):
                        original_row = weekly_df[weekly_df['Contestant'] == name].iloc[0]
                        all_results.append({
                            'Season': season, 'Week': week, 'Contestant': name,
                            'Est_Fan_Votes': res['votes'][i],
                            'Certainty': res['certainty'][i] if 'certainty' in res else 0.5,
                            'Judge Score': original_row['Judge Score'],
                            'Official Result': original_row['Result'],
                            'Model': 'Baseline'
                        })
            except: pass

    success_rate = (consistency_stats['success_count'] / consistency_stats['total_attempts']) * 100 if consistency_stats['total_attempts'] > 0 else 0
    print(f"📊 Baseline 一致性: {success_rate:.2f}%")
    
    save_results(all_results, 'output_q1', 'baseline_predictions.csv')
    if all_results:
        plot_uncertainty(pd.DataFrame(all_results), 'output_q1', 'Baseline_Model')

if __name__ == "__main__":
    run_baseline()