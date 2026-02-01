import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. 动态修复路径 (解决 ImportError) ---
# 获取当前文件所在目录 (src)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (MCM2026-C)
project_root = os.path.dirname(current_dir)
# 将根目录加入系统路径，这样才能找到 src.utils
if project_root not in sys.path:
    sys.path.append(project_root)

# 现在可以安全导入了
try:
    from src.utils import DataPreprocessor, VotingInverter, save_results
except ImportError:
    # 备用方案：如果上面失败，尝试直接从 utils 导入 (防止有些 IDE 环境不同)
    sys.path.append(current_dir)
    from utils import DataPreprocessor, VotingInverter, save_results

# 绘图函数
def plot_uncertainty(df, output_folder, model_name):
    if df.empty: return
    print(f"🎨 [Drawing] 正在为 {model_name} 绘制不确定性分析图...")
    
    def get_system(season):
        if season <= 2 or season >= 28: return 'Rank System (Discrete)'
        else: return 'Percent System (Continuous)'
    df['System'] = df['Season'].apply(get_system)
    
    sns.set(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei'] # 加上 SimHei 防止中文乱码
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
    
    # 保存路径修正
    save_dir = os.path.join(project_root, 'output', output_folder)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'{model_name}_Uncertainty_Analysis.svg'), format='svg', bbox_inches='tight')
    plt.close()

# === 核心函数 ===
def run_improved():
    print("🚀 [Refining Q1+] 开始运行微调模型 (Data-Rich Version)...")
    
    # 路径修正
    data_path = os.path.join(project_root, 'data', '2026_MCM_Problem_C_Data.csv')
    
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
                # Improved: Trust Factor = 0.3
                solver = VotingInverter(weekly_df, season, trust_factor=0.3)
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
                            'Model': 'Improved'
                        })
            except: pass

    success_rate = (consistency_stats['success_count'] / consistency_stats['total_attempts']) * 100 if consistency_stats['total_attempts'] > 0 else 0
    print(f"📊 Improved 一致性: {success_rate:.2f}%")
    
    save_results(all_results, 'output_q1_plus', 'improved_predictions.csv')
    if all_results:
        plot_uncertainty(pd.DataFrame(all_results), 'output_q1_plus', 'Improved_Model')

if __name__ == "__main__":
    run_improved()