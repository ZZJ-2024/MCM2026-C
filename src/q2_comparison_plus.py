import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# 动态添加路径以确保能导入 utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import save_results

# ==========================================
# 🎨 O-Prize 绘图风格设置
# ==========================================
def set_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    
    font_paths = [r'C:\Windows\Fonts\arial.ttf', r'/Library/Fonts/Arial.ttf', r'/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    prop = None
    for p in font_paths:
        if os.path.exists(p):
            prop = fm.FontProperties(fname=p)
            prop.set_size(12)
            break
    return prop

FONT_PROP = set_style()
# 准备字体参数，供 matplotlib 函数使用
FONT_ARGS = {'fontproperties': FONT_PROP} if FONT_PROP else {}

# ==========================================
# 🧠 核心逻辑：模拟 + 绘图
# ==========================================
def run_q2_improved():
    print("🚀 [Q2 Plus] Running Advanced Simulation (Judge Save + Sensitivity)...")
    
    # 1. 加载数据
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_path, 'output', 'output_q1_plus', 'improved_predictions.csv')
    output_dir = os.path.join(base_path, 'output', 'output_q2_plus')
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    if not os.path.exists(input_path):
        print("❌ Q1 Data not found.")
        return

    df = pd.read_csv(input_path)
    if 'Judge Score' not in df.columns:
        print("❌ 'Judge Score' missing in input data.")
        return

    # ==========================================
    # 🛡️ Part A: Judge Save Simulation
    # ==========================================
    print("   -> Simulating 'Judge Save' Mechanism...")
    save_simulation_results = []

    for (season, week), group in df.groupby(['Season', 'Week']):
        if len(group) < 3: continue 

        # Rank System Calculation
        j_rank = group['Judge Score'].rank(ascending=False)
        f_rank = group['Est_Fan_Votes'].rank(ascending=False)
        total_rank = j_rank + f_rank
        
        # Original Elimination (Highest Rank Number)
        eliminated_standard = group.loc[total_rank.idxmax()]['Contestant']

        # Judge Save Logic (Bottom 2)
        bottom_2_indices = total_rank.nlargest(2).index
        bottom_2_contestants = group.loc[bottom_2_indices]
        
        # The one with lower Judge Score is eliminated
        loser_idx = bottom_2_contestants['Judge Score'].idxmin()
        eliminated_with_save = group.loc[loser_idx]['Contestant']

        is_reversal = (eliminated_standard != eliminated_with_save)
        
        save_simulation_results.append({
            'Season': season,
            'Week': week,
            'Standard_Eliminated': eliminated_standard,
            'Save_System_Eliminated': eliminated_with_save,
            'Is_Reversal': is_reversal
        })

    # Save CSV
    save_results(save_simulation_results, 'output_q2_plus', 'q2_judge_save_simulation.csv')
    
    # Calculate Rate
    reversal_count = sum(1 for r in save_simulation_results if r['Is_Reversal'])
    total_count = len(save_simulation_results)
    reversal_rate = (reversal_count / total_count * 100) if total_count > 0 else 0
    print(f"      📊 Reversal Rate: {reversal_rate:.2f}%")


    # ==========================================
    # 📡 Part B: Sensitivity Analysis
    # ==========================================
    print("   -> Analyzing Fan Vote Sensitivity...")
    sensitivity_results = []

    for (season, week), group in df.groupby(['Season', 'Week']):
        if len(group) < 3: continue
        
        # Rank System Final Rank
        j_rank = group['Judge Score'].rank(ascending=False)
        f_rank_base = group['Est_Fan_Votes'].rank(ascending=False)
        final_rank_sys = (j_rank + f_rank_base).rank(ascending=True)
        
        # Percent System Final Rank
        j_sum = group['Judge Score'].sum()
        j_pct = group['Judge Score'] / j_sum if j_sum > 0 else 0
        f_pct = group['Est_Fan_Votes']
        final_pct_sys = (j_pct + f_pct).rank(ascending=False)
        
        # Spearman Correlation (Fan Votes vs Final Rank)
        corr_rank = group['Est_Fan_Votes'].corr(final_rank_sys, method='spearman')
        fan_power_rank = abs(corr_rank) if not pd.isna(corr_rank) else 0
        
        corr_pct = group['Est_Fan_Votes'].corr(final_pct_sys, method='spearman')
        fan_power_pct = abs(corr_pct) if not pd.isna(corr_pct) else 0
        
        sensitivity_results.append({
            'Season': season,
            'Week': week,
            'Fan_Influence_Rank': fan_power_rank,
            'Fan_Influence_Percent': fan_power_pct,
            'Difference': fan_power_pct - fan_power_rank
        })
    
    # Save CSV
    save_results(sensitivity_results, 'output_q2_plus', 'q2_sensitivity_analysis.csv')

    # ==========================================
    # 🎨 Part C: Visualization (Integrated)
    # ==========================================
    print("   -> Generating Visualization Plots...")
    try:
        plot_results(save_simulation_results, sensitivity_results, output_dir)
    except Exception as e:
        print(f"❌ Visualization Error: {e}")
        import traceback
        traceback.print_exc()

    print(f"✅ [Success] Q2 Analysis & Plotting Complete! Check: {output_dir}")


def plot_results(save_data, sens_data, output_dir):
    """
    内部绘图函数
    """
    df_save = pd.DataFrame(save_data)
    df_sens = pd.DataFrame(sens_data)

    # ------------------------------------------------------
    # Plot 1: Reversal Rate Donut Chart
    # ------------------------------------------------------
    if not df_save.empty:
        plt.figure(figsize=(7, 7))
        reversal_counts = df_save['Is_Reversal'].value_counts()
        if True not in reversal_counts: reversal_counts[True] = 0
        if False not in reversal_counts: reversal_counts[False] = 0
        
        labels = ['Outcome Changed', 'Same Outcome']
        sizes = [reversal_counts[True], reversal_counts[False]]
        colors = ['#FF6B6B', '#E0E0E0']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, pctdistance=0.85, 
                textprops={'fontsize': 14, **FONT_ARGS}, wedgeprops={'width': 0.3, 'edgecolor': 'w'})
        
        plt.title('Impact of Judge Save Mechanism', fontsize=16, fontweight='bold', **FONT_ARGS)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '1_Reversal_Rate_Donut.png'), dpi=300)
        plt.close()

        # ------------------------------------------------------
        # Plot 4: Reversal Matrix Heatmap
        # ------------------------------------------------------
        plt.figure(figsize=(12, 8))
        heatmap_data = df_save.pivot(index='Season', columns='Week', values='Is_Reversal').fillna(False)
        
        # 🔥🔥🔥 关键修复：强制转换为整数 (0/1)，避免 object 类型报错 🔥🔥🔥
        heatmap_data = heatmap_data.astype(int)
        
        sns.heatmap(heatmap_data, cmap=['#f7f7f7', '#d62728'], cbar=False, linewidths=0.5, linecolor='lightgray')
        
        plt.title('The "Chaos Map": When did Reversals Happen?', fontsize=16, fontweight='bold', **FONT_ARGS)
        plt.ylabel('Season', fontsize=12, **FONT_ARGS)
        plt.xlabel('Week', fontsize=12, **FONT_ARGS)
        
        # Legend (Custom Patches)
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#f7f7f7', edgecolor='gray', label='No Change'),
                           Patch(facecolor='#d62728', edgecolor='gray', label='Reversal Occurred')]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '4_Reversal_Heatmap.png'), dpi=300)
        plt.close()

    # ------------------------------------------------------
    # Plot 2 & 3: Sensitivity Analysis
    # ------------------------------------------------------
    if not df_sens.empty:
        # Plot 2: Boxplot
        plt.figure(figsize=(10, 6))
        plot_data = pd.melt(df_sens, value_vars=['Fan_Influence_Rank', 'Fan_Influence_Percent'], 
                            var_name='System', value_name='Influence')
        plot_data['System'] = plot_data['System'].replace({
            'Fan_Influence_Rank': 'Rank System',
            'Fan_Influence_Percent': 'Percent System'
        })
        
        sns.boxplot(x='System', y='Influence', data=plot_data, width=0.5, palette=['#4D8FAC', '#FF9F40'])
        
        plt.title('Fan Influence Comparison: Rank vs Percent', fontsize=16, fontweight='bold', **FONT_ARGS)
        plt.ylabel('Fan Influence Score (Spearman Correlation)', fontsize=12, **FONT_ARGS)
        plt.xlabel('', fontsize=12, **FONT_ARGS)
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '2_Fan_Power_Boxplot.png'), dpi=300)
        plt.close()

        # Plot 3: Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(df_sens['Difference'], kde=True, color='#2ca02c', bins=20, alpha=0.6)
        plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='No Difference')
        
        plt.title('Sensitivity Bias Distribution (Percent - Rank)', fontsize=16, fontweight='bold', **FONT_ARGS)
        plt.xlabel('Difference in Fan Influence (Positive = Percent System is More Sensitive)', fontsize=12, **FONT_ARGS)
        plt.ylabel('Frequency', fontsize=12, **FONT_ARGS)
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '3_Sensitivity_Dist.png'), dpi=300)
        plt.close()

if __name__ == "__main__":
    run_q2_improved()