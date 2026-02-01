import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# ==========================================
# 🎨 O-Prize 绘图风格
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
    return prop, {'fontproperties': prop} if prop else {}

FONT_PROP, FONT_ARGS = set_style()

# ==========================================
# 🧠 核心模型: 标准化 50/50 均衡系统
# ==========================================
def run_q4_improved():
    print("🚀 [Q4 Plus] 正在部署标准化 50/50 均衡赛制 (Standardized 50/50)...")
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_path, 'output', 'output_q1_plus', 'improved_predictions.csv')
    output_dir = os.path.join(base_path, 'output', 'output_q4_plus')
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    df = pd.read_csv(input_path)
    
    if 'Judge Score' not in df.columns:
        print("❌ 缺少数据，无法运行。")
        return

    # ----------------------------------------------------
    # 1. 全局应用新赛制 (Z-Score Standardization)
    # ----------------------------------------------------
    results = []
    
    for (season, week), group in df.groupby(['Season', 'Week']):
        group = group.copy()
        
        # 核心修改：权重始终 1:1
        w_judge = 0.5
        
        # Z-Score 标准化：(x - mean) / std
        # 这能让评委分和粉丝票在同一个“波动率”层面上对话
        j_mean, j_std = group['Judge Score'].mean(), group['Judge Score'].std()
        f_mean, f_std = group['Est_Fan_Votes'].mean(), group['Est_Fan_Votes'].std()
        
        # 防御性处理：防止标准差为0
        if j_std == 0: j_std = 1
        if f_std == 0: f_std = 1
        
        group['J_ZScore'] = (group['Judge Score'] - j_mean) / j_std
        group['F_ZScore'] = (group['Est_Fan_Votes'] - f_mean) / f_std
        
        # 计算新分数 (真正公平的 1:1)
        group['New_Score'] = 0.5 * group['J_ZScore'] + 0.5 * group['F_ZScore']
        group['Dynamic_Weight_J'] = 0.5 # 始终记录为 0.5
        
        # 计算新排名 (Z-Score 越大越好，排名越小)
        group['New_Rank'] = group['New_Score'].rank(ascending=False)
        
        # 记录旧排名 (Rank Sum)
        group['Original_Rank_Simulated'] = (group['Judge Score'].rank(ascending=False) + group['Est_Fan_Votes'].rank(ascending=False)).rank(ascending=True)
        
        results.append(group)
        
    final_df = pd.concat(results)
    final_df.to_csv(os.path.join(output_dir, 'q4_equal_weight_simulation.csv'), index=False)
    
    # ----------------------------------------------------
    # 2. 宏观验证: 权力的平行平衡 (Parallel Balance)
    # ----------------------------------------------------
    print("   -> 进行全局平衡性检验...")
    balance_stats = []
    
    for week in sorted(final_df['Week'].unique()):
        week_data = final_df[final_df['Week'] == week]
        if len(week_data) < 20: continue 
        
        corr_judge = week_data['New_Rank'].corr(week_data['Judge Score'], method='spearman')
        corr_fan = week_data['New_Rank'].corr(week_data['Est_Fan_Votes'], method='spearman')
        
        balance_stats.append({
            'Week': week,
            'Influence_Judge': abs(corr_judge),
            'Influence_Fan': abs(corr_fan),
            'Judge_Weight_Setting': 0.5
        })
        
    balance_df = pd.DataFrame(balance_stats)
    balance_df.to_csv(os.path.join(output_dir, 'q4_balance_metrics.csv'), index=False)

    # ----------------------------------------------------
    # 3. 冠军归属成分
    # ----------------------------------------------------
    print("   -> 分析新赛制下的冠军成分...")
    champions = []
    for season, group in final_df.groupby('Season'):
        last_week = group['Week'].max()
        final_week_data = group[group['Week'] == last_week]
        new_winner = final_week_data.loc[final_week_data['New_Rank'].idxmin()]
        
        j_rank = final_week_data['Judge Score'].rank(ascending=False)
        is_judge_no1 = (j_rank[new_winner.name] == 1)
        f_rank = final_week_data['Est_Fan_Votes'].rank(ascending=False)
        is_fan_no1 = (f_rank[new_winner.name] == 1)
        
        champions.append({
            'Season': season,
            'Winner': new_winner['Contestant'],
            'Is_Judge_Pick': is_judge_no1,
            'Is_Fan_Pick': is_fan_no1
        })
    champ_df = pd.DataFrame(champions)

    # ----------------------------------------------------
    # 4. Bobby Bones 案例
    # ----------------------------------------------------
    print("   -> 正在进行 Bobby Bones 压力测试...")
    bobby_df = final_df[(final_df['Season'] == 27) & (final_df['Contestant'] == 'Bobby Bones')].sort_values('Week')

    # ----------------------------------------------------
    # 5. 自动绘图
    # ----------------------------------------------------
    plot_q4_comprehensive(balance_df, champ_df, bobby_df, output_dir)
    print(f"✅ [Success] Q4 均衡赛制优化完成！图表路径: {output_dir}")


def plot_q4_comprehensive(balance_df, champ_df, bobby_df, out_dir):
    """
    绘制 Q4 的三张核心论证图 (50/50 版)
    """
    
    # ==================================================
    # 图 1: 权力的平行线 (The Parallel Balance)
    # ==================================================
    if not balance_df.empty:
        plt.figure(figsize=(10, 6))
        
        # 绘制评委影响力曲线
        plt.plot(balance_df['Week'], balance_df['Influence_Judge'], 
                 marker='o', color='#6C63FF', linewidth=3, label='Judge Influence (Target: 0.5)', alpha=0.9)
        
        # 绘制粉丝影响力曲线
        plt.plot(balance_df['Week'], balance_df['Influence_Fan'], 
                 marker='D', color='#FF6B6B', linewidth=3, label='Fan Influence (Target: 0.5)', alpha=0.9)
        
        plt.title('Perfect Balance: Judge vs Fan Influence (1:1 Ratio)', fontsize=14, fontweight='bold', **FONT_ARGS)
        plt.xlabel('Competition Week', fontsize=12, **FONT_ARGS)
        plt.ylabel('Correlation with Final Outcome', fontsize=12, **FONT_ARGS)
        plt.ylim(0, 1.0)
        plt.legend(loc='center right', frameon=True)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # 标注
        plt.text(6, 0.5, "Balanced Power Structure", ha='center', va='center', 
                 fontsize=14, color='gray', alpha=0.3, rotation=0, fontweight='bold', **FONT_ARGS)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '1_Power_Balance.png'), dpi=300)
        plt.close()

    # ==================================================
    # 图 2: 冠军成分分析
    # ==================================================
    if not champ_df.empty:
        plt.figure(figsize=(8, 6))
        consensus = champ_df[champ_df['Is_Judge_Pick'] & champ_df['Is_Fan_Pick']].shape[0]
        judge_only = champ_df[champ_df['Is_Judge_Pick'] & (~champ_df['Is_Fan_Pick'])].shape[0]
        fan_only = champ_df[(~champ_df['Is_Judge_Pick']) & champ_df['Is_Fan_Pick']].shape[0]
        compromise = champ_df[(~champ_df['Is_Judge_Pick']) & (~champ_df['Is_Fan_Pick'])].shape[0]
        
        counts = [consensus, judge_only, fan_only, compromise]
        labels = ['Consensus Winner', 'Judge Pick', 'Fan Pick', 'Compromise Winner']
        colors = ['#2ca02c', '#6C63FF', '#FF6B6B', 'gray']
        
        final_counts, final_labels, final_colors = [], [], []
        for c, l, col in zip(counts, labels, colors):
            if c > 0:
                final_counts.append(c)
                final_labels.append(l)
                final_colors.append(col)
        
        plt.pie(final_counts, labels=final_labels, colors=final_colors, autopct='%1.1f%%', 
                startangle=140, pctdistance=0.85, 
                textprops={'fontsize': 11, **FONT_ARGS}, wedgeprops={'width': 0.4, 'edgecolor': 'w'})
        
        plt.title('New Champions under 50/50 System', fontsize=14, fontweight='bold', **FONT_ARGS)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '2_Champion_Composition.png'), dpi=300)
        plt.close()

    # ==================================================
    # 图 3: Bobby Bones 案例
    # ==================================================
    if not bobby_df.empty:
        plt.figure(figsize=(10, 6))
        
        plt.plot(bobby_df['Week'], bobby_df['Original_Rank_Simulated'], 
                 marker='o', linestyle='--', color='gray', label='Original System (Rank 1)', alpha=0.6)
        plt.plot(bobby_df['Week'], bobby_df['New_Rank'], 
                 marker='D', linestyle='-', color='#FF9F40', linewidth=3, label='50/50 Z-Score System')
        
        plt.gca().invert_yaxis()
        
        plt.title('Case Study: Bobby Bones (50/50 Weight)', fontsize=14, fontweight='bold', **FONT_ARGS)
        plt.xlabel('Week', fontsize=12, **FONT_ARGS)
        plt.ylabel('Rank (Lower is Better)', fontsize=12, **FONT_ARGS)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        final_week = bobby_df['Week'].max()
        final_rank = bobby_df[bobby_df['Week']==final_week]['New_Rank'].values[0]
        
        plt.annotate(f'Final Rank: {int(final_rank)}', 
                     xy=(final_week, final_rank), 
                     xytext=(final_week-3, final_rank+3),
                     arrowprops=dict(facecolor='#FF9F40', shrink=0.05),
                     fontsize=12, color='#FF9F40', fontweight='bold', **FONT_ARGS)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '3_Bobby_Bones_Validation.png'), dpi=300)
        plt.close()

if __name__ == "__main__":
    run_q4_improved()