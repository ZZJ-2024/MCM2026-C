import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches

# ==========================================
# 🎨 Nature/Science 顶级期刊绘图风格
# ==========================================
def set_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    # 尝试使用 Helvetica 或 Arial，如果没有则回退
    plt.rcParams['font.family'] = 'sans-serif'
    font_paths = [r'C:\Windows\Fonts\arial.ttf', r'/Library/Fonts/Arial.ttf']
    prop = None
    for p in font_paths:
        if os.path.exists(p):
            prop = fm.FontProperties(fname=p)
            prop.set_size(12)
            break
    return prop, {'fontproperties': prop} if prop else {}

FONT_PROP, FONT_ARGS = set_style()

# 定义学术级配色
COLOR_JUDGE = '#2A4B7C'  # 深海军蓝 (代表专业)
COLOR_FAN = '#E63946'    # 活力红 (代表大众)
COLOR_GAP_POS = '#457B9D' # 柔和蓝
COLOR_GAP_NEG = '#E63946' # 柔和红
COLOR_GRAY = '#6C757D'

# ==========================================
# 🧠 核心模型: Tanh 有界积分系统
# ==========================================
def run_q4_strategy():
    print("🚀 [Q4 Final] Generating O-Prize Standard Visualizations...")
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_path, 'output', 'output_q1_plus', 'improved_predictions.csv')
    output_dir = os.path.join(base_path, 'output', 'output_q4') 
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    if not os.path.exists(input_path):
        print("❌ Data missing.")
        return

    df = pd.read_csv(input_path)
    if 'Judge Score' not in df.columns: return

    # --- 1. 计算核心指标 ---
    results = []
    for (season, week), group in df.groupby(['Season', 'Week']):
        group = group.copy()
        
        # Z-Score
        j_std = group['Judge Score'].std()
        f_std = group['Est_Fan_Votes'].std()
        j_std = 1.0 if j_std == 0 or np.isnan(j_std) else j_std
        f_std = 1.0 if f_std == 0 or np.isnan(f_std) else f_std
        
        z_judge = (group['Judge Score'] - group['Judge Score'].mean()) / j_std
        z_fan = (group['Est_Fan_Votes'] - group['Est_Fan_Votes'].mean()) / f_std
        
        # Tanh 激活 (Bounded Meritocracy)
        group['Score_Judge_Tanh'] = np.tanh(z_judge)
        group['Score_Fan_Tanh'] = np.tanh(z_fan)
        
        # 1:1 Fusion
        group['Final_Score'] = 0.5 * group['Score_Judge_Tanh'] + 0.5 * group['Score_Fan_Tanh']
        group['Final_Rank'] = group['Final_Score'].rank(ascending=False)
        
        # Original Rank (Comparison)
        rank_sum_old = group['Judge Score'].rank(ascending=False) + group['Est_Fan_Votes'].rank(ascending=False)
        group['Original_Rank'] = rank_sum_old.rank(ascending=True)
        
        results.append(group)
        
    final_df = pd.concat(results)
    final_df.to_csv(os.path.join(output_dir, 'q4_final_data.csv'), index=False)

    # --- 2. 准备绘图数据 ---
    balance_stats = []
    for week in sorted(final_df['Week'].unique()):
        week_data = final_df[final_df['Week'] == week]
        if len(week_data) < 5: continue
        
        c_judge = week_data['Final_Rank'].corr(week_data['Judge Score'], method='spearman')
        c_fan = week_data['Final_Rank'].corr(week_data['Est_Fan_Votes'], method='spearman')
        
        gap = abs(c_judge) - abs(c_fan)
        balance_stats.append({
            'Week': week, 'Gap': gap, 'Influence_Judge': abs(c_judge), 'Influence_Fan': abs(c_fan)
        })
    balance_df = pd.DataFrame(balance_stats)

    champions = []
    for season, group in final_df.groupby('Season'):
        winner = group.loc[group['Final_Rank'].idxmin()]
        is_j = (group['Judge Score'].rank(ascending=False)[winner.name] == 1)
        is_f = (group['Est_Fan_Votes'].rank(ascending=False)[winner.name] == 1)
        champions.append({'Type': (is_j, is_f)})
        
    bobby_case = final_df[(final_df['Season'] == 27) & (final_df['Contestant'] == 'Bobby Bones')].sort_values('Week')

    # --- 执行完美绘图 ---
    plot_perfect_evidence(balance_df, champions, bobby_case, output_dir)
    print(f"✅ [Success] Visualizations saved to: {output_dir}")


def plot_perfect_evidence(balance_df, champions, bobby_case, out_dir):
    """
    生成 O 奖级别的三张核心图表
    """
    
    # ==================================================
    # 图 1: Mechanism Tug-of-War (机制博弈图)
    # ==================================================
    if not balance_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        weeks = balance_df['Week']
        gap = balance_df['Gap']
        
        # 0 轴线 (Strong Baseline)
        ax.axhline(0, color='black', linewidth=1.5, zorder=2)
        
        # 柱状图
        colors = [COLOR_GAP_POS if g >= 0 else COLOR_GAP_NEG for g in gap]
        ax.bar(weeks, gap, color=colors, alpha=0.85, width=0.6, label='Net Influence', zorder=3)
        
        # 背景网格
        ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
        
        # 标注
        ax.set_title('Dynamic Equilibrium: The Influence Tug-of-War', fontsize=16, fontweight='bold', **FONT_ARGS)
        ax.set_xlabel('Competition Week', fontsize=12, **FONT_ARGS)
        ax.set_ylabel('Net Influence Gap (Judge - Fan)', fontsize=12, **FONT_ARGS)
        
        # 区域解释
        y_lim = max(abs(gap.max()), abs(gap.min())) * 1.3
        ax.set_ylim(-y_lim, y_lim)
        ax.text(weeks.min(), y_lim*0.85, "▲ Blue Zone: Meritocracy (Skill) Leads", color=COLOR_GAP_POS, fontweight='bold', **FONT_ARGS)
        ax.text(weeks.min(), -y_lim*0.85, "▼ Red Zone: Popularity (Votes) Leads", color=COLOR_GAP_NEG, fontweight='bold', **FONT_ARGS)
        
        # 简洁图例
        blue_patch = mpatches.Patch(color=COLOR_GAP_POS, label='Judge Dominates')
        red_patch = mpatches.Patch(color=COLOR_GAP_NEG, label='Fan Dominates')
        ax.legend(handles=[blue_patch, red_patch], loc='upper right', frameon=True, fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '1_Mechanism_TugOfWar.png'), dpi=300)
        plt.close()

    # ==================================================
    # 图 2: Legitimacy Analysis (合法性分析)
    # ==================================================
    if champions:
        counts = {'Consensus': 0, 'Merit': 0, 'Popular': 0, 'Compromise': 0}
        for c in champions:
            j, f = c['Type']
            if j and f: counts['Consensus'] += 1
            elif j: counts['Merit'] += 1
            elif f: counts['Popular'] += 1
            else: counts['Compromise'] += 1
            
        labels = ['Consensus Winner\n(Best of Both)', 'Merit Winner\n(Skill Driven)', 'Popular Winner\n(Vote Driven)', 'Balanced Winner']
        sizes = [counts['Consensus'], counts['Merit'], counts['Popular'], counts['Compromise']]
        colors = ['#2A9D8F', COLOR_JUDGE, COLOR_FAN, '#E9C46A']
        
        # Filter
        p_l, p_s, p_c = [], [], []
        for l, s, c in zip(labels, sizes, colors):
            if s > 0: p_l.append(l); p_s.append(s); p_c.append(c)
            
        fig, ax = plt.subplots(figsize=(8, 6))
        wedges, texts, autotexts = ax.pie(p_s, labels=p_l, colors=p_c, autopct='%1.1f%%', startangle=140, 
                                          textprops={'fontsize': 11, **FONT_ARGS}, 
                                          wedgeprops={'width': 0.5, 'edgecolor': 'white', 'linewidth': 2})
        
        # 中心文字
        ax.text(0, 0, "Champion\nTypes", ha='center', va='center', fontsize=12, fontweight='bold', color='#333333', **FONT_ARGS)
        
        ax.set_title('Legitimacy: Who Wins under Tanh System?', fontsize=14, fontweight='bold', **FONT_ARGS)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '2_Outcome_Legitimacy.png'), dpi=300)
        plt.close()

    # ==================================================
    # 图 3: Stress Test (Bobby Bones 完美版)
    # ==================================================
    # 关键改进：引入双坐标轴，同时展示“排名”和“评委打分”
    # 逻辑：展示他因为评委分低 (Low Judge Score) 才导致排名下降
    if not bobby_case.empty:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        weeks = bobby_case['Week']
        
        # --- 左轴：排名 (Rank) ---
        line1, = ax1.plot(weeks, bobby_case['Original_Rank'], color='gray', linestyle=':', marker='o', label='Original System (Rank)', alpha=0.6)
        line2, = ax1.plot(weeks, bobby_case['Final_Rank'], color=COLOR_FAN, linewidth=3, marker='D', label='New System (Rank)')
        
        ax1.set_ylabel('Rank (Lower is Better)', fontsize=12, fontweight='bold', color=COLOR_FAN, **FONT_ARGS)
        ax1.invert_yaxis() # 排名越小越好
        ax1.tick_params(axis='y', labelcolor=COLOR_FAN)
        
        # --- 右轴：评委分 (Judge Score) ---
        ax2 = ax1.twinx()
        # 用柱状图或面积图展示他的评委分，放在背景里
        bars = ax2.bar(weeks, bobby_case['Judge Score'], color=COLOR_JUDGE, alpha=0.2, width=0.4, label='Judge Score (Low)')
        
        ax2.set_ylabel('Judge Score (Technical)', fontsize=12, fontweight='bold', color=COLOR_JUDGE, **FONT_ARGS)
        ax2.set_ylim(0, 30) # 假设满分30
        ax2.tick_params(axis='y', labelcolor=COLOR_JUDGE)
        
        # 标题与解释
        ax1.set_title('Stress Test: Why Bobby Bones Lost? (Rank vs. Skill)', fontsize=16, fontweight='bold', **FONT_ARGS)
        ax1.set_xlabel('Competition Week', fontsize=12, **FONT_ARGS)
        
        # 关键标注：解释因果关系
        final_rank = bobby_case.iloc[-1]['Final_Rank']
        final_score = bobby_case.iloc[-1]['Judge Score']
        
        ax1.annotate(f"Rank Drops to {int(final_rank)}\ndue to Tanh limit", 
                     xy=(weeks.max(), final_rank), xytext=(weeks.max()-3, final_rank+1),
                     arrowprops=dict(facecolor=COLOR_FAN, shrink=0.05),
                     color=COLOR_FAN, fontweight='bold', **FONT_ARGS)
        
        ax2.text(weeks.min(), 5, "Background Bars: Judge Scores\n(Consistently Low)", color=COLOR_JUDGE, fontsize=10, style='italic', **FONT_ARGS)

        # 合并图例
        lines = [line1, line2, bars]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', frameon=True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '3_Stress_Test.png'), dpi=300)
        plt.close()

if __name__ == "__main__":
    run_q4_strategy()