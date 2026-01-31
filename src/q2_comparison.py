import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import save_results

def run_q2_baseline():
    print("🚀 [Q2 Baseline] 重新运行赛制对比 (Using Real Data)...")
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_path, 'output', 'output_q1_plus', 'improved_predictions.csv')
    
    if not os.path.exists(input_path):
        return

    df = pd.read_csv(input_path)
    comparison_results = []
    
    # 🔥 检查：确保使用了真实数据
    if 'Judge Score' not in df.columns:
        print("❌ 警告：输入数据缺少 Judge Score，请重新运行 Q1！")
        return

    for (season, week), group in df.groupby(['Season', 'Week']):
        # 使用真实数据计算
        j_rank = group['Judge Score'].rank(ascending=False)
        f_rank = group['Est_Fan_Votes'].rank(ascending=False)
        
        # Rank System Result
        rank_outcome = (j_rank + f_rank).rank(ascending=True)

        # Percent System Result
        # 注意：这里我们做一个防御性处理，防止评委总分为0
        j_sum = group['Judge Score'].sum()
        j_pct = group['Judge Score'] / j_sum if j_sum > 0 else 0
        f_pct = group['Est_Fan_Votes']
        percent_outcome = (j_pct + f_pct).rank(ascending=False)

        for idx, row in group.iterrows():
            local_idx = group.index.get_loc(idx)
            rank_place = rank_outcome.iloc[local_idx]
            pct_place = percent_outcome.iloc[local_idx]
            
            comparison_results.append({
                'Season': season,
                'Week': week,
                'Contestant': row['Contestant'],
                'Rank_System_Place': rank_place,
                'Percent_System_Place': pct_place,
                'Diff': rank_place - pct_place
            })

    save_results(comparison_results, 'output_q2', 'q2_baseline_comparison.csv')

if __name__ == "__main__":
    run_q2_baseline()