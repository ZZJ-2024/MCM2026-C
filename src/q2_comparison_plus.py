import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import save_results

def run_q2_improved():
    print("🚀 [Q2 Plus] 重新运行高级模拟 (Using Real Data)...")
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_path, 'output', 'output_q1_plus', 'improved_predictions.csv')
    df = pd.read_csv(input_path)
    
    if 'Judge Score' not in df.columns:
        print("❌ 警告：输入数据缺少 Judge Score，请重新运行 Q1！")
        return

    simulation_results = []

    for (season, week), group in df.groupby(['Season', 'Week']):
        # 1. 正常的 Rank 制
        j_rank = group['Judge Score'].rank(ascending=False)
        f_rank = group['Est_Fan_Votes'].rank(ascending=False)
        total_rank = j_rank + f_rank
        
        # 正常情况下被淘汰的（Rank数值最大的）
        eliminated_standard = group.loc[total_rank.idxmax()]['Contestant']

        # 2. 评委拯救 (Judge Save)
        # 找到倒数两名
        bottom_2_indices = total_rank.nlargest(2).index
        bottom_2_contestants = group.loc[bottom_2_indices]
        
        # 评委分较低的那个被淘汰 (如果同分，这里简单取第一个，实际规则可能更复杂)
        loser_idx = bottom_2_contestants['Judge Score'].idxmin()
        eliminated_with_save = group.loc[loser_idx]['Contestant']

        is_reversal = (eliminated_standard != eliminated_with_save)
        
        simulation_results.append({
            'Season': season,
            'Week': week,
            'Standard_Eliminated': eliminated_standard,
            'Save_System_Eliminated': eliminated_with_save,
            'Is_Reversal': is_reversal
        })

    save_results(simulation_results, 'output_q2_plus', 'q2_judge_save_simulation.csv')
    
    # 统计
    reversal_rate = pd.DataFrame(simulation_results)['Is_Reversal'].mean()
    print(f"   -> 📊 更新后的逆转率: {reversal_rate*100:.2f}%")
    if reversal_rate > 0:
        print("   ✅ 数据正常！我们发现了评委拯救机制的作用！")
    else:
        print("   ⚠️ 逆转率依然为0，可能说明评委分和粉丝票高度一致，或者样本量太小。")

if __name__ == "__main__":
    run_q2_improved()