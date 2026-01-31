import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def run_q4_improved():
    print("🚀 [Q4 Plus] 开始高级赛制设计 (Adaptive Dynamic Weights)...")
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_path, 'output', 'output_q1_plus', 'improved_predictions.csv')
    df = pd.read_csv(input_path)
    
    if 'Judge Score' not in df.columns:
        df['Judge Score'] = 100 - df.groupby(['Season', 'Week'])['Est_Fan_Votes'].rank(ascending=False) * 5

    results = []
    
    # 按场次遍历
    for (season, week), group in df.groupby(['Season', 'Week']):
        group = group.copy()
        
        # --- 动态权重核心逻辑 ---
        # 假设最大周次是 12
        # w 从 0.3 (第1周) 线性增加到 0.8 (第12周)
        progress = min(week / 12, 1.0) 
        w_judge = 0.3 + 0.5 * progress
        
        # 归一化
        j_max = group['Judge Score'].max() if group['Judge Score'].max() > 0 else 1
        f_max = group['Est_Fan_Votes'].max() if group['Est_Fan_Votes'].max() > 0 else 1
        
        group['New_Score'] = w_judge * (group['Judge Score']/j_max) + (1-w_judge) * (group['Est_Fan_Votes']/f_max)
        group['Dynamic_Weight_J'] = w_judge
        
        results.append(group)
        
    final_df = pd.concat(results)
    
    out_dir = os.path.join(base_path, 'output', 'output_q4_plus')
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    final_df.to_csv(os.path.join(out_dir, 'q4_adaptive_simulation.csv'), index=False)
    
    # 画个权重变化图
    weeks = np.arange(1, 13)
    weights = [0.3 + 0.5 * min(w/12, 1.0) for w in weeks]
    plt.figure(figsize=(6, 4))
    plt.plot(weeks, weights, marker='o', color='crimson')
    plt.title('Dynamic Weight Strategy w(t)')
    plt.xlabel('Week')
    plt.ylabel('Weight of Judge')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'dynamic_weight_curve.png'))
    print(f"   ✅ 动态赛制模拟及曲线图已保存至 {out_dir}")

if __name__ == "__main__":
    run_q4_improved()