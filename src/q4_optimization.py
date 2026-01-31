import pandas as pd
import matplotlib.pyplot as plt
import os

def run_q4_baseline():
    print("🚀 [Q4 Baseline] 开始基础赛制设计 (Fixed Weights)...")
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_path, 'output', 'output_q1_plus', 'improved_predictions.csv')
    df = pd.read_csv(input_path)
    
    if 'Judge Score' not in df.columns:
        df['Judge Score'] = 100 - df.groupby(['Season', 'Week'])['Est_Fan_Votes'].rank(ascending=False) * 5

    # --- 策略：固定 50% 评委 + 50% 粉丝 ---
    w_judge = 0.5
    
    # 归一化
    df['J_Norm'] = df.groupby(['Season', 'Week'])['Judge Score'].transform(lambda x: x / x.max())
    df['F_Norm'] = df.groupby(['Season', 'Week'])['Est_Fan_Votes'].transform(lambda x: x / x.max())
    
    # 计算新分数
    df['New_Score'] = w_judge * df['J_Norm'] + (1 - w_judge) * df['F_Norm']
    
    # 简单保存前几行看看
    out_dir = os.path.join(base_path, 'output', 'output_q4')
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    df.head(20).to_csv(os.path.join(out_dir, 'q4_fixed_weight_simulation.csv'), index=False)
    print(f"   ✅ 固定权重模拟结果已保存至 {out_dir}")

if __name__ == "__main__":
    run_q4_baseline()