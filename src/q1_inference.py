import pandas as pd
import os
import sys

# 确保能找到 utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import DataPreprocessor, VotingInverter, save_results

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

    for season in seasons:
        # print(f"Processing Season {season}...", end='\r')
        for week in range(1, 13):
            weekly_df = processor.get_weekly_data(season, week)
            
            if weekly_df.empty or 'Eliminated' not in weekly_df['Result'].values:
                continue
            
            try:
                # Trust Factor = 0.0 (纯随机)
                solver = VotingInverter(weekly_df, season, trust_factor=0.0)
                res = solver.solve()
                
                if res['success']:
                    for i, name in enumerate(solver.contestants):
                        # --- 🔥 核心修改：保存更多原始信息 ---
                        # 从原始数据中找回该选手的评委分和结果
                        original_row = weekly_df[weekly_df['Contestant'] == name].iloc[0]
                        
                        all_results.append({
                            'Season': season,
                            'Week': week,
                            'Contestant': name,
                            'Est_Fan_Votes': res['votes'][i], # 估算的粉丝票
                            'Judge Score': original_row['Judge Score'], # 🔥 真实的评委分
                            'Official Result': original_row['Result'],   # 🔥 真实的淘汰结果
                            'Model': 'Baseline'
                        })
            except Exception as e:
                pass # 忽略计算错误的个例

    print("\n✅ Q1 Baseline 数据增强版处理完成！")
    save_results(all_results, 'output_q1', 'baseline_predictions.csv')

if __name__ == "__main__":
    run_baseline()