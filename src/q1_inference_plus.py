import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import DataPreprocessor, VotingInverter, save_results

def run_improved():
    print("🚀 [Refining Q1+] 开始运行微调模型 (Data-Rich Version)...")
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'data', '2026_MCM_Problem_C_Data.csv')
    
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
                # Trust Factor = 0.3 (微调)
                solver = VotingInverter(weekly_df, season, trust_factor=0.3)
                res = solver.solve()
                
                if res['success']:
                    for i, name in enumerate(solver.contestants):
                        # --- 🔥 核心修改：保存更多原始信息 ---
                        original_row = weekly_df[weekly_df['Contestant'] == name].iloc[0]
                        
                        all_results.append({
                            'Season': season,
                            'Week': week,
                            'Contestant': name,
                            'Est_Fan_Votes': res['votes'][i],
                            'Judge Score': original_row['Judge Score'], # 🔥 带上它！
                            'Official Result': original_row['Result'],   # 🔥 带上它！
                            'Model': 'Improved'
                        })
            except:
                pass

    print("\n✅ Q1 Improved 数据增强版处理完成！")
    save_results(all_results, 'output_q1_plus', 'improved_predictions.csv')

if __name__ == "__main__":
    run_improved()