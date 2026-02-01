import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def run_q3_baseline():
    print("🚀 [Q3 Baseline] 开始基础因素相关性分析...")
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    q1_output_path = os.path.join(base_path, 'output', 'output_q1_plus', 'improved_predictions.csv')
    raw_data_path = os.path.join(base_path, 'data', '2026_MCM_Problem_C_Data.csv')
    out_dir = os.path.join(base_path, 'output', 'output_q3')
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    # 1. 数据加载与融合
    df_q1 = pd.read_csv(q1_output_path)
    df_raw = pd.read_csv(raw_data_path)
    
    # 融合原始列 (职业、年龄)
    df_raw_renamed = df_raw.rename(columns={'celebrity_name': 'Contestant', 'season': 'Season'})
    df_static = df_raw_renamed[['Season', 'Contestant', 'celebrity_industry', 'celebrity_age_during_season']].drop_duplicates()
    df = pd.merge(df_q1, df_static, on=['Season', 'Contestant'], how='left')
    
    # 2. 编码与清洗
    le = LabelEncoder()
    df['Industry_Code'] = le.fit_transform(df['celebrity_industry'].fillna('Unknown').astype(str))
    
    # 3. 计算相关性
    # 我们对比：哪些因素影响评委分，哪些因素影响粉丝票
    corr_cols = ['Judge Score', 'Est_Fan_Votes', 'celebrity_age_during_season', 'Industry_Code', 'Week']
    corr_matrix = df[corr_cols].corr()
    
    # 4. 绘图：侧重于对比
    # 提取对 Judge Score 和 Est_Fan_Votes 的相关性
    comparison = corr_matrix[['Judge Score', 'Est_Fan_Votes']].drop(['Judge Score', 'Est_Fan_Votes'])
    comparison.columns = ['对评委分的影响', '对粉丝票的影响']
    
    plt.figure(figsize=(10, 6))
    comparison.plot(kind='bar', ax=plt.gca(), color=['#FF9F40', '#36A2EB'])
    plt.title('因素对评委与粉丝打分的相关性对比 (Baseline)', fontsize=14)
    plt.ylabel('相关系数 (Pearson r)')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path = os.path.join(out_dir, 'q3_baseline_correlation.png')
    plt.savefig(save_path)
    plt.close()
    print(f"   ✅ 基础分析图已保存至: {save_path}")

if __name__ == "__main__":
    run_q3_baseline()