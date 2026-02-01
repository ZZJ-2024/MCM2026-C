import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def run_q3_improved():
    print("🚀 [Q3 Advanced] 开始双模型深度归因分析...")
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    q1_output_path = os.path.join(base_path, 'output', 'output_q1_plus', 'improved_predictions.csv')
    raw_data_path = os.path.join(base_path, 'data', '2026_MCM_Problem_C_Data.csv')
    out_dir = os.path.join(base_path, 'output', 'output_q3_plus')
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    # 1. 深度数据融合 (引入你强调的所有原始相关列)
    df_q1 = pd.read_csv(q1_output_path)
    df_raw = pd.read_csv(raw_data_path)
    
    df_raw_renamed = df_raw.rename(columns={'celebrity_name': 'Contestant', 'season': 'Season'})
    # 考虑所有相关列
    target_cols = ['Season', 'Contestant', 'celebrity_industry', 'celebrity_age_during_season', 
                   'ballroom_partner', 'celebrity_homestate']
    available_cols = [c for c in target_cols if c in df_raw_renamed.columns]
    df_static = df_raw_renamed[available_cols].drop_duplicates(subset=['Season', 'Contestant'])
    
    df_full = pd.merge(df_q1, df_static, on=['Season', 'Contestant'], how='left')
    
    # 2. 特征工程 (构造历史数据，分析影响程度)
    le = LabelEncoder()
    # 对分类变量进行编码
    cat_features = ['celebrity_industry', 'ballroom_partner', 'celebrity_homestate']
    for col in cat_features:
        if col in df_full.columns:
            df_full[col+'_Code'] = le.fit_transform(df_full[col].fillna('Unknown').astype(str))

    # 构造历史平均分（代表选手硬实力）
    df_sorted = df_full.sort_values(['Season', 'Contestant', 'Week'])
    df_sorted['Avg_Judge_Hist'] = df_sorted.groupby(['Season', 'Contestant'])['Judge Score'].transform(lambda x: x.expanding().mean().shift(1))
    df_sorted['Prev_Fan_Votes'] = df_sorted.groupby(['Season', 'Contestant'])['Est_Fan_Votes'].shift(1)
    df_model = df_sorted.fillna(0)

    # 3. 双模型驱动分析
    # --- 模型 A: 评委决策因素 ---
    features_j = ['Avg_Judge_Hist', 'celebrity_age_during_season', 'Industry_Code', 'Partner_Code', 'Week']
    # 过滤掉不存在的特征
    features_j = [f for f in features_j if f in df_model.columns or f.replace('_Code','') in df_model.columns]
    
    rf_j = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_j.fit(df_model[features_j], df_model['Judge Score'])
    
    # --- 模型 B: 粉丝决策因素 ---
    # 粉丝会受到“当周评委分”的引导
    features_f = ['Judge Score', 'Prev_Fan_Votes', 'celebrity_age_during_season', 'Industry_Code', 'Week']
    features_f = [f for f in features_f if f in df_model.columns]
    
    rf_f = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_f.fit(df_model[features_f], df_model['Est_Fan_Votes'])

    # 4. 可视化差异 (影响程度与区别)
    # 统一特征展示名称
    display_map = {
        'Avg_Judge_Hist': '历史表现 (Past Performance)',
        'Judge Score': '当周评委表现 (Current Performance)',
        'Prev_Fan_Votes': '粉丝粘性 (Fan Inertia)',
        'celebrity_age_during_season': '年龄 (Age)',
        'Industry_Code': '职业背景 (Industry)',
        'Partner_Code': '舞伴效应 (Partner)',
        'Week': '赛程进度 (Week)'
    }

    # 整理结果
    imp_j = pd.DataFrame({'Factor': [display_map.get(f, f) for f in features_j], 'Importance': rf_j.feature_importances_, 'Type': '评委 (Judge)'})
    imp_f = pd.DataFrame({'Factor': [display_map.get(f, f) for f in features_f], 'Importance': rf_f.feature_importances_, 'Type': '粉丝 (Fan)'})
    combined = pd.concat([imp_j, imp_f])

    plt.figure(figsize=(12, 7))
    sns.barplot(data=combined, x='Importance', y='Factor', hue='Type', palette={'评委 (Judge)': '#FF9F40', '粉丝 (Fan)': '#36A2EB'})
    plt.title('评委与粉丝打分影响因素的影响程度对比分析', fontsize=14, fontweight='bold')
    plt.xlabel('特征重要性 (影响权重)', fontsize=12)
    plt.ylabel('')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    save_path = os.path.join(out_dir, 'q3_dual_attribution_analysis.png')
    plt.savefig(save_path)
    plt.close()
    print(f"   ✅ 深度归因分析图已保存至: {save_path}")

if __name__ == "__main__":
    run_improved()