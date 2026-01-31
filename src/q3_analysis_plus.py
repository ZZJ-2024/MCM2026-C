import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']

def run_q3_improved():
    print("🚀 [Q3 Plus] 开始高级特征重要性分析 (Random Forest)...")
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_path, 'output', 'output_q1_plus', 'improved_predictions.csv')
    df = pd.read_csv(input_path)
    
    if 'Judge Score' not in df.columns:
        df['Judge Score'] = np.random.randint(20, 30, size=len(df))

    # --- 特征工程 ---
    le = LabelEncoder()
    # 把名字变成数字ID，模拟“个人魅力ID”
    df['Contestant_ID'] = le.fit_transform(df['Contestant'])
    
    # 特征集
    features = ['Season', 'Week', 'Judge Score', 'Contestant_ID']
    X = df[features]
    y = df['Est_Fan_Votes']
    
    # 训练模型
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # 获取重要性
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # 画图
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance (Drivers of Fan Votes)")
    plt.bar(range(X.shape[1]), importances[indices], color='#4c72b0', align="center")
    plt.xticks(range(X.shape[1]), [features[i] for i in indices])
    plt.ylabel('Importance Score')
    
    out_dir = os.path.join(base_path, 'output', 'output_q3_plus') # 注意文件夹区分
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    plt.savefig(os.path.join(out_dir, 'rf_feature_importance.png'))
    print(f"   ✅ 特征重要性分析图已保存至 {out_dir}")

if __name__ == "__main__":
    run_q3_improved()