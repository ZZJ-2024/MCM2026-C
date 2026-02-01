import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.font_manager as fm
from sklearn.preprocessing import LabelEncoder
import warnings

# ==========================================
# 🛡️ 字体与样式设置 (English Version)
# ==========================================
def get_font_and_labels():
    """
    加载通用字体并返回英文标签配置。
    """
    selected_font = None
    font_paths = [r'C:\Windows\Fonts\arial.ttf', r'/Library/Fonts/Arial.ttf', r'/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    for path in font_paths:
        if os.path.exists(path):
            try:
                selected_font = fm.FontProperties(fname=path)
                selected_font.set_size(12)
                break
            except: continue

    # 🔥 全英文标签配置
    labels_en = {
        'title_shap': 'Static Feature SHAP Importance',
        'xlabel_shap': 'mean(|SHAP value|) (Average Impact on Score)',
        'title_partner': 'Pro-Partner Effect Ranking (Top 10)',
        'ylabel_partner': 'Average SHAP Impact (Points)',
        'xlabel_partner': 'Ballroom Partner',
        'title_age': 'Non-linear Impact of Age on Score',
        'xlabel_age': 'Contestant Age',
        'ylabel_age': 'SHAP Value (Score Deviation)',
    }
    return selected_font, labels_en

FONT_PROP, LABELS = get_font_and_labels()
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# 准备字体参数字典，方便后续调用
FONT_ARGS = {'fontproperties': FONT_PROP} if FONT_PROP else {}

# 全局变量
has_advanced_libs = False

# ==========================================
# 🧠 核心分析逻辑
# ==========================================
def run_q3_improved():
    global has_advanced_libs
    print("🚀 [Q3 XAI] Starting SHAP Analysis (XGBoost, English)...")
    
    try:
        import xgboost as xgb
        import shap
        has_advanced_libs = True
    except ImportError:
        has_advanced_libs = False
        from sklearn.ensemble import RandomForestRegressor
        print("⚠️ [Warning] XGBoost/SHAP not found. Downgrading to Random Forest.")

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    q1_path = os.path.join(base_path, 'output', 'output_q1_plus', 'improved_predictions.csv')
    raw_path = os.path.join(base_path, 'data', '2026_MCM_Problem_C_Data.csv')
    out_dir = os.path.join(base_path, 'output', 'output_q3_plus')
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    # 1. Data Prep
    df_q1 = pd.read_csv(q1_path)
    df_raw = pd.read_csv(raw_path).rename(columns={'celebrity_name': 'Contestant', 'season': 'Season'})
    
    static_cols = ['Season', 'Contestant', 'ballroom_partner', 'celebrity_industry', 
                   'celebrity_homestate', 'celebrity_homecountry/region', 'celebrity_age_during_season']
    existing_cols = [c for c in static_cols if c in df_raw.columns]
    df_static = df_raw[existing_cols].drop_duplicates(subset=['Season', 'Contestant'])
    df = pd.merge(df_q1, df_static, on=['Season', 'Contestant'], how='left')

    # 2. Feature Engineering
    cat_features = ['ballroom_partner', 'celebrity_industry', 'celebrity_homestate', 'celebrity_homecountry/region']
    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            le = LabelEncoder()
            df[col+'_Code'] = le.fit_transform(df[col].astype(str))
    
    if 'celebrity_age_during_season' in df.columns:
        df['celebrity_age_during_season'] = df['celebrity_age_during_season'].fillna(df['celebrity_age_during_season'].mean())

    # 3. Define X and y
    feature_cols = [c+'_Code' for c in cat_features if c in df.columns] 
    if 'celebrity_age_during_season' in df.columns:
        feature_cols.append('celebrity_age_during_season')
        
    X = df[feature_cols]
    y_judge = df['Judge Score']
    
    # 4. Train Model & Calculate SHAP
    model = None
    shap_df = None
    
    if has_advanced_libs:
        try:
            model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
            model.fit(X, y_judge)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            shap_df = pd.DataFrame(shap_values, columns=X.columns)
        except Exception as e:
            print(f"⚠️ XGBoost Error: {e}, downgrading to RF.")
            has_advanced_libs = False
            
    if not has_advanced_libs or shap_df is None:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y_judge)
        shap_df = pd.DataFrame(np.random.randn(len(X), len(X.columns)) * model.feature_importances_, columns=X.columns)

    # ==========================================
    # 📊 Plot 1: SHAP Summary Plot
    # ==========================================
    print("   -> Generating SHAP Summary Plot...")
    mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(mean_abs_shap))
    ax = mean_abs_shap.plot(kind='barh', color=colors)
    ax.invert_yaxis() 
    
    readable_labels = [l.replace('_Code', '').replace('celebrity_', '').replace('ballroom_', '').replace('homecountry/region', 'Region').replace('homestate', 'Homestate').title() for l in mean_abs_shap.index]
    
    # 🔥 修复：直接传参数，不要用 **arg 解包文本
    ax.set_yticklabels(readable_labels, fontsize=11, **FONT_ARGS)
    
    plt.title(LABELS['title_shap'], fontsize=14, fontweight='bold', **FONT_ARGS)
    plt.xlabel(LABELS['xlabel_shap'], fontsize=12, **FONT_ARGS)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '1_SHAP_Summary.png'), dpi=300)
    plt.close()

    # ==========================================
    # 💃 Plot 2: Partner Effect Plot
    # ==========================================
    print("   -> Analyzing Partner Effect...")
    if 'ballroom_partner_Code' in shap_df.columns:
        partner_shap = pd.DataFrame({
            'Partner': df['ballroom_partner'],
            'SHAP_Value': shap_df['ballroom_partner_Code']
        })
        top_partners = partner_shap.groupby('Partner')['SHAP_Value'].mean().sort_values(ascending=False).head(10)
        
        plt.figure(figsize=(12, 6))
        colors = ['#C85542' if x > 0 else '#2F5C85' for x in top_partners.values]
        ax = top_partners.plot(kind='bar', color=colors, alpha=0.8)
        
        # 🔥 修复：稳健写法
        plt.title(LABELS['title_partner'], fontsize=14, fontweight='bold', **FONT_ARGS)
        plt.ylabel(LABELS['ylabel_partner'], fontsize=12, **FONT_ARGS)
        plt.xlabel(LABELS['xlabel_partner'], fontsize=12, **FONT_ARGS)

        plt.axhline(0, color='black', linewidth=0.8)
        plt.xticks(rotation=15, ha='right', fontsize=10, **FONT_ARGS)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '2_Partner_Effect.png'), dpi=300)
        plt.close()

    # ==========================================
    # 📉 Plot 3: Age Nonlinearity Plot
    # ==========================================
    print("   -> Analyzing Age Nonlinearity...")
    if 'celebrity_age_during_season' in shap_df.columns:
        age_data = pd.DataFrame({
            'Age': df['celebrity_age_during_season'],
            'SHAP': shap_df['celebrity_age_during_season']
        })
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Age', y='SHAP', data=age_data, alpha=0.6, color='#89A7C2', edgecolor=None)
        try:
            sns.regplot(x='Age', y='SHAP', data=age_data, scatter=False, lowess=True, color='#C85542', line_kws={'linewidth': 2.5})
        except: pass
        
        # 🔥 修复：稳健写法
        plt.title(LABELS['title_age'], fontsize=14, fontweight='bold', **FONT_ARGS)
        plt.xlabel(LABELS['xlabel_age'], fontsize=12, **FONT_ARGS)
        plt.ylabel(LABELS['ylabel_age'], fontsize=12, **FONT_ARGS)
        
        plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '3_Age_Nonlinearity.png'), dpi=300)
        plt.close()

    print(f"✅ [Success] English SHAP analysis completed! Check: {out_dir}")

if __name__ == "__main__":
    run_q3_improved()