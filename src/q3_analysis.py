import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.font_manager as fm
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 🛡️ 字体与样式设置 (English Version)
# ==========================================
def get_font_and_labels():
    """
    加载通用字体并返回英文标签配置。
    """
    selected_font = None
    # 优先找 Arial 或 Times New Roman，保证英文论文的质感
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
        'title': 'Static Feature Correlation Heatmap', 
        'cols': {
            'Judge Score': 'Judge Score', 
            'Est_Fan_Votes': 'Fan Votes', 
            'celebrity_age_during_season': 'Age',
            'Partner_Code': 'Partner', 
            'Industry_Code': 'Industry',
            'Homestate_Code': 'Homestate', 
            'Region_Code': 'Region'
        },
        'cbar_label': 'Pearson Correlation'
    }
    return selected_font, labels_en

FONT_PROP, LABELS = get_font_and_labels()
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

def run_q3_baseline():
    print("🚀 [Q3 Baseline] Starting Static Feature Correlation Scan (English)...")
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    q1_path = os.path.join(base_path, 'output', 'output_q1_plus', 'improved_predictions.csv')
    raw_path = os.path.join(base_path, 'data', '2026_MCM_Problem_C_Data.csv')
    out_dir = os.path.join(base_path, 'output', 'output_q3')
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    if not os.path.exists(q1_path) or not os.path.exists(raw_path):
        print("❌ Data files not found.")
        return

    # 1. Data Merge
    df_q1 = pd.read_csv(q1_path)
    df_raw = pd.read_csv(raw_path).rename(columns={'celebrity_name': 'Contestant', 'season': 'Season'})
    
    target_cols = ['Season', 'Contestant', 'ballroom_partner', 'celebrity_industry', 
                   'celebrity_homestate', 'celebrity_homecountry/region', 'celebrity_age_during_season']
    
    valid_cols = [c for c in target_cols if c in df_raw.columns]
    df_static = df_raw[valid_cols].drop_duplicates(subset=['Season', 'Contestant'])
    df = pd.merge(df_q1, df_static, on=['Season', 'Contestant'], how='left')

    # 2. Label Encoding
    le = LabelEncoder()
    if 'ballroom_partner' in df.columns:
        df['Partner_Code'] = le.fit_transform(df['ballroom_partner'].fillna('Unknown').astype(str))
    if 'celebrity_industry' in df.columns:
        df['Industry_Code'] = le.fit_transform(df['celebrity_industry'].fillna('Unknown').astype(str))
    if 'celebrity_homestate' in df.columns:
        df['Homestate_Code'] = le.fit_transform(df['celebrity_homestate'].fillna('Unknown').astype(str))
    if 'celebrity_homecountry/region' in df.columns:
        df['Region_Code'] = le.fit_transform(df['celebrity_homecountry/region'].fillna('Unknown').astype(str))
    if 'celebrity_age_during_season' in df.columns:
        df['celebrity_age_during_season'] = df['celebrity_age_during_season'].fillna(df['celebrity_age_during_season'].mean())

    # 3. Feature Selection (确保没有 Week)
    analysis_cols = ['Judge Score', 'Est_Fan_Votes', 'celebrity_age_during_season', 
                     'Partner_Code', 'Industry_Code', 'Homestate_Code', 'Region_Code']
    
    final_cols = [c for c in analysis_cols if c in df.columns]
    df_plot = df[final_cols].rename(columns=LABELS['cols'])
    
    # 4. Correlation Calculation
    corr_matrix = df_plot.corr()

    # 5. Plotting
    plt.figure(figsize=(10, 8))
    
    # 🔥 修复点：cbar_kws 里不放 fontproperties，只放 label
    heatmap_args = {
        'annot': True, 
        'fmt': ".2f", 
        'cmap': 'coolwarm', 
        'vmin': -0.3, 
        'vmax': 0.3, 
        'square': True, 
        'linewidths': 1,
        'cbar_kws': {'label': LABELS['cbar_label']} 
    }

    ax = sns.heatmap(corr_matrix, **heatmap_args)
    
    # 🔥 修复点：图画完后，再手动设置 Colorbar 的字体
    if FONT_PROP:
        cbar = ax.collections[0].colorbar
        cbar.set_label(LABELS['cbar_label'], fontproperties=FONT_PROP)
    
    # Set titles and labels
    title_args = {'label': LABELS['title'], 'fontsize': 16, 'fontweight': 'bold', 'pad': 20}
    if FONT_PROP: title_args['fontproperties'] = FONT_PROP
    plt.title(**title_args)
    
    xticks_args = {'rotation': 45, 'ha': 'right', 'fontsize': 11}
    if FONT_PROP: xticks_args['fontproperties'] = FONT_PROP
    plt.xticks(**xticks_args)
    
    yticks_args = {'rotation': 0, 'fontsize': 11}
    if FONT_PROP: yticks_args['fontproperties'] = FONT_PROP
    plt.yticks(**yticks_args)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(out_dir, '1_global_correlation_scan.png')
    if os.path.exists(save_path):
        try: os.remove(save_path)
        except: pass
        
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"✅ [Success] English heatmap saved to: {save_path}")

if __name__ == "__main__":
    run_q3_baseline()