import pandas as pd
import numpy as np
from scipy.optimize import minimize
import os
import re
import shutil  # <--- 新增这个库，用来删文件
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# --- 1. 新增：自动清空输出文件夹函数 ---
def clean_previous_outputs():
    """
    每次运行前，清空 output 文件夹下的所有子文件夹内容
    但保留文件夹结构
    """
    print("🧹 [System] 正在清理旧的输出文件...")
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_path, 'output')
    
    # 如果 output 文件夹不存在，直接跳过
    if not os.path.exists(output_dir):
        return

    # 遍历 output 下的所有子文件夹 (q1, q2, q3...)
    for folder_name in os.listdir(output_dir):
        folder_path = os.path.join(output_dir, folder_name)
        
        # 只清理文件夹，不清理单独的文件（如果有的话）
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path) # 删除文件
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path) # 删除子文件夹
                except Exception as e:
                    print(f"⚠️ 无法删除 {file_path}: {e}")
    
    print("✨ 清理完毕！环境已重置。")

# --- 2. 数据预处理器 ---
class DataPreprocessor:
    def __init__(self, filepath):
        self.raw_df = pd.read_csv(filepath)
        self.raw_df.columns = [c.strip().lower().replace(' ', '_') for c in self.raw_df.columns]

    def get_seasons(self):
        if 'season' not in self.raw_df.columns:
            raise ValueError("CSV中找不到 'season' 列，请检查表头！")
        return sorted(self.raw_df['season'].unique())

    def get_weekly_data(self, season, week):
        season_df = self.raw_df[self.raw_df['season'] == season].copy()
        active_contestants = []
        
        for idx, row in season_df.iterrows():
            result_str = str(row['results']) 
            contestant_name = row['celebrity_name']
            
            elim_match = re.search(r'Eliminated Week (\d+)', result_str, re.IGNORECASE)
            if elim_match:
                elim_week = int(elim_match.group(1))
                if elim_week < week:
                    continue 
            
            # --- 🔥 修复点：这里加了 'r'，消除了那个黄色警告 ---
            score_cols = [c for c in season_df.columns if re.match(rf'week{week}_judge\d+_score', c)]
            
            if not score_cols:
                continue

            week_scores = row[score_cols]
            if week_scores.isna().all():
                continue
            
            total_score = week_scores.sum(skipna=True)
            status = "Safe"
            if elim_match and int(elim_match.group(1)) == week:
                status = "Eliminated"
            
            active_contestants.append({
                'Contestant': contestant_name,
                'Judge Score': total_score,
                'Result': status
            })
            
        return pd.DataFrame(active_contestants)

# --- 3. 核心反推模型 (保持不变) ---
class VotingInverter:
    def __init__(self, weekly_data, season_num, trust_factor=0.0):
        self.data = weekly_data.copy()
        self.season_num = season_num
        self.trust_factor = trust_factor
        self.contestants = self.data['Contestant'].values
        self.n = len(self.contestants)
        
        elim_rows = self.data[self.data['Result'] == 'Eliminated']
        if len(elim_rows) == 0:
            self.has_elimination = False
        else:
            self.has_elimination = True
            self.elim_idx = self.data.index[self.data['Result'] == 'Eliminated'][0]
            self.eliminated_name = self.data.loc[self.elim_idx, 'Contestant']

        raw_scores = self.data['Judge Score'].values
        self.judge_sum = raw_scores.sum()
        self.judge_percent = raw_scores / self.judge_sum if self.judge_sum > 0 else np.ones(self.n)/self.n
        self.judge_ranks = np.argsort(np.argsort(-raw_scores)) + 1

    def solve(self):
        if not self.has_elimination:
            return {'votes': np.ones(self.n)/self.n, 'success': True, 'method': 'No_Elimination'}
        if self.season_num <= 2:
            return self._solve_rank_monte_carlo()
        else:
            return self._solve_percent_optimization()

    def _solve_percent_optimization(self):
        epsilon = 1e-5
        def objective(x):
            term_entropy = np.sum((x - 1.0/self.n)**2)
            term_judge = np.sum((x - self.judge_percent)**2)
            return (1 - self.trust_factor) * term_entropy + self.trust_factor * term_judge

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        for i in range(self.n):
            if i == self.elim_idx: continue
            def constr(x, idx=i, elim=self.elim_idx):
                return (self.judge_percent[idx] + x[idx]) - (self.judge_percent[elim] + x[elim]) - epsilon
            constraints.append({'type': 'ineq', 'fun': constr})

        x0 = np.ones(self.n) / self.n
        bounds = [(0.0, 1.0) for _ in range(self.n)]
        try:
            res = minimize(objective, x0, bounds=bounds, constraints=constraints, method='SLSQP')
            return {'votes': res.x, 'success': res.success, 'method': 'Percent_Opt'}
        except:
            return {'votes': x0, 'success': False, 'method': 'Percent_Fail'}

    def _solve_rank_monte_carlo(self):
        num_samples = 3000
        valid_samples = []
        samples = np.random.dirichlet(np.ones(self.n), num_samples)
        for x in samples:
            fan_ranks = np.argsort(np.argsort(-x)) + 1
            total_ranks = self.judge_ranks + fan_ranks
            if np.all(total_ranks[self.elim_idx] >= total_ranks):
                valid_samples.append(x)
        if len(valid_samples) > 0:
            return {'votes': np.mean(valid_samples, axis=0), 'success': True, 'method': 'Rank_MC'}
        else:
            return {'votes': np.ones(self.n)/self.n, 'success': False, 'method': 'Rank_Fail'}

def save_results(results_list, folder_name, file_name):
    if not results_list:
        return
    df = pd.DataFrame(results_list)
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_path, 'output', folder_name)
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    save_path = os.path.join(out_dir, file_name)
    df.to_csv(save_path, index=False)
    print(f"✅ 成功保存: {save_path}")