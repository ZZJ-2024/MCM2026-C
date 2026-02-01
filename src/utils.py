import pandas as pd
import numpy as np
import re
import os
import shutil
from scipy.optimize import minimize, linprog
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 🧹 辅助功能
# ==========================================
def clean_previous_outputs(dirs_to_clean):
    print("🧹 [System] 正在清理旧的输出文件...")
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for dir_name in dirs_to_clean:
        dir_path = os.path.join(base_path, 'output', dir_name)
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
            except Exception as e:
                print(f"   ⚠️ 无法删除 {dir_name}: {e}")
    print("✨ 清理完毕！环境已重置。")

# ==========================================
# 🧹 数据预处理类
# ==========================================
class DataPreprocessor:
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到数据文件: {file_path}")
        self.raw_df = pd.read_csv(file_path)
    
    def get_seasons(self):
        return sorted(self.raw_df['season'].unique())
    
    def get_weekly_data(self, season, week):
        season_df = self.raw_df[self.raw_df['season'] == season]
        if season_df.empty: return pd.DataFrame()

        judge_cols = [c for c in season_df.columns if f'week{week}_judge' in c and 'score' in c]
        if not judge_cols: return pd.DataFrame()

        records = []
        for _, row in season_df.iterrows():
            scores = []
            for c in judge_cols:
                val = row[c]
                if pd.notna(val): scores.append(float(val))
            if not scores: continue
            
            total_score = sum(scores)
            result_str = str(row['results'])
            is_eliminated_this_week = False
            if 'Eliminated' in result_str:
                try:
                    elim_week_num = int(re.search(r'Week\s*(\d+)', result_str, re.IGNORECASE).group(1))
                    if elim_week_num == week: is_eliminated_this_week = True
                except: pass
            
            final_result_tag = "Eliminated" if is_eliminated_this_week else "Safe"
            records.append({
                'Contestant': row['celebrity_name'],
                'Judge Score': total_score,
                'Result': final_result_tag,
                'Placement': row['placement']
            })
        return pd.DataFrame(records)

# ==========================================
# 🔥 核心反演算法类 (Interval Analysis Version)
# ==========================================
class VotingInverter:
    def __init__(self, week_df, season_num, trust_factor=0.0):
        self.data = week_df.copy()
        self.season_num = season_num
        self.trust_factor = trust_factor 
        self.contestants = self.data['Contestant'].values
        self.n = len(self.contestants)
        
        elim_rows = self.data[self.data['Result'] == 'Eliminated']
        if elim_rows.empty:
            self.elim_idx = -1 
        else:
            elim_name = elim_rows.iloc[0]['Contestant']
            indices = np.where(self.contestants == elim_name)[0]
            self.elim_idx = indices[0] if len(indices) > 0 else -1
            
        raw_scores = self.data['Judge Score'].values
        self.judge_sum = raw_scores.sum()
        self.judge_percent = raw_scores / self.judge_sum if self.judge_sum > 0 else np.ones(self.n)/self.n
        self.judge_ranks = np.argsort(np.argsort(-raw_scores)) + 1

    def solve(self):
        if self.season_num <= 2 or self.season_num >= 28:
            return self._solve_rank_mc()
        else:
            return self._solve_percent_opt()

    def _solve_percent_opt(self):
        """
        百分比制求解器 (Interval Analysis Mode)
        思路：你的核心思想 —— 范围越大，不确定性越高。
        """
        if self.elim_idx == -1: return self._return_uniform()

        # ---------------------------------------------------
        # 🎯 任务一：求最优解 (Prediction)
        # ---------------------------------------------------
        prior = (1 - self.trust_factor) * (np.ones(self.n)/self.n) + \
                self.trust_factor * self.judge_percent
        
        def objective(x): return np.sum((x - prior)**2)
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        epsilon = 1e-4
        
        # 基础约束：每个人都必须赢过淘汰者
        for i in range(self.n):
            if i == self.elim_idx: continue
            def constr(x, idx=i, elim=self.elim_idx):
                return (self.judge_percent[idx] + x[idx]) - (self.judge_percent[elim] + x[elim]) - epsilon
            constraints.append({'type': 'ineq', 'fun': constr})

        x0 = prior
        bounds = [(0, 1) for _ in range(self.n)]
        
        best_votes = x0
        success = False
        try:
            res = minimize(objective, x0, bounds=bounds, constraints=constraints, method='SLSQP')
            if res.success:
                best_votes = res.x
                success = True
        except: pass

        if not success:
            return self._return_uniform(success=False)

        # ---------------------------------------------------
        # 📏 任务二：区间范围计算 (Interval Calculation) - 你的核心要求
        # 不再用随机，而是用优化算法算出每个人的 [Min, Max]
        # ---------------------------------------------------
        
        ranges = []
        
        # 只要任务一成功了，说明肯定有解。现在我们探索解的边界。
        # 我们需要运行 2*N 次优化：对每个人求 Min 和 Max
        
        for i in range(self.n):
            # 1. 找下限 (Min)
            # 目标函数：让 x[i] 越小越好
            def obj_min(x, idx=i): return x[idx]
            try:
                res_min = minimize(obj_min, best_votes, bounds=bounds, constraints=constraints, method='SLSQP')
                val_min = res_min.x[i] if res_min.success else best_votes[i]
            except: val_min = best_votes[i]
            
            # 2. 找上限 (Max)
            # 目标函数：让 x[i] 越大越好 (即 -x[i] 越小越好)
            def obj_max(x, idx=i): return -x[idx]
            try:
                res_max = minimize(obj_max, best_votes, bounds=bounds, constraints=constraints, method='SLSQP')
                val_max = res_max.x[i] if res_max.success else best_votes[i]
            except: val_max = best_votes[i]
            
            # 计算这一维度的可行宽度
            width = max(0, val_max - val_min)
            ranges.append(width)
        
        ranges = np.array(ranges)
        
        # --- 计算 Certainty Based on Range ---
        # 逻辑：
        # 无条件情况下，x[i] 的理论范围接近 [0, 1]，宽度为 1。
        # 现在有了约束，宽度变成了 range[i]。
        # 占比 Ratio = range[i] / 1.0 = range[i]
        # 不确定性 = Ratio
        # 确定性 = 1 - Ratio
        
        certainty_arr = 1.0 - ranges
        
        # 稍微修饰一下数值，避免极端情况 (因为数值算法可能有误差)
        certainty_arr = np.clip(certainty_arr, 0.0, 1.0)
        
        # 这是一个极其严格的指标，往往分会很低，为了可视化好看，可以做一个映射
        # (可选) certainty_arr = certainty_arr ** 0.5 
        
        return {'votes': best_votes, 'certainty': certainty_arr, 'success': True}

    def _solve_rank_mc(self):
        """
        Rank 制还是保留蒙特卡洛，因为它是离散的，无法用梯度下降求边界。
        但我们把 Certainty 的计算逻辑也改成 'Range-like' 的思路。
        """
        if self.elim_idx == -1: return self._return_uniform()
        
        num_samples = 3000
        alpha_base = np.ones(self.n)
        if self.trust_factor > 0:
            inv_ranks = self.n - self.judge_ranks
            alpha_base += self.trust_factor * (inv_ranks + 1)
            
        samples = np.random.dirichlet(alpha_base, num_samples)
        valid_samples = []
        
        for x in samples:
            fan_ranks = np.argsort(np.argsort(-x)) + 1
            total_ranks = self.judge_ranks + fan_ranks
            sorted_sums = np.sort(total_ranks)
            elim_rank_sum = total_ranks[self.elim_idx]
            
            is_valid = False
            if self.season_num <= 2:
                if elim_rank_sum == sorted_sums[-1]: is_valid = True
            else: 
                if len(sorted_sums) >= 2:
                    if elim_rank_sum >= sorted_sums[-2]: is_valid = True
                else: is_valid = True 
            
            if is_valid:
                valid_samples.append(x)
        
        if len(valid_samples) > 10:
            valid_samples = np.array(valid_samples)
            avg_votes = np.mean(valid_samples, axis=0)
            
            # 🔥 模拟 'Range' 的概念
            # 计算每个维度的 Max - Min，作为“采样到的范围”
            mins = np.min(valid_samples, axis=0)
            maxs = np.max(valid_samples, axis=0)
            ranges = maxs - mins
            
            certainty_arr = 1.0 - ranges
            certainty_arr = np.clip(certainty_arr, 0.1, 0.95)
            
            return {'votes': avg_votes, 'certainty': certainty_arr, 'success': True}
        else:
            return self._return_uniform(success=False)

    def _return_uniform(self, success=True):
        return {'votes': np.ones(self.n)/self.n, 'certainty': np.zeros(self.n), 'success': success}

# ==========================================
# 💾 结果保存
# ==========================================
def save_results(results_list, output_folder, file_name):
    if not results_list: return
    df = pd.DataFrame(results_list)
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_path, 'output', output_folder)
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    df.to_csv(os.path.join(output_dir, file_name), index=False)
    print(f"✅ 成功保存: {os.path.join(output_dir, file_name)}")