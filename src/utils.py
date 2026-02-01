import pandas as pd
import numpy as np
import re
import os
import shutil
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 🧹 辅助功能：清理旧输出 (修复 ImportError)
# ==========================================
def clean_previous_outputs(dirs_to_clean):
    """
    清理指定的输出目录，防止旧文件干扰
    """
    print("🧹 [System] 正在清理旧的输出文件...")
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    for dir_name in dirs_to_clean:
        dir_path = os.path.join(base_path, 'output', dir_name)
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path) # 递归删除
                # print(f"   - 已删除: {dir_name}")
            except Exception as e:
                print(f"   ⚠️ 无法删除 {dir_name}: {e}")
    print("✨ 清理完毕！环境已重置。")

# ==========================================
# 🧹 数据预处理类 (DataPreprocessor)
# ==========================================
class DataPreprocessor:
    def __init__(self, file_path):
        """
        初始化：只接收文件路径 (修复 TypeError)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到数据文件: {file_path}")
        self.raw_df = pd.read_csv(file_path)
    
    def get_seasons(self):
        """获取所有赛季列表"""
        return sorted(self.raw_df['season'].unique())
    
    def get_weekly_data(self, season, week):
        """
        获取指定赛季、指定周次的完整数据
        返回 DataFrame: ['Contestant', 'Judge Score', 'Result', 'Placement']
        """
        season_df = self.raw_df[self.raw_df['season'] == season]
        if season_df.empty:
            return pd.DataFrame()

        # 构造周次列名，例如 "week1_judge1_score"
        judge_cols = [c for c in season_df.columns if f'week{week}_judge' in c and 'score' in c]
        
        if not judge_cols:
            return pd.DataFrame()

        records = []
        for _, row in season_df.iterrows():
            scores = []
            for c in judge_cols:
                val = row[c]
                if pd.notna(val):
                    scores.append(float(val))
            
            if not scores: continue
            
            total_score = sum(scores)
            result_str = str(row['results'])
            
            # 判断是否在本周淘汰
            is_eliminated_this_week = False
            if 'Eliminated' in result_str:
                try:
                    elim_week_num = int(re.search(r'Week\s*(\d+)', result_str, re.IGNORECASE).group(1))
                    if elim_week_num == week:
                        is_eliminated_this_week = True
                except:
                    pass
            
            final_result_tag = "Eliminated" if is_eliminated_this_week else "Safe"
            
            records.append({
                'Contestant': row['celebrity_name'],
                'Judge Score': total_score,
                'Result': final_result_tag,
                'Placement': row['placement']
            })
            
        return pd.DataFrame(records)

# ==========================================
# 🔥 核心反演算法类 (VotingInverter)
# ==========================================
class VotingInverter:
    def __init__(self, week_df, season_num, trust_factor=0.0):
        self.data = week_df.copy()
        self.season_num = season_num
        self.trust_factor = trust_factor 
        
        self.contestants = self.data['Contestant'].values
        self.n = len(self.contestants)
        
        # 找到淘汰者
        elim_rows = self.data[self.data['Result'] == 'Eliminated']
        if elim_rows.empty:
            self.elim_idx = -1 
        else:
            elim_name = elim_rows.iloc[0]['Contestant']
            indices = np.where(self.contestants == elim_name)[0]
            self.elim_idx = indices[0] if len(indices) > 0 else -1
            
        # 评委分处理
        raw_scores = self.data['Judge Score'].values
        self.judge_sum = raw_scores.sum()
        self.judge_percent = raw_scores / self.judge_sum if self.judge_sum > 0 else np.ones(self.n)/self.n
        # 排名: 分高(Rank数值小)
        self.judge_ranks = np.argsort(np.argsort(-raw_scores)) + 1

    def solve(self):
        # 策略选择: S1-2, S28+ 使用 Rank; S3-27 使用 Percent
        if self.season_num <= 2 or self.season_num >= 28:
            return self._solve_rank_mc()
        else:
            return self._solve_percent_opt()

    def _solve_percent_opt(self):
        if self.elim_idx == -1: return self._return_uniform()

        prior = (1 - self.trust_factor) * (np.ones(self.n)/self.n) + \
                self.trust_factor * self.judge_percent
        
        def objective(x):
            return np.sum((x - prior)**2)

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        epsilon = 1e-4
        
        for i in range(self.n):
            if i == self.elim_idx: continue
            def constr(x, idx=i, elim=self.elim_idx):
                # 约束: (J_i + x_i) - (J_elim + x_elim) > 0
                diff = (self.judge_percent[idx] + x[idx]) - (self.judge_percent[elim] + x[elim])
                return diff - epsilon
            constraints.append({'type': 'ineq', 'fun': constr})

        x0 = prior 
        bounds = [(0, 1) for _ in range(self.n)]
        
        try:
            res = minimize(objective, x0, bounds=bounds, constraints=constraints, method='SLSQP')
            success = res.success
            votes = res.x
        except:
            success = False
            votes = x0

        # Certainty 计算
        current_scores = self.judge_percent + votes
        elim_score = current_scores[self.elim_idx]
        survivor_scores = np.delete(current_scores, self.elim_idx)
        
        if len(survivor_scores) > 0:
            min_survivor_score = np.min(survivor_scores)
            actual_margin = min_survivor_score - elim_score
            certainty_score = 1.0 / (1.0 + 10 * max(0, actual_margin))
        else:
            certainty_score = 0.0
        
        certainty_arr = np.ones(self.n) * certainty_score
        if self.elim_idx >= 0:
            certainty_arr[self.elim_idx] = min(1.0, certainty_score * 1.2)

        return {'votes': votes, 'certainty': certainty_arr, 'success': success}

    def _solve_rank_mc(self):
        if self.elim_idx == -1: return self._return_uniform()
        
        num_samples = 2000
        alpha_base = np.ones(self.n)
        if self.trust_factor > 0:
            inv_ranks = self.n - self.judge_ranks
            alpha_base += self.trust_factor * (inv_ranks + 1)
            
        samples = np.random.dirichlet(alpha_base, num_samples)
        valid_samples = []
        
        for x in samples:
            fan_ranks = np.argsort(np.argsort(-x)) + 1
            total_ranks = self.judge_ranks + fan_ranks
            
            elim_rank_sum = total_ranks[self.elim_idx]
            sorted_sums = np.sort(total_ranks)
            
            is_valid = False
            if self.season_num <= 2:
                if elim_rank_sum == sorted_sums[-1]: is_valid = True
            else:
                if len(sorted_sums) >= 2:
                    if elim_rank_sum >= sorted_sums[-2]: is_valid = True
                else:
                    is_valid = True 
            
            if is_valid:
                valid_samples.append(x)
        
        if len(valid_samples) > 10:
            valid_samples = np.array(valid_samples)
            avg_votes = np.mean(valid_samples, axis=0)
            stds = np.std(valid_samples, axis=0)
            
            certainty_arr = 1.0 - (stds / 0.25)
            certainty_arr = np.clip(certainty_arr, 0.1, 0.95)
            
            return {'votes': avg_votes, 'certainty': certainty_arr, 'success': True}
        else:
            return self._return_uniform(success=False)

    def _return_uniform(self, success=True):
        return {
            'votes': np.ones(self.n)/self.n, 
            'certainty': np.zeros(self.n), 
            'success': success
        }

# ==========================================
# 💾 结果保存工具
# ==========================================
def save_results(results_list, output_folder, file_name):
    if not results_list:
        print("⚠️ 结果为空，未保存 CSV。")
        return
        
    df = pd.DataFrame(results_list)
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_path, 'output', output_folder)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_path = os.path.join(output_dir, file_name)
    df.to_csv(save_path, index=False)
    print(f"✅ 成功保存: {save_path}")