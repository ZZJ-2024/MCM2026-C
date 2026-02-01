import os
import sys
import time

# --- 1. 环境初始化 ---
# 将 src 目录加入系统路径，确保能找到所有模块
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')

# 🔥 关键修复：确保 sys.path 里有 src 目录
if src_path not in sys.path:
    sys.path.append(src_path)

# --- 2. 模块导入 (带容错处理) ---
print("📦 [System] 正在加载模块...")

try:
    # 导入工具
    from src.utils import clean_previous_outputs
    
    # 导入 Q1 (数据重构)
    # 🔥 关键修复：分别从两个不同的文件导入对应的函数
    from src.q1_baseline import run_baseline as run_q1_base
    from src.q1_improved import run_improved as run_q1_plus
    
    # 导入 Q2 (赛制对比)
    from src.q2_comparison import run_q2_baseline
    from src.q2_comparison_plus import run_q2_improved
    
    # 导入 Q3 (因素分析)
    from src.q3_analysis import run_q3_baseline
    from src.q3_analysis_plus import run_q3_improved
    
    # 导入 Q4 (策略优化)
    from src.q4_optimization import run_q4_baseline
    from src.q4_optimization_plus import run_q4_improved
    
    # 导入 绘图模块
    from src.visualization import main as run_visualization
    
    print("✅ 所有模块加载成功！")

except ImportError as e:
    print(f"\n❌ 严重错误: 模块导入失败。")
    print(f"详细错误: {e}")
    print("\n🔍 排查指南:")
    print("1. 请检查 src 目录下是否有 q1_baseline.py, q1_improved.py 等文件。")
    print("2. 请检查这些文件里是否定义了 run_baseline, run_improved 等函数。")
    sys.exit(1)


# --- 3. 主流程流水线 ---
def main():
    print("\n==========================================")
    print(" 🏆 MCM 2026 Problem C - 全流程自动化系统")
    print("==========================================\n")

    start_time = time.time()

    # ------------------------------------------------
    # Step 0: 环境清理
    # ------------------------------------------------
    dirs_to_clean = [
        'output_q1', 'output_q1_plus', 
        'output_q2', 'output_q2_plus',
        'output_q3', 'output_q3_plus',
        'output_q4', 'output_q4_plus'
    ]
    try:
        clean_previous_outputs(dirs_to_clean)
    except Exception as e:
        print(f"⚠️ 清理部分跳过: {e}")

    # ------------------------------------------------
    # Step 1: Q1 反向推断 (数据地基)
    # ------------------------------------------------
    print("\n" + "="*40)
    print("🧩 [Phase 1] Q1: Inverse Inference (数据重构)")
    print("="*40)
    
    print("\n--- 1.1 Running Baseline Model ---")
    run_q1_base()
    
    print("\n--- 1.2 Running Improved Model (Data Source for Q2-Q4) ---")
    run_q1_plus()

    # ------------------------------------------------
    # Step 2: Q2 赛制对比 (模型验证)
    # ------------------------------------------------
    print("\n" + "="*40)
    print("⚖️ [Phase 2] Q2: System Comparison (赛制对比)")
    print("="*40)
    
    print("\n--- 2.1 Rank vs Percent Comparison ---")
    run_q2_baseline()
    
    print("\n--- 2.2 Judge's Save Simulation ---")
    run_q2_improved()

    # ------------------------------------------------
    # Step 3: Q3 因素分析 (特征工程)
    # ------------------------------------------------
    print("\n" + "="*40)
    print("🔍 [Phase 3] Q3: Factor Analysis (归因分析)")
    print("="*40)
    
    print("\n--- 3.1 Basic Correlation ---")
    run_q3_baseline()
    
    print("\n--- 3.2 Random Forest Importance ---")
    run_q3_improved()

    # ------------------------------------------------
    # Step 4: Q4 策略优化 (模型应用)
    # ------------------------------------------------
    print("\n" + "="*40)
    print("🎯 [Phase 4] Q4: Strategy Optimization (策略设计)")
    print("="*40)
    
    print("\n--- 4.1 Fixed Weights Strategy ---")
    run_q4_baseline()
    
    print("\n--- 4.2 Adaptive Dynamic Weights ---")
    run_q4_improved()

    # ------------------------------------------------
    # Step 5: 高级绘图 (论文插图)
    # ------------------------------------------------
    print("\n" + "="*40)
    print("🎨 [Phase 5] Visualization (O-Prize Style Plots)")
    print("="*40)
    run_visualization()

    # ------------------------------------------------
    # 结束
    # ------------------------------------------------
    end_time = time.time()
    total_minutes = (end_time - start_time) / 60
    
    print("\n==========================================")
    print(f"🎉 任务全部完成！Mission Complete!")
    print(f"⏱️  总耗时: {total_minutes:.2f} 分钟")
    print(f"📂 结果文件: 请查看 /output 下的各个子文件夹")
    print("==========================================")

if __name__ == "__main__":
    main()