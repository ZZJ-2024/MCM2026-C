import os
import sys
import time
import shutil

# --- 1. 环境初始化 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')

# 🔥 关键修复：确保 sys.path 里有 src 目录
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# ==========================================
# 🧹 模块 0: 幽灵缓存清理 (Ghost Buster)
# ==========================================
def force_clean_pycache():
    """
    强制删除所有 __pycache__ 文件夹，确保 Python 读取最新代码。
    """
    print("🧹 [System] 正在执行前置清理 (删除 __pycache__)...")
    if os.path.exists(src_path):
        for dirpath, dirnames, filenames in os.walk(src_path):
            if '__pycache__' in dirnames:
                try: shutil.rmtree(os.path.join(dirpath, '__pycache__'))
                except: pass
    print("   ✨ 缓存已清理，准备运行。\n")

force_clean_pycache()

# ==========================================
# 📦 模块 2: 模块导入
# ==========================================
print("📦 [System] 正在加载模块...")

try:
    from src.utils import clean_previous_outputs
    from src.q1_baseline import run_baseline as run_q1_base
    from src.q1_improved import run_improved as run_q1_plus
    from src.q2_comparison import run_q2_baseline
    from src.q2_comparison_plus import run_q2_improved
    from src.q3_analysis import run_q3_baseline
    from src.q3_analysis_plus import run_q3_improved
    from src.q4_optimization import run_q4_baseline
    from src.q4_optimization_plus import run_q4_improved
    from src.visualization import main as run_visualization
    print("✅ 所有模块加载成功！")

except ImportError as e:
    print(f"\n❌ 模块导入失败: {e}")
    sys.exit(1)


# ==========================================
# 🔫 模块 3: 废图狙击手 (精准删除)
# ==========================================
def sniper_delete_bad_plots():
    """
    运行结束后执行：
    1. 保护：'1_global_correlation_scan.png' (正确的图)
    2. 击毙：'1_baseline_heatmap.png' 等旧图
    """
    print("\n🔫 [Sniper] 正在执行最终清理...")
    
    # 目标文件夹
    q3_dir = os.path.join(current_dir, 'output', 'output_q3')
    
    # 💀 黑名单：所有你不想见到的旧图名字
    blacklist = [
        '1_baseline_heatmap.png',        # 最常见的旧图
        '1_Final_Static_Heatmap.png',    # 调试用的名字
        'correlation_heatmap.png',       # visualization 可能生成的旧图
        'heatmap.png',
        'raw_heatmap.png'
    ]
    
    deleted_count = 0
    if os.path.exists(q3_dir):
        for filename in os.listdir(q3_dir):
            file_path = os.path.join(q3_dir, filename)
            
            # 如果文件名在黑名单里，或者是以 'heatmap' 结尾但不是我们想要的那个
            if filename in blacklist or ('heatmap' in filename.lower() and 'global' not in filename.lower()):
                try:
                    os.remove(file_path)
                    print(f"   💥 已击毙废图: {filename}")
                    deleted_count += 1
                except: pass

    if deleted_count == 0:
        print("   ✅ 扫描完毕，目录很干净。")
    else:
        print(f"   🧹 清理完毕，共删除了 {deleted_count} 张废图。")
        print("   🛡️ 已保留: 1_global_correlation_scan.png")


# ==========================================
# ▶️ 模块 4: 主流程流水线
# ==========================================
def main():
    print("\n==========================================")
    print(" 🏆 MCM 2026 Problem C - 主程序")
    print("==========================================\n")

    start_time = time.time()

    # Step 0: 环境清理
    # 跳过 output_q3 的清理，防止误删，交给 Sniper 处理
    dirs_to_clean = ['output_q1', 'output_q1_plus', 'output_q2', 'output_q2_plus', 'output_q3_plus', 'output_q4']
    try: clean_previous_outputs(dirs_to_clean)
    except: pass

    # Step 1: Q1
    print("\n--- Phase 1: Q1 ---")
    run_q1_base()
    run_q1_plus()

    # Step 2: Q2
    print("\n--- Phase 2: Q2 ---")
    run_q2_baseline()
    run_q2_improved()

    # Step 3: Q3
    print("\n--- Phase 3: Q3 ---")
    run_q3_baseline()  # 生成 1_global_correlation_scan.png
    run_q3_improved()  # 生成 SHAP 图

    # Step 4: Q4
    print("\n--- Phase 4: Q4 ---")
    run_q4_baseline()
    run_q4_improved()

    # Step 5: Visualization
    print("\n--- Phase 5: Visualization ---")
    run_visualization()

    # 🔥 最后一步：执行狙击，确保废图消失
    sniper_delete_bad_plots()

    end_time = time.time()
    print("\n==========================================")
    print(f"⏱️ 总耗时: {(end_time - start_time) / 60:.2f} 分钟")
    print("==========================================")

if __name__ == "__main__":
    main()