import os
import sys
import time
import shutil

# --- 1. 环境初始化 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- 2. 模块导入 ---
print("📦 [System] Loading Modules...")
try:
    from src.utils import clean_previous_outputs
    
    # Q1
    from src.q1_baseline import run_baseline as run_q1_base
    from src.q1_improved import run_improved as run_q1_plus
    
    # Q2
    from src.q2_comparison import run_q2_baseline
    from src.q2_comparison_plus import run_q2_improved
    
    # Q3
    from src.q3_analysis import run_q3_baseline
    from src.q3_analysis_plus import run_q3_improved
    
    # Q4
    from src.q4_optimization import run_q4_strategy 
    
    # Visualization
    from src.visualization import main as run_visualization
    
    print("✅ All Modules Loaded Successfully.")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

def force_clean_start(folder_name):
    """启动前清理：删除整个文件夹"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(base_path, 'output', folder_name)
    if os.path.exists(target_dir):
        try: shutil.rmtree(target_dir)
        except: pass
    os.makedirs(target_dir, exist_ok=True)

def sniper_cleanup():
    """🔥 狙击清理：程序结束前，强制检查并删除所有不该出现的垃圾图"""
    print("\n🧹 [Final Sweep] Checking for unwanted artifacts...")
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    q4_dir = os.path.join(base_path, 'output', 'output_q4')
    
    # 这里列出所有你不想看见的文件名 (黑名单)
    blacklist = [
        'adaptive.png',                # 那个顽固的 S 型图
        '1_Dynamic_Weight_Curve.png',  # 旧版图名
        'q4_strategy_nature.png',      # 中间版本
        'q4_adaptive_simulation.csv',  # 旧数据
        'q4_baseline_comparison.csv'   # 旧数据
    ]
    
    deleted_count = 0
    if os.path.exists(q4_dir):
        for filename in os.listdir(q4_dir):
            if filename in blacklist:
                file_path = os.path.join(q4_dir, filename)
                try:
                    os.remove(file_path)
                    print(f"   🔥 Destroyed banned file: {filename}")
                    deleted_count += 1
                except Exception as e:
                    print(f"   ⚠️ Failed to delete {filename}: {e}")
    
    if deleted_count == 0:
        print("   ✅ No junk files found. Clean output confirmed.")

def main():
    print("\n==========================================")
    print(" 🏆 MCM 2026 Problem C - Full Pipeline 🏆")
    print("==========================================\n")

    start_time = time.time()

    # Step 0: 启动前清理
    print("🧹 [System] Initializing workspace...")
    folders_to_clean = ['output_q1', 'output_q1_plus', 'output_q2', 'output_q2_plus', 'output_q3', 'output_q3_plus', 'output_q4']
    for folder in folders_to_clean:
        force_clean_start(folder)

    # Step 1: Q1
    print("\n--- Phase 1: Q1 Data Reconstruction ---")
    run_q1_base()
    run_q1_plus()

    # Step 2: Q2
    print("\n--- Phase 2: Q2 System Diagnosis ---")
    run_q2_baseline()
    run_q2_improved()

    # Step 3: Q3
    print("\n--- Phase 3: Q3 Bias Attribution ---")
    run_q3_baseline()
    run_q3_improved()

    # Step 4: Q4
    print("\n--- Phase 4: Q4 Optimal Strategy (Tanh System) ---")
    run_q4_strategy()

    # Step 5: Visualization
    print("\n--- Phase 5: Generating Final Report Visuals ---")
    try: run_visualization()
    except: pass

    # 🔥 Step 6: 最终狙击清理 (Sniper Check) 🔥
    sniper_cleanup()

    end_time = time.time()
    duration = (end_time - start_time) / 60
    print("\n==========================================")
    print(f"🎉 Pipeline Complete.")
    print(f"⏱️  Total Runtime: {duration:.2f} minutes")
    print("==========================================")

if __name__ == "__main__":
    main()