import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

try:
    # 引入新写的清理函数
    from utils import clean_previous_outputs  
    
    from q1_inference import run_baseline as run_q1_base
    from q1_inference_plus import run_improved as run_q1_plus
    from q2_comparison import run_q2_baseline
    from q2_comparison_plus import run_q2_improved
    from q3_analysis import run_q3_baseline
    from q3_analysis_plus import run_q3_improved
    from q4_optimization import run_q4_baseline
    from q4_optimization_plus import run_q4_improved
    from visualization import main as run_viz
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

def main():
    print("==========================================")
    print("   MCM 2026 Problem C - 全流程自动化系统")
    print("==========================================\n")
    
    start_time = time.time()

    # 🔥 Step 0: 启动前先打扫卫生
    clean_previous_outputs()

    # --- Step 1: Q1 Inverse Inference ---
    print("\n📦 [Step 1] Q1: Inverse Inference")
    run_q1_base()
    run_q1_plus()

    # --- Step 2: Q2 System Comparison ---
    print("\n⚔️ [Step 2] Q2: System Comparison")
    run_q2_baseline()
    run_q2_improved()

    # --- Step 3: Q3 Factor Analysis ---
    print("\n🔍 [Step 3] Q3: Factor Analysis")
    run_q3_baseline()
    run_q3_improved()

    # --- Step 4: Q4 Optimization ---
    print("\n⚖️ [Step 4] Q4: Optimization Strategy")
    run_q4_baseline()
    run_q4_improved()
    
    # --- Step 5: Final Rendering ---
    print("\n🎨 [Step 5] Final Rendering")
    try:
        run_viz()
    except Exception as e:
        print(f"   ⚠️ 绘图故障: {e}")

    print(f"\n🎉 全流程完毕！总耗时: {time.time() - start_time:.2f} 秒")

if __name__ == "__main__":
    main()