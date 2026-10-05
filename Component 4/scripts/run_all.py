import subprocess, sys

for s in ["so1_profile_demo", "so2_comparison_demo", "so3_bandits_demo",
          "so4_online_learning_demo", "so5_benchmark_demo"]:
    print(f"\n=== {s} ===")
    subprocess.run([sys.executable, f"scripts/{s}.py"], check=False)