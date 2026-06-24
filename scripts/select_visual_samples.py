import pandas as pd

metrics_path = "results_sen12mscrts/final_test_random_all_r3/metrics.csv"
df = pd.read_csv(metrics_path)

easy = df.sort_values("union_coverage").iloc[0]
medium = df.iloc[(df["union_coverage"] - df["union_coverage"].median()).abs().argsort().iloc[0]]
hard = df.sort_values("union_coverage").iloc[-1]

for label, row in [("easy", easy), ("medium", medium), ("hard", hard)]:
    sample = int(row["sample"])
    print()
    print(label.upper())
    print(f"sample_{sample:04d}.png")
    print(f"union_coverage = {row['union_coverage']:.6f}")
    print(f"mae = {row['mae']:.6f}")
    print(f"psnr = {row['psnr']:.6f}")
    print(f"ssim = {row['ssim']:.6f}")