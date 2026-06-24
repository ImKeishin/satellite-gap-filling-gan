import pandas as pd

metrics_path = "results_sen12mscrts/final_test_random_all_r3/metrics.csv"
df = pd.read_csv(metrics_path)

print("\nNumăr eșantioane evaluate:")
print(len(df))

print("\nTabelul 5.3 - Rezultatele cantitative medii")
print(df[["mae", "psnr", "ssim"]].mean().round(6))

print("\nTabelul 5.4 - Distribuția metricilor pe setul de testare")
table_54 = df[["mae", "psnr", "ssim"]].agg(["mean", "std", "min", "max"]).T
print(table_54.round(6))

print("\nAcoperire nori - pentru text interpretativ")
print(df[["average_input_coverage", "target_coverage", "union_coverage"]].agg(["mean", "std", "min", "max"]).T.round(6))