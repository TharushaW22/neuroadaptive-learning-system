from evaluation.benchmark import run as bench
from evaluation.stats import paired_test
import pandas as pd

bench()
df = pd.read_csv("results/benchmark_raw.csv")
a = df[df.algorithm == "linucb"]["mean_gain"].values
b = df[df.algorithm == "fixed_rules"]["mean_gain"].values
print(paired_test(a, b))