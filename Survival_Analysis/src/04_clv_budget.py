import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(__file__))
from importlib import import_module
preprocess = import_module("02_aft_models").preprocess

with open("final_model.pkl", "rb") as f:
    bundle = pickle.load(f)
model, features = bundle["model"], bundle["features"]

df = pd.read_csv("data/telco.csv")
data = preprocess(df)

HORIZON = 60
ANNUAL_DISCOUNT = 0.10
MARGIN_RATE = 0.20
r = (1 + ANNUAL_DISCOUNT) ** (1/12) - 1

monthly_margin = data["income"] * 1000 / 12 * MARGIN_RATE
times = np.arange(1, HORIZON + 1)
sf = model.predict_survival_function(data[features], times=times)
disc = 1 / (1 + r) ** (times - 1)
expected_months = (sf.values.T * disc).sum(axis=1)
clv = monthly_margin.values * expected_months

data["CLV"] = clv
data["churn_prob_12m"] = 1 - sf.loc[12].values

print("CLV summary")
print(data["CLV"].describe().round(2))

# segment analysis on the original df
seg_df = df.loc[data.index].copy()
seg_df["CLV"] = data["CLV"].values
seg_df["churn_prob_12m"] = data["churn_prob_12m"].values
seg_df["age_band"] = pd.cut(seg_df["age"], [0, 30, 45, 60, 100],
                            labels=["<30", "30-45", "45-60", "60+"])

print("\nCLV by segment")
for col in ["custcat", "region", "marital", "ed", "internet", "voice", "age_band"]:
    print(f"\n[{col}]")
    print(seg_df.groupby(col, observed=True)["CLV"]
          .agg(["mean", "median", "count"]).round(2))

seg_df["value_score"] = seg_df["CLV"] * (1 - seg_df["churn_prob_12m"])
print("\n[Most valuable customer categories by value_score]")
print(seg_df.groupby("custcat")["value_score"]
      .agg(["mean", "count"]).sort_values("mean", ascending=False).round(2))

# retention budget
print("\nRetention budget")
print(f"{'thr':>5} | {'at_risk':>7} | {'% base':>7} | "
      f"{'expected_loss':>14} | {'budget@30%':>11}")
print("-" * 60)
for thr in [0.10, 0.20, 0.30, 0.40, 0.50]:
    ar = seg_df[seg_df["churn_prob_12m"] > thr]
    loss = (ar["churn_prob_12m"] * ar["CLV"]).sum()
    print(f"{thr:>5.2f} | {len(ar):>7} | {len(ar)/len(seg_df):>6.1%} | "
          f"${loss:>12,.0f} | ${loss*0.30:>9,.0f}")

# plot
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(seg_df["CLV"], bins=40, edgecolor="black")
axes[0].set_title("CLV distribution")
axes[0].set_xlabel("CLV ($)")
axes[0].axvline(seg_df["CLV"].median(), color="pink", ls="--", label="median")
axes[0].legend()

mc = seg_df.groupby("custcat")["CLV"].mean().sort_values()
axes[1].barh(mc.index.astype(str), mc.values)
axes[1].set_title("Mean CLV by customer category")
axes[1].set_xlabel("Mean CLV ($)")
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/clv_distribution.png", dpi=110)
plt.show()
print("\nsaved clv_distribution.png")

# save per-customer CLV
seg_df[["ID", "tenure", "churn", "CLV", "churn_prob_12m"]].to_csv(
    "outputs/clv_per_customer.csv", index=False)
print("saved outputs/clv_per_customer.csv")