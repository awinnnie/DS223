import warnings
warnings.filterwarnings("ignore")

import pickle
import pandas as pd
from lifelines import LogNormalAFTFitter

import sys, os
sys.path.append(os.path.dirname(__file__))
from importlib import import_module
preprocess = import_module("02_aft_models").preprocess

df = pd.read_csv("data/telco.csv")
data = preprocess(df)

# 1) full LogNormal model
full = LogNormalAFTFitter()
full.fit(data, duration_col="tenure", event_col="churn")
print("Full LogNormal AFT")
print(full.summary[["coef", "exp(coef)", "p"]].round(4))

# 2) keep significant features (p < 0.05) on mu_
mu = full.summary.loc["mu_"]
keep = mu[(mu["p"] < 0.05) & (mu.index != "Intercept")].index.tolist()
print(f"\nSignificant features kept ({len(keep)}): {keep}")

# 3) refit with only kept features
final = LogNormalAFTFitter()
final.fit(data[keep + ["tenure", "churn"]],
          duration_col="tenure", event_col="churn")

print("\nFinal LogNormal AFT (significant features only)")
print(final.summary[["coef", "exp(coef)", "p"]].round(4))
print(f"\nAIC: {final.AIC_:.2f}   concordance: {final.concordance_index_:.3f}")

# save model + feature list
with open("outputs/final_model.pkl", "wb") as f:
    pickle.dump({"model": final, "features": keep}, f)
print("\nsaved final_model.pkl")