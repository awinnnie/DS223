import pandas as pd
import matplotlib.pyplot as plt
import os
from lifelines import KaplanMeierFitter

df = pd.read_csv("data/telco.csv")
print("shape:", df.shape)
print(df.head())
print("\nchurn rate:", (df["churn"].str.lower() == "yes").mean().round(3))
print("median tenure:", df["tenure"].median())

# convert event for KM
event = (df["churn"].astype(str).str.lower() == "yes").astype(int)

kmf = KaplanMeierFitter()
kmf.fit(df["tenure"], event, label="All subscribers")

ax = kmf.plot_survival_function()
ax.set_title("Kaplan–Meier baseline")
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("S(t)")
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/km_baseline.png", dpi=110)
plt.show()
print("saved km_baseline.png")