import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import (
    WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter,
    GeneralizedGammaRegressionFitter,
)


def preprocess(df):
    """Encode the dataset for AFT models."""
    data = df.copy()
    data["churn"] = (data["churn"].astype(str).str.lower() == "yes").astype(int)
    for col in ["retire", "voice", "internet", "forward"]:
        data[col] = (data[col].astype(str).str.lower() == "yes").astype(int)
    data["gender"] = (data["gender"] == "Male").astype(int)
    data["marital"] = (data["marital"] == "Married").astype(int)

    ed_order = {
        "Did not complete high school": 1, "High school degree": 2,
        "Some college": 3, "College degree": 4,
        "Post-undergraduate degree": 5,
    }
    data["ed"] = data["ed"].map(ed_order)
    data = pd.get_dummies(data, columns=["region", "custcat"], drop_first=True)
    data = data.drop(columns=["ID"])
    data = data[data["tenure"] > 0].copy()
    for c in data.columns:
        data[c] = data[c].astype(float)
    return data


if __name__ == "__main__":
    df = pd.read_csv("data/telco.csv")
    data = preprocess(df)

    fitters = {
        "Weibull": WeibullAFTFitter(),
        "LogNormal": LogNormalAFTFitter(),
        "LogLogistic": LogLogisticAFTFitter(),
        "GeneralizedGamma": GeneralizedGammaRegressionFitter(penalizer=0.01),
    }

    results, fitted = [], {}
    for name, m in fitters.items():
        try:
            m.fit(data, duration_col="tenure", event_col="churn")
            results.append({
                "model": name,
                "AIC": m.AIC_,
                "log_likelihood": m.log_likelihood_,
                "concordance": m.concordance_index_,
            })
            fitted[name] = m
        except Exception as e:
            print(f"{name} failed: {str(e)[:120]}")

    cmp = pd.DataFrame(results).sort_values("AIC").reset_index(drop=True)
    print("\nAFT model comparison")
    print(cmp.round(3))

    plt.figure(figsize=(9, 6))
    times = np.linspace(1, data["tenure"].max(), 200)
    mean_row = data.drop(columns=["tenure", "churn"]).mean().to_frame().T
    for name, m in fitted.items():
        sf = m.predict_survival_function(mean_row, times=times)
        plt.plot(times, sf.values.ravel(), label=name, lw=2)
    plt.title("AFT survival curves. All distributions (at mean covariates)")
    plt.xlabel("Tenure (months)")
    plt.ylabel("S(t)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("aft_all_curves.png", dpi=110)
    plt.show()
    print("saved aft_all_curves.png")
    print("\nBest by AIC:", cmp.iloc[0]["model"])