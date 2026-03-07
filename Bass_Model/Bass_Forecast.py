import numpy as np
import pandas as pd

p = 0.0118 
q = 0.0522  
M = 500    

def bass_cumulative(t, p, q, M):
    """
    Computes cumulative Bass model adoption at time t.

    The Bass model assumes adoption comes from two groups:
      - Innovators: adopt at rate p regardless of others
      - Imitators: adopt at rate q proportional to existing adopters

    Parameters:
        t (array): time periods (quarters, starting at 1)
        p (float): coefficient of innovation
        q (float): coefficient of imitation
        M (float): total market potential (thousands of units)

    Returns:
        array: cumulative adopters at each time period t
    """
    return M * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))


def build_forecast(p, q, M, n_quarters=60, start_year=2025):
    """
    Builds a quarterly adoption forecast using the Bass model.

    Computes both cumulative and new adopters per quarter,
    then returns an annual summary (Q4 snapshot of each year).

    Parameters:
        p (float): coefficient of innovation
        q (float): coefficient of imitation
        M (float): market potential in thousands of units
        n_quarters (int): number of quarters to forecast (default: 60 = 15 years)
        start_year (int): first year of forecast (default: 2025)

    Returns:
        pd.DataFrame: annual summary with new adopters, cumulative adopters,
                      and market penetration % per year
    """
    t = np.arange(1, n_quarters + 1)

    cum = bass_cumulative(t, p, q, M)

    new = np.diff(np.concatenate([[0], cum]))

    labels = [f"Q{((i) % 4) + 1} {start_year + i // 4}" for i in range(n_quarters)]

    df = pd.DataFrame({
        'Quarter':           labels,
        'New Adopters':      (new * 1000).astype(int), # converting from thousands
        'Cumulative':        (cum * 1000).astype(int),
        'Penetration (%)':   (cum / M * 100).round(1)
    })

    return df.iloc[3::4].copy()

forecast = build_forecast(p, q, M)

print("DGX Spark Adoption Forecast (Annual Summary)")
print(f"Parameters: p={p}, q={q}, M={M*1000:,} units\n")
print(forecast.to_string(index=False))

t_all = np.arange(1, 61)
cum_all = bass_cumulative(t_all, p, q, M)
new_all = np.diff(np.concatenate([[0], cum_all]))
labels_all = [f"Q{(i % 4) + 1} {2025 + i // 4}" for i in range(60)]

peak_idx = np.argmax(new_all)
half_idx = np.argmax(cum_all >= M * 0.5)

print(f"\nPeak adoption:      {labels_all[peak_idx]} ({int(new_all[peak_idx]*1000):,} units/quarter)")
print(f"50% penetration:    {labels_all[half_idx]}")