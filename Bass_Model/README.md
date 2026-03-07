# Bass Diffusion Model: Nvidia DGX Spark
**Technology Diffusion Analysis · 2025**

---

## Innovation Selected

**Nvidia DGX Spark** — Desktop AI Supercomputer  
TIME Best Inventions 2025: https://time.com/collections/best-inventions-2025/7318247/nvidia-dgx-spark/

The DGX Spark packs 1 petaflop of AI compute and 128 GB of unified memory into a desktop
device powered by the Grace Blackwell GB10 Superchip. It brings cluster-level AI compute
to individual desks for the first time, at a price of $3,999.

---

## Project Summary

This project applies the Bass diffusion model to predict the adoption trajectory of the
Nvidia DGX Spark, using global professional workstation shipments (Q3 2008 - Q3 2019) as the
historical look-alike innovation.

**Key Results:**

| Parameter | Value |
|-----------|-------|
| p (coefficient of innovation) | 0.0118 (quarterly) |
| q (coefficient of imitation) | 0.0522 (quarterly) |
| M (market potential) | 500,000 units |
| Peak adoption | Q4 2030 (~9,806 units/quarter) |
| 50% market penetration | Q2 2032 |

---

## Repository Structure

```
Bass_Model/
├── README.md                          # This file
├── Report.ipynb                       # Main Jupyter notebook - all 7 steps
├── Bass_Forecast.py                   # Standalone Bass model forecast script
├── Fermi_Estimation.py                # Standalone Fermi estimation of market potential M
├── requirements.txt                   # Python dependencies
│
├── data/
│   ├── statistic_id268429_(...).xlsx  # Raw workstation shipment data (Statista)
│   └── workstation_shipments.csv      # Cleaned quarterly data used for model fitting
│
├── img/
│   ├── fig1_historical_fit.png        # Workstation shipments vs Bass model fit
│   ├── fig2_dgx_forecast.png          # DGX Spark new and cumulative adopters forecast
│   └── fig3_penetration.png           # DGX Spark market penetration S-curve
│
└── report/
    ├── report.pdf                     # Final report 
    └── report.md                      # Markdown source of the report
```

---

## How to Run

### Setup
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Run the Notebook
```bash
jupyter lab
# then open Report.ipynb
```

### Run Scripts Standalone
```bash
# Fermi estimation of market potential M
python Fermi_Estimation.py

# Bass model forecast and adoption table
python Bass_Forecast.py
```

### Regenerate PDF Report
```bash
python -m jupyter nbconvert --to pdf --no-input Report.ipynb --output report/report.pdf
```

---

## Data Source

| Dataset | Period | Source | URL |
|---------|--------|--------|-----|
| Global Workstation Shipments (Quarterly) | Q3 2008 – Q3 2019 | Jon Peddie Research via Statista | https://www.statista.com/statistics/268429/workstation-shipments-worldwide-since-the-3rd-quarter-2008/ |

> Access requires university library subscription (available via AUA Wi-Fi).

---

## Methodology

1. **Look-alike:** Professional workstation computers (SGI, Sun, Dell, HP), same democratization pattern as DGX Spark
2. **Data:** Quarterly workstation shipments used to fit Bass model parameters
3. **Parameter fitting:** Nonlinear least squares via `scipy.optimize.curve_fit`, with M constrained to realistic range
4. **M estimation:** Fermi logic based on global ML developer population
5. **Forecast:** Bass model applied to DGX Spark with transferred p, q and Fermi-estimated M

---

## References

1. Bass, F.M. (1969). A new product growth for model consumer durables. *Management Science*, 15(5), 215–227.
2. Sultan, F., Farley, J.U., & Lehmann, D.R. (1990). A meta-analysis of applications of diffusion models. *Journal of Marketing Research*, 27(1), 70–77.
3. Jon Peddie Research / Statista (2021). Workstation shipments worldwide 2008–2019. https://www.statista.com/statistics/268429/
4. Stack Overflow (2023). Developer Survey 2023. https://survey.stackoverflow.co/2023/
5. Nvidia Corporation (2023). Annual Report 2023. https://www.nvidia.com/en-us/about-nvidia/corporate-sustainability/annual-report/
6. Grand View Research (2023). AI Accelerator Market Size Report. https://www.grandviewresearch.com/industry-analysis/ai-accelerator-market
7. TIME Magazine (2025). Nvidia DGX Spark. https://time.com/collections/best-inventions-2025/7318247/nvidia-dgx-spark/
8. Evans Data Corporation (2023). Global Developer Population and Demographic Study. https://evansdata.com/reports/viewRelease.php?reportID=9
