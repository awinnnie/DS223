# Telco Churn | Survival Analysis & CLV

Survival analysis of telecom subscribers using parametric AFT models. The
final model is a **LogNormal AFT** that predicts per-customer survival,
which is then used to compute Customer Lifetime Value (CLV) and recommend
an annual retention budget.

## Structure
```
churn-survival/
├── data/telco.csv              # input dataset
├── src/
│   ├── 01_eda_km.py            # EDA + Kaplan–Meier baseline
│   ├── 02_aft_models.py        # fit all AFT distributions, compare
│   ├── 03_final_model.py       # significant features, final LogNormal
│   └── 04_clv_budget.py        # CLV, segments, retention budget
├── notebook.ipynb              # narrative report with code + plots
├── requirements.txt
└── README.md
```

## How to run
```bash
pip install -r requirements.txt
python src/01_eda_km.py
python src/02_aft_models.py
python src/03_final_model.py
python src/04_clv_budget.py
```

Or open `notebook.ipynb` for the full report with narrative.

## Author
Anna Khurshudyan 