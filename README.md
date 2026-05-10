<div align="center">

# 📈 Time Series Forecasting

### From EDA to ARIMA to SARIMAX — A Complete Practitioner's Guide to Forecasting Sequential Data

> _Most people treat time series as "just regression with dates". This repo shows why that's wrong — and what to do instead._

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-Time_Series-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-Computation-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-ARIMA_SARIMAX-4B8BBE?style=for-the-badge&logo=python&logoColor=white)](https://www.statsmodels.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org)
[![Seaborn](https://img.shields.io/badge/Seaborn-Statistical_Plots-4C72B0?style=for-the-badge&logo=python&logoColor=white)](https://seaborn.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Metrics-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

</div>

---

## 📌 What It Does

**Time_Series_Forecasting** is a structured, notebook-first deep dive into forecasting sequential data — covering everything from understanding what makes time series unique, to running a full EDA, to building and tuning statistical models like **ARIMA** and **SARIMAX**.

This is not a one-notebook tutorial. It's a **three-stage pipeline** — understand the data → explore it properly → model it correctly — with every decision explained.

### 🎯 What's Covered

| Stage | Notebook | What You Learn |
|---|---|---|
| 🔍 **Stage 1 — Theory** | `Different_Types_of_Time_Series` | Trend, seasonality, stationarity, white noise, decomposition |
| 📊 **Stage 2 — EDA** | `Time_Series_EDA.ipynb` | Rolling stats, ACF/PACF plots, ADF test, decomposition |
| 🤖 **Stage 3 — Modeling** | `ARIMA_and_SARIMAX` | Model selection, parameter tuning, forecasting, evaluation |

---

## 🏛️ Architecture — The Forecasting Pipeline

```
  RAW TIME SERIES DATA
  (CSV / API / Database)
          │
          ▼
  ┌───────────────────────────────────────────────────────┐
  │              STAGE 1 — UNDERSTAND THE DATA            │
  │                                                       │
  │   Is it stationary?    Does it have trend?            │
  │   Does it have seasonality?   Is it noisy?            │
  │                                                       │
  │   Types:                                              │
  │   • Stationary        → mean/var constant over time   │
  │   • Trend             → long-term direction           │
  │   • Seasonal          → repeating patterns            │
  │   • Cyclical          → irregular multi-year waves    │
  │   • Irregular/Noise   → random fluctuations           │
  └───────────────────────┬───────────────────────────────┘
                          │
                          ▼
  ┌───────────────────────────────────────────────────────┐
  │              STAGE 2 — EXPLORATORY DATA ANALYSIS      │
  │                                                       │
  │   Rolling Mean & Std   →  spot non-stationarity       │
  │   ADF Test             →  confirm statistically       │
  │   Differencing         →  make it stationary          │
  │   Seasonal Decompose   →  separate trend + season     │
  │   ACF Plot             →  find q (MA order)           │
  │   PACF Plot            →  find p (AR order)           │
  │                                                       │
  └───────────────────────┬───────────────────────────────┘
                          │
                          ▼
  ┌───────────────────────────────────────────────────────┐
  │              STAGE 3 — MODELING & FORECASTING         │
  │                                                       │
  │                                                       │
  │   ARIMA(p, d, q)                                      │
  │   ├── p = AR order   (from PACF)                      │
  │   ├── d = differencing (from ADF test)                │
  │   └── q = MA order   (from ACF)                       │
  │                                                       │
  │   SARIMAX(p,d,q)(P,D,Q,m)                             │
  │   ├── Seasonal AR, MA, differencing                   │
  │   ├── m = seasonal period (12 for monthly)            │
  │   └── X = exogenous variables                         │
  │                                                       │
  │   Evaluation:                                         │
  │   MAE · RMSE · MAPE · AIC · BIC                       │
  │                                                       │
  │   Output:                                             │
  │   Forecast plot + confidence intervals                │
  └───────────────────────────────────────────────────────┘
```

### ARIMA Parameter Selection Flow

```
         Start
           │
    Run ADF Test
           │
    ┌──────┴──────┐
    │             │
Stationary    Non-Stationary
  d = 0       Difference → d = 1 (or 2)
    │             │
    └──────┬───── ┘
           │
      Plot ACF  →  cut off at lag q  →  q value
      Plot PACF →  cut off at lag p  →  p value
           │
    Fit ARIMA(p, d, q)
           │
    Check Residuals
    (should be white noise)
           │
    ┌──────┴──────┐
    │             │
White Noise    Not White Noise
  ✅ Done      Adjust p, q → Retry
           │
    Has Seasonality?
    ├── No  → ARIMA is final model
    └── Yes → Upgrade to SARIMAX(p,d,q)(P,D,Q,m)
           │
    Forecast + Plot
    Confidence Intervals
```

---

## 🖥️ Demo

> **Replace this with actual output plots** — run the notebooks and screenshot these specific outputs:

```
📸  5 screenshots that sell this repo:

  1. Time_Series_EDA.ipynb
     → Rolling mean + std plot showing non-stationarity

  2. Time_Series_EDA.ipynb
     → ACF and PACF side-by-side plots

  3. Time_Series_EDA.ipynb
     → Seasonal decomposition (trend + seasonal + residual)

  4. ARIMA_and_SARIMAX
     → ARIMA forecast vs actual with confidence interval shading

  5. ARIMA_and_SARIMAX
     → SARIMAX forecast showing seasonal pattern captured
```

**Sample Output Format** *(replace with real screenshots)*

```
┌─────────────────────────────────────────────────────────┐
│  ARIMA Forecast vs Actual                               │
│                                                         │
│  ████                              Legend:              │
│  █  ██                             ── Actual            │
│  █    ████                         -- Forecast          │
│  █        ████                     ░░ 95% CI            │
│  █            ████░░░░░            ░░░░░░░░             │
│  █                ████░░░░░░░░░░░░░░░░░░░░░             │
│  └─────────────────────────────────────────             │
│   Jan    Apr    Jul    Oct    Jan(forecast)              │
│                                                         │
│  MAE: 12.3  │  RMSE: 18.7  │  MAPE: 4.2%              │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Library | Version | Role |
|---|---|---|
| **Python** | 3.9+ | Core language |
| **Pandas** | 1.5+ | Time series indexing, resampling, rolling windows |
| **NumPy** | 1.23+ | Numerical computation |
| **Statsmodels** | 0.13+ | ARIMA, SARIMAX, ACF/PACF, ADF test, decomposition |
| **Matplotlib** | 3.6+ | Forecast plots, residual plots, decomposition charts |
| **Seaborn** | 0.12+ | Statistical visualizations |
| **scikit-learn** | 1.2+ | MAE, RMSE metric computation |
| **Jupyter** | Latest | Interactive notebook environment |

---

## ⚡ How to Run

### Prerequisites

- Python 3.9+
- pip

### 1. Clone the Repo

```bash
git clone https://github.com/Rohan171819/Time_Series_Forecasting.git
cd Time_Series_Forecasting
```

### 2. Install Dependencies

```bash
pip install pandas numpy statsmodels matplotlib seaborn scikit-learn jupyter
```

Or with a requirements file:

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter

```bash
jupyter notebook
```

### 4. Run Notebooks in This Order

```
Step 1 → Different_Types_of_Time_Series.*
         Understand what kind of series you're dealing with
         (trend / seasonal / stationary / cyclical)

Step 2 → Time_Series_EDA.ipynb
         Load your data → rolling stats → ADF test
         → decomposition → ACF/PACF plots → note p, d, q

Step 3 → ARIMA_and_SARIMAX.*
         Plug in p, d, q → fit ARIMA → check residuals
         → add seasonality → fit SARIMAX → forecast
```

### 5. Use Your Own Dataset

Replace the data loading cell in `Time_Series_EDA.ipynb`:

```python
# Replace this line with your own CSV
df = pd.read_csv("your_data.csv", parse_dates=["date_column"], index_col="date_column")

# Ensure datetime index with frequency
df = df.asfreq("MS")   # MS = Month Start; use D for daily, H for hourly
df = df.fillna(method="ffill")
```

The rest of the pipeline runs automatically on any well-formatted time series.

---

## 📂 Repository Structure

```
Time_Series_Forecasting/
│
├── Different_Types_of_Time_Series.*   # Theory: trend, seasonal, stationary
├── Time_Series_EDA.ipynb              # EDA: ADF, ACF/PACF, decomposition
├── ARIMA_and_SARIMAX.*                # Modeling: ARIMA + SARIMAX + forecast
└── README.md
```

---

## 📐 Key Concepts Explained

### Stationarity

```
Non-Stationary (bad for ARIMA)        Stationary (good for ARIMA)
─────────────────────────────         ─────────────────────────────
Mean changes over time                Mean is constant
Variance changes over time            Variance is constant
                                      No trend, no seasonality

Fix: Apply differencing (d times) until ADF test p-value < 0.05
```

### ARIMA vs SARIMAX

```
ARIMA(p, d, q)                        SARIMAX(p,d,q)(P,D,Q,m)
──────────────                        ───────────────────────────────
No seasonality handling               Handles seasonal patterns
Good for: stock prices, sales         Good for: monthly temp, retail
          economic indicators                   airline passengers
```

### How to Read ACF / PACF

```
ACF (Autocorrelation Function)        → tells you MA order (q)
  Significant spikes at lag 1, 2...  → q = last significant lag

PACF (Partial Autocorrelation)        → tells you AR order (p)
  Significant spikes at lag 1, 2...  → p = last significant lag
```

---

## 🗺️ Roadmap

- [x] Time series types and components
- [x] Exploratory Data Analysis (EDA)
- [x] ARIMA modeling and forecasting
- [x] SARIMAX with seasonal components
- [ ] Prophet (Facebook) forecasting
- [ ] LSTM-based forecasting (deep learning approach)
- [ ] Multivariate time series
- [ ] Auto ARIMA (pmdarima)

---

## 👤 Author

**Rohan Sharma** — AI/ML Engineer · LangGraph Developer  
MCA @ GL Bajaj Institute of Technology & Management | BCA @ GGSIPU — 9.5 CGPA

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/rohan-sharma-048266246)
[![Email](https://img.shields.io/badge/Email-sharma1718rohan@gmail.com-D14836?style=flat-square&logo=gmail)](mailto:sharma1718rohan@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/Rohan171819)

---

<div align="center">

**⭐ Star this repo if it helped you understand forecasting — not just copy ARIMA parameters**

_"A forecast is only as good as the stationarity test that precedes it."_

</div>
