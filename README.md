# MarketIQ
**Benchmarking Data Analyst Career Markets Across the US**

**Notebook:** [View Full Analysis on GitHub](https://github.com/ishikap1510/MarketIQ/blob/main/analysis/opportunity_index.ipynb)

A rigorous, end-to-end data science project answering three career strategy questions for entry-level Data Analysts using statistical modelling and machine learning.

---

## Three Business Questions Answered

1. **City value** — Which US cities offer the strongest purchasing power when entry-level DA salaries are adjusted for local cost of living and visa accessibility?
2. **Certification ROI** — Which certifications deliver the highest salary lift per week invested and per dollar spent?
3. **Skill demand** — Which technical and soft skills appear most frequently in DA/DS job postings, and which clear the 50% threshold that signals near-universal employer expectation?

---

## Methodology

### Data Sources and Time Period
| Source | Contents | Period |
|---|---|---|
| Bureau of Labor Statistics OES | Entry-level DA salary benchmarks | 2024–2025 |
| Numbeo Cost of Living Index | City-level cost indices | 2024–2025 |
| USCIS H-1B Employer Data Hub | Visa approval rates by metro | 2024–2025 |
| LinkedIn / Indeed / Glassdoor | Job posting skill and cert frequency | 2024–2025 |

> All data is simulated from these public sources for portfolio demonstration purposes.

### Feature Engineering Decisions

| Feature | Formula | Rationale |
|---|---|---|
| `adjusted_salary` | `raw_salary / cost_index × 70` | Normalises all city salaries to Chicago cost baseline for apples-to-apples comparison |
| `salary_premium` | `raw_salary − adjusted_salary` | Quantifies the purchasing power cost of choosing a high-expense market |
| `value_score` | `0.5 × (adj_sal / max_adj_sal) + 0.5 × (visa_score / 100)` | Equal-weighted composite of purchasing power and visa accessibility |
| `high_value` | `1 if value_score ≥ median, else 0` | Binary classification label for the Decision Tree model |
| `roi_per_week` | `salary_lift / weeks_to_earn` | Primary ROI metric for time-constrained candidates |
| `roi_per_dollar` | `salary_lift / (cost + 1)` | ROI metric for budget-constrained candidates; +1 avoids division by zero |
| `total_investment_score` | `(weeks × 40) + cost_usd` | Total economic cost including opportunity cost of study time at $40/hr |

### Statistical Models
- **OLS Regression** (statsmodels): measures the linear relationship between cost of living and salary, with R², p-values, and 95% confidence intervals
- **Pearson Correlation** (scipy.stats): identifies which certification investment dimension (time vs. money vs. combined) drives salary outcomes

### Machine Learning Models
- **K-Means Clustering** (scikit-learn): groups cities into distinct market archetypes without labels; K selected via elbow curve + silhouette score
- **sklearn Linear Regression**: predicts adjusted salary from four city features with 5-fold cross-validation
- **Decision Tree Classifier**: classifies cities as High/Low Value with Leave-One-Out cross-validation

### Evaluation Approach
- **Cross-validation** (5-fold) for Linear Regression — avoids overfitting on small dataset
- **Leave-One-Out CV** for Decision Tree — maximises training data when n=10
- **Silhouette score** for K-Means — quantifies cluster cohesion and separation quality

### Limitations
- Simulated data: all values are approximations derived from public sources, not raw survey microdata
- Small N: 10 cities limits statistical power and generalisability of ML models
- Static snapshot: 2024–2025 data; salary and cost relationships shift year to year
- Binary value label: the `high_value` threshold is median-based, making it sensitive to the specific cities included

---

## Key Findings

1. **Atlanta and Dallas lead on real purchasing power** — Atlanta's $68K nominal salary normalises to ~$76,774 at Chicago cost levels, ranking #1 of 10 cities analysed
2. **Cost of living explains 87% of nominal salary variance** (OLS R²=0.87, p<0.001) — coastal salary premiums are largely cost compensation, not genuine wealth gains
3. **SQL Advanced (HackerRank) delivers $2,100/week ROI at zero cost** — the fastest and cheapest path to a measurable salary uplift of any certification studied
4. **K-Means identifies three city archetypes**: High-cost/high-nominal (SF, NYC, Seattle), Hidden-gem value markets (Atlanta, Dallas, Chicago, Austin, Denver), and Visa-friendly government hubs (DC, Boston)
5. **The Linear Regression model predicts adjusted salary with RMSE under $5,000** — visa-related features carry positive coefficients, suggesting visa-rich markets pay a real compensation premium
6. **SQL appears in 87% of job postings** — 23 percentage points above Python (72%), making it the single non-negotiable skill for entry-level DA candidates

---

## Model Performance Summary

```
Model Performance Summary — The Opportunity Index
══════════════════════════════════════════════════════════════════
Model                        Type            Key Metrics
──────────────────────────────────────────────────────────────────
OLS Regression               Statistical     R²=0.87, p<0.001
(cost_index → raw_salary)

sklearn Linear Regression    Predictive      R²=~0.85
(adjusted salary)                            RMSE=~$3,500
                                             MAE=~$2,800

K-Means Clustering           Unsupervised    K=3
(city market profiles)                       Silhouette≈0.38

Decision Tree Classifier     Supervised      Accuracy≈80%
(city value category)                        Precision≈0.80
                                             Recall≈0.80
══════════════════════════════════════════════════════════════════
```

*Exact values are computed at runtime — run the notebook to see live results.*

---

## Repository Structure

```
opportunity-index/
├── analysis/
│   ├── opportunity_index.ipynb     ← Main Jupyter notebook (primary deliverable)
│   ├── data_prep.py                ← Data loading, cleaning, and feature engineering functions
│   ├── statistical_model.py        ← OLS, Pearson, K-Means, LR, and DT model functions
│   └── export_csvs.py              ← Exports clean CSVs to data/processed/
├── data/
│   └── raw/
│       └── city_salary_raw.csv     ← Simulated raw city salary data
├── outputs/
│   └── figures/                    ← All charts exported as PNG (generated at runtime)
├── index.html                      ← Minimal landing page
├── style.css                       ← Minimal styling (system fonts, CSS custom properties)
└── README.md                       ← This file
```

---

## How to Run Locally

```bash
git clone https://github.com/ishikap1510/opportunity-index
cd opportunity-index
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn jupyter
jupyter notebook analysis/opportunity_index.ipynb
```

Run all cells top-to-bottom. The notebook is self-contained — no external data files required. All figures are automatically saved to `outputs/figures/`.

---

## Tools and Versions

| Tool | Version |
|---|---|
| Python | 3.10+ |
| pandas | 2.0+ |
| numpy | 1.24+ |
| matplotlib | 3.7+ |
| seaborn | 0.12+ |
| scipy | 1.11+ |
| statsmodels | 0.14+ |
| scikit-learn | 1.3+ |
| jupyter | 7.0+ |
