# Retail Customer Pattern Mining
### CS 4412 – Data Mining | Kennesaw State University
**Author:** Ulrich Batanado | ubatanad@students.kennesaw.edu

---

## Project Overview

This project applies the Knowledge Discovery in Databases (KDD) process to a 100,000-row retail transaction dataset to discover meaningful patterns in customer purchasing behavior. The analysis addresses three discovery questions:

1. **Q1 – Association:** Which product categories are co-purchased by customers?
2. **Q2 – Segmentation:** What distinct customer segments exist based on purchasing behavior?
3. **Q3 – Anomalies:** Which transactions are unusual or potentially anomalous?

## Key Findings

| Discovery Question | Finding |
|---|---|
| Category co-purchase patterns | **Category loyalty dominates** — 94.6% of customers buy from only one category. All co-purchase lift values < 0.1. |
| Customer segmentation | **3 archetypes**: Budget Single-Purchase (~49%), Repeat Mid-Tier (~39%), Premium Single-Purchase (~12%) |
| Anomalous transactions | **5.0% flagged** by Isolation Forest — characterized by high amount, high quantity, high price |

## Techniques Applied

| Technique | Purpose | Result |
|---|---|---|
| K-Means Clustering | Customer segmentation | k=3, silhouette=0.437 |
| Association Rule Mining | Category co-purchase analysis | Lift < 1.0 (category loyalty) |
| Isolation Forest | Transaction anomaly detection | 4,999 anomalies (5%) |
| Decision Tree | High-value customer classification | 99.98% CV accuracy |


## How to Run

### Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn reportlab
```

### Run Analysis
```bash
python3 code/analysis.py          # Generates all figures and outputs/
```

## Dataset

- **Source:** [Kaggle – Retail Transaction Dataset](https://www.kaggle.com/datasets/fahadrehman07/retail-transaction-dataset/data)
- **Size:** 100,000 rows × 10 columns
- **Date range:** April 2023 – April 2024
- **Unique customers:** 95,215
- **Categories:** Books, Clothing, Electronics, Home Decor
- **Payment methods:** Cash, Credit Card, Debit Card, PayPal

## Critical Notes

- ProductID contains only 4 values (A–D), mapping 1:1 to ProductCategory — product-level basket analysis is degenerate
- Dataset exhibits hallmarks of synthetic generation (uniform distributions, no seasonality)
- All findings should be interpreted as illustrative of methodology rather than claims about real retail consumer behavior

## References

1. Han, Kamber & Pei (2012). *Data Mining: Concepts and Techniques*, 3rd ed. Morgan Kaufmann.
2. Liu, Ting & Zhou (2008). Isolation Forest. IEEE ICDM.
3. Pedregosa et al. (2011). Scikit-learn. JMLR 12:2825–2830.
