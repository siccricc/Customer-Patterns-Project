"""
CS 4412 - Data Mining
Retail Transaction Dataset - Full Analysis (M3 + M4)
Author: Ulrich Batanado
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

# Base directory = project root (one level up from Code)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Define paths
DATA_PATH = os.path.join(BASE_DIR, 'Data', 'Retail_Transaction_Dataset.csv')
FIGS = os.path.join(BASE_DIR, 'Figures')
OUTPUTS = os.path.join(BASE_DIR, 'Outputs')

# Create folders if they don’t exist
os.makedirs(FIGS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)


np.random.seed(42)

# ─────────────────────────────────────────────
# 1. DATA LOADING & INITIAL EXPLORATION
# ─────────────────────────────────────────────
print("=" * 60)
print("1. DATA LOADING & EXPLORATION")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBasic stats:\n{df.describe()}")

# ─────────────────────────────────────────────
# 2. DATA CLEANING & FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. DATA CLEANING & FEATURE ENGINEERING")
print("=" * 60)

# Parse date and extract temporal features
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
df['Month'] = df['TransactionDate'].dt.month
df['DayOfWeek'] = df['TransactionDate'].dt.dayofweek
df['Hour'] = df['TransactionDate'].dt.hour
df['Quarter'] = df['TransactionDate'].dt.quarter

# Remove duplicates
before = len(df)
df.drop_duplicates(inplace=True)
print(f"Duplicates removed: {before - len(df)}")

# Drop impossible values
df = df[df['Price'] > 0]
df = df[df['Quantity'] > 0]
df = df[df['TotalAmount'] > 0]
print(f"After cleaning: {len(df)} rows")

# Rename discount column for convenience
df.rename(columns={'DiscountApplied(%)': 'Discount'}, inplace=True)

# Derived columns
df['DiscountAmount'] = df['Price'] * df['Quantity'] * df['Discount'] / 100
df['EffectiveUnitPrice'] = df['TotalAmount'] / df['Quantity']

print(f"\nProduct categories: {df['ProductCategory'].value_counts().to_dict()}")
print(f"Payment methods:    {df['PaymentMethod'].value_counts().to_dict()}")
print(f"Date range: {df['TransactionDate'].min()} to {df['TransactionDate'].max()}")

# ─────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS VISUALIZATIONS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. EDA VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Retail Transaction Dataset – Exploratory Data Analysis', fontsize=16, fontweight='bold')

# Transaction amount distribution
axes[0, 0].hist(df['TotalAmount'], bins=60, color='steelblue', edgecolor='white', alpha=0.85)
axes[0, 0].axvline(df['TotalAmount'].median(), color='red', linestyle='--',
                   label=f"Median: ${df['TotalAmount'].median():.0f}")
axes[0, 0].set_title('Distribution of Transaction Amounts')
axes[0, 0].set_xlabel('Total Amount ($)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()

# Category breakdown
cat_counts = df['ProductCategory'].value_counts()
axes[0, 1].bar(cat_counts.index, cat_counts.values,
               color=['#4e79a7', '#f28e2b', '#e15759', '#76b7b2'])
axes[0, 1].set_title('Transactions by Product Category')
axes[0, 1].set_xlabel('Category')
axes[0, 1].set_ylabel('Number of Transactions')

# Payment method pie
pay_counts = df['PaymentMethod'].value_counts()
axes[0, 2].pie(pay_counts.values, labels=pay_counts.index, autopct='%1.1f%%',
               colors=sns.color_palette('pastel'))
axes[0, 2].set_title('Payment Method Distribution')

# Monthly transaction volume
monthly = df.groupby('Month').size()
axes[1, 0].plot(monthly.index, monthly.values, marker='o', color='darkorange', linewidth=2)
axes[1, 0].set_title('Monthly Transaction Volume')
axes[1, 0].set_xlabel('Month')
axes[1, 0].set_ylabel('Number of Transactions')
axes[1, 0].set_xticks(range(1, 13))
axes[1, 0].grid(alpha=0.3)

# Quantity distribution
axes[1, 1].hist(df['Quantity'], bins=30, color='mediumseagreen', edgecolor='white', alpha=0.85)
axes[1, 1].set_title('Distribution of Quantity per Transaction')
axes[1, 1].set_xlabel('Quantity')
axes[1, 1].set_ylabel('Frequency')

# Discount vs. TotalAmount scatter
sample = df.sample(3000, random_state=42)
axes[1, 2].scatter(sample['Discount'], sample['TotalAmount'], alpha=0.3, color='purple', s=10)
axes[1, 2].set_title('Discount % vs. Transaction Amount')
axes[1, 2].set_xlabel('Discount (%)')
axes[1, 2].set_ylabel('Total Amount ($)')

plt.tight_layout()
plt.savefig(f'{FIGS}/fig1_eda_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 1 saved.")

# ─────────────────────────────────────────────
# 4. CUSTOMER-LEVEL FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. CUSTOMER-LEVEL AGGREGATION")
print("=" * 60)

customer = df.groupby('CustomerID').agg(
    TotalSpend=('TotalAmount', 'sum'),
    NumTransactions=('TotalAmount', 'count'),
    AvgTransactionValue=('TotalAmount', 'mean'),
    AvgQuantity=('Quantity', 'mean'),
    AvgDiscount=('Discount', 'mean'),
    UniqueCategories=('ProductCategory', 'nunique'),
    UniqueProducts=('ProductID', 'nunique'),
).reset_index()

customer['SpendPerTransaction'] = customer['TotalSpend'] / customer['NumTransactions']

print(f"Unique customers: {len(customer)}")
print(customer.describe())

# ─────────────────────────────────────────────
# 5. CLUSTERING – K-MEANS CUSTOMER SEGMENTATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. K-MEANS CUSTOMER SEGMENTATION")
print("=" * 60)

features_cluster = ['TotalSpend', 'NumTransactions', 'AvgTransactionValue', 'AvgDiscount', 'UniqueCategories']
X_cluster = customer[features_cluster].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Evaluate k=2..8 via elbow + silhouette
inertias = []
silhouettes = []
k_range = range(2, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels, sample_size=10000)
    silhouettes.append(sil)
    print(f"  k={k}: inertia={km.inertia_:.0f}, silhouette={sil:.4f}")

# k=3 gives the best silhouette after the large drop from k=2
best_k = 3
print(f"\nChosen k: {best_k}")

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=200)
customer['Cluster'] = km_final.fit_predict(X_scaled)

# Named segments based on profile inspection
cluster_names = ['Budget Single-Purchase', 'Repeat Mid-Tier', 'Premium Single-Purchase']
colors_c = ['#e6194b', '#3cb44b', '#4363d8']

cluster_profile = customer.groupby('Cluster')[features_cluster].mean()
print("\nCluster Profiles:")
print(cluster_profile)
print(f"\nCluster sizes: {customer['Cluster'].value_counts().sort_index().to_dict()}")

# ── Figure 2: K selection curves + meaningful scatter ────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('K-Means Customer Segmentation (k=3)', fontsize=14, fontweight='bold')

# Elbow curve — full k=2..8 range
axes[0].plot(list(k_range), inertias, 'bo-', lw=2, ms=7)
axes[0].axvline(best_k, color='gray', ls='--', alpha=0.7, label=f'Chosen k={best_k}')
axes[0].set_title('Elbow Curve (Inertia vs k)')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia')
axes[0].set_xticks(list(k_range))
axes[0].legend()
axes[0].grid(alpha=0.3)

# Silhouette scores — full k=2..8 range
axes[1].plot(list(k_range), silhouettes, 'rs-', lw=2, ms=7)
axes[1].axvline(best_k, color='gray', ls='--', alpha=0.7, label=f'Chosen k={best_k}')
axes[1].set_title('Silhouette Score vs k')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_xticks(list(k_range))
axes[1].legend()
axes[1].grid(alpha=0.3)

# Scatter: AvgTransactionValue vs AvgDiscount — these two features are independent of each
# other (no mathematical relationship), so clusters separate visibly without collapsing to a
# diagonal line (as TotalSpend vs AvgTransactionValue would) or vertical bars (NumTransactions
# is an integer with 90%+ of values = 1, producing degenerate vertical-line scatter plots).
sample_c = customer.sample(8000, random_state=42)
for c in range(best_k):
    m = sample_c['Cluster'] == c
    axes[2].scatter(sample_c.loc[m, 'AvgTransactionValue'], sample_c.loc[m, 'AvgDiscount'],
                    s=15, alpha=0.4, color=colors_c[c], label=cluster_names[c])
axes[2].set_title('Cluster Separation:\nAvg Transaction Value vs Avg Discount')
axes[2].set_xlabel('Avg Transaction Value ($)')
axes[2].set_ylabel('Avg Discount (%)')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{FIGS}/fig2_kmeans.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 2 saved.")

# ── Figure 3: Cluster profiles ───────────────────────────────
# Box plots + grouped bar chart replace the original heatmap (all values near-identical,
# looked uniform) and the Spend vs Transactions scatter (NumTransactions is integer-valued
# with ~90% = 1, producing meaningless vertical lines at x=1, 2, 3).
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Customer Cluster Profiles', fontsize=14, fontweight='bold')

# Box plots of AvgTransactionValue per cluster
data_by_cluster = [customer[customer['Cluster'] == c]['AvgTransactionValue'].values
                   for c in range(best_k)]
bp = axes[0].boxplot(data_by_cluster, patch_artist=True, labels=cluster_names,
                     flierprops=dict(marker='.', markersize=3, alpha=0.3))
for patch, color in zip(bp['boxes'], colors_c):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0].set_title('Avg Transaction Value Distribution by Cluster')
axes[0].set_ylabel('Avg Transaction Value ($)')
axes[0].tick_params(axis='x', rotation=12)
axes[0].grid(alpha=0.3, axis='y')

# Grouped bar chart: normalized feature means per cluster
prof = customer.groupby('Cluster')[['TotalSpend', 'AvgTransactionValue', 'AvgDiscount']].mean()
prof_n = (prof - prof.min()) / (prof.max() - prof.min())
x = np.arange(best_k)
w = 0.25
metrics = ['TotalSpend', 'AvgTransactionValue', 'AvgDiscount']
metric_labels = ['Total Spend', 'Avg Txn Value', 'Avg Discount']
bar_colors = ['#4e79a7', '#f28e2b', '#e15759']
for i, (met, lbl, bc) in enumerate(zip(metrics, metric_labels, bar_colors)):
    axes[1].bar(x + i * w, prof_n[met].values, w, label=lbl, color=bc, alpha=0.85)
axes[1].set_title('Normalized Feature Comparison by Cluster\n(0 = lowest, 1 = highest across clusters)')
axes[1].set_xlabel('Cluster')
axes[1].set_ylabel('Normalized Mean Value')
axes[1].set_xticks(x + w)
axes[1].set_xticklabels(cluster_names, rotation=12, fontsize=9)
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{FIGS}/fig3_cluster_profiles.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 3 saved.")

# ─────────────────────────────────────────────
# 6. ASSOCIATION RULE MINING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. ASSOCIATION RULE MINING")
print("=" * 60)

# ProductID maps 1:1 to ProductCategory, so category-level is the meaningful grain.
# Each customer's lifetime set of categories is treated as one "transaction".
cust_categories = df.groupby('CustomerID')['ProductCategory'].apply(list)
cat_list = sorted(df['ProductCategory'].unique())
total_customers = len(cust_categories)

cat_counts = {}
pair_counts = {}
for cats in cust_categories:
    unique_cats = set(cats)
    for c in unique_cats:
        cat_counts[c] = cat_counts.get(c, 0) + 1
    for pair in combinations(sorted(unique_cats), 2):
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

rules = []
for (A, B), count_AB in pair_counts.items():
    sup_AB = count_AB / total_customers
    sup_A = cat_counts[A] / total_customers
    sup_B = cat_counts[B] / total_customers
    lift = sup_AB / (sup_A * sup_B)
    rules.append({'antecedent': A, 'consequent': B,
                  'support': round(sup_AB, 4), 'confidence': round(sup_AB / sup_A, 4),
                  'lift': round(lift, 4)})
    rules.append({'antecedent': B, 'consequent': A,
                  'support': round(sup_AB, 4), 'confidence': round(sup_AB / sup_B, 4),
                  'lift': round(lift, 4)})

rules_df = pd.DataFrame(rules).sort_values('lift', ascending=False)
print("All association rules (sorted by lift):")
print(rules_df.to_string(index=False))

# Category diversity: how many distinct categories per customer?
cat_diversity = df.groupby('CustomerID')['ProductCategory'].nunique()
loyalty_counts = cat_diversity.value_counts().sort_index()
print(f"\nCategory diversity distribution:\n{loyalty_counts}")
print(f"Single-category customers: {loyalty_counts[1]} ({loyalty_counts[1] / total_customers * 100:.1f}%)")

rules_df.to_csv('outputs/association_rules.csv', index=False)

# ── Figure 4: Association rules ──────────────────────────────
# The original heatmap was entirely one color (all lifts ~0.096, indistinguishable on
# the coolwarm scale) and the confidence bar chart bars were all the same length — both
# panels conveyed nothing. Replaced with the actual headline finding (category loyalty)
# and observed vs expected rates that explain why lift is uniformly below 1.
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Association Rule Mining – Category Co-Purchase Analysis', fontsize=14, fontweight='bold')

# LEFT: Customer category loyalty — the key discovery
loyalty_pct = loyalty_counts / loyalty_counts.sum() * 100
bars = axes[0].bar(loyalty_counts.index, loyalty_pct.values,
                   color=['#2196F3', '#FF9800', '#4CAF50', '#9C27B0'],
                   edgecolor='white', alpha=0.87, width=0.6)
for bar, pct in zip(bars, loyalty_pct.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
axes[0].set_title('Customer Category Loyalty\n(Number of Distinct Categories Purchased)')
axes[0].set_xlabel('Number of Distinct Product Categories Purchased')
axes[0].set_ylabel('Percentage of Customers (%)')
axes[0].set_xticks([1, 2, 3, 4])
axes[0].set_xticklabels(['1 category\n(Loyal)', '2 categories', '3 categories', '4 categories'])
axes[0].set_ylim(0, 105)
axes[0].grid(alpha=0.3, axis='y')

# RIGHT: Observed vs expected co-purchase rates — explains why lift < 1
pair_labels = [f'{A[:6]}-{B[:6]}' for (A, B) in pair_counts.keys()]
pair_supports = [cnt / total_customers * 100 for cnt in pair_counts.values()]
pair_expected = [(cat_counts[A] / total_customers) * (cat_counts[B] / total_customers) * 100
                 for (A, B) in pair_counts.keys()]

x = np.arange(len(pair_labels))
w = 0.35
axes[1].bar(x - w / 2, pair_supports, w, label='Observed co-purchase rate',
            color='steelblue', alpha=0.85)
axes[1].bar(x + w / 2, pair_expected, w, label='Expected if independent',
            color='coral', alpha=0.85)
axes[1].set_title('Observed vs Expected Co-Purchase Rate\n(% of customers buying both categories)')
axes[1].set_xlabel('Category Pair')
axes[1].set_ylabel('% of All Customers')
axes[1].set_xticks(x)
axes[1].set_xticklabels(pair_labels, rotation=30, fontsize=8)
axes[1].legend()
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{FIGS}/fig4_association_rules.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 4 saved.")

# ─────────────────────────────────────────────
# 7. ANOMALY DETECTION – ISOLATION FOREST
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. ANOMALY DETECTION")
print("=" * 60)

features_anomaly = ['TotalAmount', 'Quantity', 'Price', 'Discount']
X_anomaly = df[features_anomaly].values

iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
df['AnomalyScore'] = iso.fit_predict(X_anomaly)
df['AnomalyScoreRaw'] = iso.decision_function(X_anomaly)

n_anomalies = (df['AnomalyScore'] == -1).sum()
print(f"Anomalies detected: {n_anomalies} ({n_anomalies / len(df) * 100:.1f}%)")

anomalies = df[df['AnomalyScore'] == -1]
normal = df[df['AnomalyScore'] == 1]

print("\nAnomaly vs Normal Transaction Stats:")
for col in features_anomaly:
    print(f"  {col}: anomaly_mean={anomalies[col].mean():.2f}, normal_mean={normal[col].mean():.2f}")

print("\nAnomaly category distribution:")
print(anomalies['ProductCategory'].value_counts())
print("\nAnomaly payment method distribution:")
print(anomalies['PaymentMethod'].value_counts())

# IQR comparison check
Q1 = df['TotalAmount'].quantile(0.25)
Q3 = df['TotalAmount'].quantile(0.75)
IQR = Q3 - Q1
iqr_outliers = df[(df['TotalAmount'] < Q1 - 1.5 * IQR) | (df['TotalAmount'] > Q3 + 1.5 * IQR)]
print(f"\nIQR-based outliers (TotalAmount): {len(iqr_outliers)}")

# ── Figure 5: Anomaly detection ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Anomaly Detection – Isolation Forest (5% contamination)', fontsize=14, fontweight='bold')

axes[0].hist(df['AnomalyScoreRaw'], bins=50, color='gray', edgecolor='white', alpha=0.7)
axes[0].axvline(0, color='red', linestyle='--', label='Decision Boundary')
axes[0].set_title('Anomaly Score Distribution')
axes[0].set_xlabel('Anomaly Score (lower = more anomalous)')
axes[0].set_ylabel('Count')
axes[0].legend()

sample_normal = normal.sample(min(2000, len(normal)), random_state=42)
sample_anom = anomalies.sample(min(500, len(anomalies)), random_state=42)
axes[1].scatter(sample_normal['Quantity'], sample_normal['TotalAmount'],
                s=8, alpha=0.3, color='steelblue', label='Normal')
axes[1].scatter(sample_anom['Quantity'], sample_anom['TotalAmount'],
                s=25, alpha=0.6, color='red', label='Anomaly')
axes[1].set_title('Amount vs Quantity\n(Anomalies Highlighted)')
axes[1].set_xlabel('Quantity')
axes[1].set_ylabel('Total Amount ($)')
axes[1].legend()

anom_rate = df.groupby('ProductCategory')['AnomalyScore'].apply(lambda x: (x == -1).mean() * 100)
anom_rate.sort_values().plot(kind='barh', ax=axes[2], color='tomato', alpha=0.8)
axes[2].set_title('Anomaly Rate by Product Category (%)')
axes[2].set_xlabel('Anomaly Rate (%)')

plt.tight_layout()
plt.savefig(f'{FIGS}/fig5_anomaly_detection.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 5 saved.")

# ─────────────────────────────────────────────
# 8. DECISION TREE CLASSIFICATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. DECISION TREE – HIGH-VALUE CUSTOMER CLASSIFICATION")
print("=" * 60)

# High-value = top 25% of customers by total spend
threshold = customer['TotalSpend'].quantile(0.75)
customer['HighValue'] = (customer['TotalSpend'] >= threshold).astype(int)
print(f"High-value threshold: ${threshold:.2f}")
print(f"High-value customers: {customer['HighValue'].sum()} ({customer['HighValue'].mean() * 100:.1f}%)")

feat_dt = ['NumTransactions', 'AvgTransactionValue', 'AvgDiscount', 'UniqueCategories', 'AvgQuantity']
X_dt = customer[feat_dt]
y_dt = customer['HighValue']

dt = DecisionTreeClassifier(max_depth=4, random_state=42, min_samples_leaf=20)
dt.fit(X_dt, y_dt)

cv_scores = cross_val_score(dt, X_dt, y_dt, cv=5, scoring='accuracy')
print(f"\nDecision Tree CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

importances = pd.Series(dt.feature_importances_, index=feat_dt).sort_values(ascending=False)
print(f"\nFeature Importances:\n{importances}")
print(f"\nDecision Rules (depth=4):")
print(export_text(dt, feature_names=feat_dt))

# ── Figure 6: Decision tree ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Decision Tree – High-Value Customer Classification', fontsize=14, fontweight='bold')

importances.plot(kind='bar', ax=axes[0], color='mediumorchid', alpha=0.8)
axes[0].set_title('Feature Importances')
axes[0].set_ylabel('Importance Score')
axes[0].tick_params(axis='x', rotation=30)

customer[customer['HighValue'] == 0]['TotalSpend'].hist(bins=40, ax=axes[1],
                                                        color='steelblue', alpha=0.6, label='Regular')
customer[customer['HighValue'] == 1]['TotalSpend'].hist(bins=40, ax=axes[1],
                                                        color='tomato', alpha=0.6, label='High-Value')
axes[1].axvline(threshold, color='black', linestyle='--', label=f'Threshold (${threshold:.0f})')
axes[1].set_title('Total Spend Distribution by Customer Class')
axes[1].set_xlabel('Total Spend ($)')
axes[1].set_ylabel('Count')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{FIGS}/fig6_decision_tree.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 6 saved.")

# ─────────────────────────────────────────────
# 9. TEMPORAL ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. TEMPORAL PATTERNS")
print("=" * 60)

monthly_cat = df.groupby(['Month', 'ProductCategory'])['TotalAmount'].sum().reset_index()
monthly_pivot = monthly_cat.pivot(index='Month', columns='ProductCategory', values='TotalAmount').fillna(0)

dow_spend = df.groupby('DayOfWeek')['TotalAmount'].mean()
dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

print("Average spend by day of week:")
for i, v in dow_spend.items():
    print(f"  {dow_labels[i]}: ${v:.2f}")

# ── Figure 7: Temporal patterns ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Temporal Sales Patterns', fontsize=14, fontweight='bold')

monthly_pivot.plot(ax=axes[0], linewidth=1.5, marker='o', markersize=4)
axes[0].set_title('Monthly Revenue by Product Category')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Total Revenue ($)')
axes[0].legend(fontsize=8, loc='upper right')
axes[0].grid(alpha=0.3)

dow_spend.plot(kind='bar', ax=axes[1], color='teal', alpha=0.8)
axes[1].set_title('Average Transaction Amount by Day of Week')
axes[1].set_xlabel('Day of Week')
axes[1].set_ylabel('Average Amount ($)')
axes[1].set_xticklabels(dow_labels, rotation=0)
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{FIGS}/fig7_temporal_patterns.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 7 saved.")

# ─────────────────────────────────────────────
# 10. STABILITY ANALYSIS (M4 Critical Assessment)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("10. CLUSTERING STABILITY ANALYSIS")
print("=" * 60)

sil_scores_stability = []
for seed in range(10):
    km_test = KMeans(n_clusters=best_k, random_state=seed, n_init=10)
    lbl = km_test.fit_predict(X_scaled)
    sil_scores_stability.append(silhouette_score(X_scaled, lbl, sample_size=10000))

print(f"Silhouette across 10 seeds: mean={np.mean(sil_scores_stability):.4f}, std={np.std(sil_scores_stability):.4f}")
print(f"Scores: {[round(s, 4) for s in sil_scores_stability]}")

# ─────────────────────────────────────────────
# 11. SUMMARY STATISTICS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

print(f"\nDataset: {len(df):,} transactions, {len(customer):,} unique customers")
print(f"Date range: {df['TransactionDate'].min().date()} to {df['TransactionDate'].max().date()}")
print(f"Total revenue: ${df['TotalAmount'].sum():,.2f}")
print(f"Average transaction: ${df['TotalAmount'].mean():.2f}")

print(f"\nClustering: {best_k} customer segments")
for c in range(best_k):
    seg = customer[customer['Cluster'] == c]
    print(f"  {cluster_names[c]}: n={len(seg)}, avg_spend=${seg['TotalSpend'].mean():.0f}, "
          f"avg_transactions={seg['NumTransactions'].mean():.1f}")

print(f"\nTop association rule: {rules_df.iloc[0]['antecedent']} -> {rules_df.iloc[0]['consequent']} "
      f"(lift={rules_df.iloc[0]['lift']:.3f}, conf={rules_df.iloc[0]['confidence']:.3f})")

print(f"\nAnomalies: {n_anomalies:,} ({n_anomalies / len(df) * 100:.1f}% of transactions)")
print(f"Anomaly avg transaction: ${anomalies['TotalAmount'].mean():.2f} "
      f"vs normal ${normal['TotalAmount'].mean():.2f}")

print(f"\nDecision tree accuracy: {cv_scores.mean() * 100:.1f}%")
print(f"Top predictive feature: {importances.index[0]} (importance={importances.iloc[0]:.4f})")

# Save key data for report
customer.to_csv(os.path.join(OUTPUTS, 'customer_segments.csv'), index=False)
rules_df.to_csv(os.path.join(OUTPUTS, 'association_rules.csv'), index=False)
anomalies.to_csv(os.path.join(OUTPUTS, 'anomalies.csv'), index=False)

print("\nAll analysis complete. Figures and outputs saved.") 
#And done
