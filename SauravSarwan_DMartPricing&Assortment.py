"""
DMart Product Catalogue Analysis
=================================
IBM SkillsBuild Data Analytics with AI Academic Internship
BharatCares in association with AICTE

Author   : Saurav Sarwan
Dataset  : DMart.csv
File     : SauravSarwan_DMartPricing&Assortment.py

Problem Statement
-----------------
DMart offers a wide range of products across grocery, packaged food,
personal care, home & kitchen, beverages, and other categories. Managing
such a large product assortment requires effective pricing and assortment
decisions to maintain customer value while avoiding excessive discounting
and identifying gaps in product and brand coverage.

Business Objectives
-------------------
1. Understand the overall product and brand assortment
2. Evaluate price and discount patterns across categories, subcategories,
   and brands
3. Identify factors associated with higher discounting and pricing differences
4. Detect pricing, data-quality, and assortment-related risks
5. Identify potential opportunities for assortment expansion and pricing review
6. Provide management-oriented recommendations through an interactive
   decision dashboard

Usage
-----
    python "SauravSarwan_DMartPricing&Assortment.py"

After execution all JSON outputs are written to outputs/ and the
interactive dashboard can be opened at frontend/dashboard.html.
"""

import pandas as pd
import numpy as np
import json
import os
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH   = "DMart.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEPARATOR = "=" * 60


def save_json(obj, filename: str) -> None:
    """Serialise obj to JSON and write to outputs/<filename>."""
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    print(f"  [saved] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────────────────────
print(SEPARATOR)
print("  STEP 1 — DATA LOADING & CLEANING")
print(SEPARATOR)

df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
df.columns = df.columns.str.strip()
print(f"Raw records loaded  : {len(df):,}")
print(f"Columns             : {list(df.columns)}")

# Coerce numeric types
df["Price"]           = pd.to_numeric(df["Price"],           errors="coerce")
df["DiscountedPrice"] = pd.to_numeric(df["DiscountedPrice"], errors="coerce")

# Drop null / zero-price rows
df = df[(df["Price"] > 0) & (df["DiscountedPrice"] > 0)].copy()

# Fix reversed prices (discounted > MRP)
mask = df["DiscountedPrice"] > df["Price"]
df.loc[mask, ["Price", "DiscountedPrice"]] = df.loc[
    mask, ["DiscountedPrice", "Price"]
].values
print(f"Reversed-price rows fixed: {mask.sum()}")

# Feature engineering
df["DiscountAmount"] = df["Price"] - df["DiscountedPrice"]
df["DiscountPct"]    = (df["DiscountAmount"] / df["Price"] * 100).round(2)
df["PriceRange"]     = pd.cut(
    df["Price"],
    bins=[0, 100, 300, 600, 1000, 5000, np.inf],
    labels=["Rs0-100", "Rs101-300", "Rs301-600", "Rs601-1K", "Rs1K-5K", "Rs5K+"],
)

# Clean text fields
df["Brand"]       = df["Brand"].fillna("").str.strip()
df["Category"]    = df["Category"].fillna("Unknown").str.strip()
df["SubCategory"] = df["SubCategory"].fillna("Unknown").str.strip()

# Boolean anomaly flags
df["IsHighDiscount"] = df["DiscountPct"] >= 40
df["IsZeroDiscount"] = df["DiscountPct"] == 0
df["IsNoBrand"]      = df["Brand"] == ""

print(f"Clean records       : {len(df):,}")
print(f"Unique categories   : {df['Category'].nunique()}")
print(f"Unique subcategories: {df['SubCategory'].nunique()}")
print(f"Unique brands       : {df[df['Brand'] != '']['Brand'].nunique()}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print(SEPARATOR)
print("  STEP 2 — EXPLORATORY DATA ANALYSIS")
print(SEPARATOR)

print("\nBasic Price & Discount Statistics:")
print(df[["Price", "DiscountedPrice", "DiscountPct"]].describe().round(2).to_string())

print("\nMissing Values:")
print(df[["Name", "Brand", "Category", "SubCategory", "Price", "DiscountedPrice"]].isnull().sum().to_string())

print("\nTop 10 Categories by Product Count:")
print(df["Category"].value_counts().head(10).to_string())
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — KPI COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
print(SEPARATOR)
print("  STEP 3 — KPI COMPUTATION")
print(SEPARATOR)

kpis = {
    "total_products":             len(df),
    "total_brands":               df[df["Brand"] != ""]["Brand"].nunique(),
    "total_categories":           df["Category"].nunique(),
    "total_subcategories":        df["SubCategory"].nunique(),
    "avg_mrp":                    round(df["Price"].mean(), 2),
    "median_mrp":                 round(df["Price"].median(), 2),
    "avg_discount_pct":           round(df["DiscountPct"].mean(), 2),
    "max_discount_pct":           round(df["DiscountPct"].max(), 2),
    "high_discount_products":     int(df["IsHighDiscount"].sum()),
    "high_discount_pct_of_total": round(df["IsHighDiscount"].sum() / len(df) * 100, 1),
    "no_brand_products":          int(df["IsNoBrand"].sum()),
    "zero_discount_products":     int(df["IsZeroDiscount"].sum()),
}

for k, v in kpis.items():
    print(f"  {k:<35}: {v}")

save_json(kpis, "kpis.json")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — CATEGORY & SUBCATEGORY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print(SEPARATOR)
print("  STEP 4 — CATEGORY & SUBCATEGORY ANALYSIS")
print(SEPARATOR)

cat_grp = (
    df.groupby("Category")
    .agg(
        product_count      = ("Name",           "count"),
        brand_count        = ("Brand",           lambda x: x[x != ""].nunique()),
        avg_mrp            = ("Price",           "mean"),
        avg_discounted     = ("DiscountedPrice", "mean"),
        avg_discount_pct   = ("DiscountPct",     "mean"),
        max_discount_pct   = ("DiscountPct",     "max"),
        high_discount_count= ("IsHighDiscount",  "sum"),
    )
    .reset_index()
)
for col in ["avg_mrp", "avg_discounted", "avg_discount_pct", "max_discount_pct"]:
    cat_grp[col] = cat_grp[col].round(2)
cat_grp["high_discount_pct"] = (
    cat_grp["high_discount_count"] / cat_grp["product_count"] * 100
).round(1)
cat_grp = cat_grp.sort_values("product_count", ascending=False)

print("\nTop 10 Categories:")
print(
    cat_grp.head(10)[
        ["Category", "product_count", "brand_count", "avg_mrp", "avg_discount_pct", "high_discount_pct"]
    ].to_string(index=False)
)
save_json(cat_grp.to_dict(orient="records"), "category_analysis.json")

sub_grp = (
    df.groupby(["Category", "SubCategory"])
    .agg(
        product_count    = ("Name",       "count"),
        brand_count      = ("Brand",       lambda x: x[x != ""].nunique()),
        avg_mrp          = ("Price",       "mean"),
        avg_discount_pct = ("DiscountPct", "mean"),
        max_discount_pct = ("DiscountPct", "max"),
    )
    .reset_index()
)
for col in ["avg_mrp", "avg_discount_pct", "max_discount_pct"]:
    sub_grp[col] = sub_grp[col].round(2)
sub_grp = sub_grp.sort_values("product_count", ascending=False)

print("\nTop 10 Subcategories:")
print(
    sub_grp.head(10)[
        ["Category", "SubCategory", "product_count", "brand_count", "avg_mrp", "avg_discount_pct"]
    ].to_string(index=False)
)
save_json(sub_grp.to_dict(orient="records"), "subcategory_analysis.json")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — BRAND ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print(SEPARATOR)
print("  STEP 5 — BRAND ANALYSIS")
print(SEPARATOR)

branded   = df[df["Brand"] != ""]
brand_grp = (
    branded.groupby("Brand")
    .agg(
        product_count      = ("Name",           "count"),
        category_count     = ("Category",       "nunique"),
        avg_mrp            = ("Price",           "mean"),
        avg_discount_pct   = ("DiscountPct",     "mean"),
        max_discount_pct   = ("DiscountPct",     "max"),
        high_discount_count= ("IsHighDiscount",  "sum"),
    )
    .reset_index()
)
for col in ["avg_mrp", "avg_discount_pct", "max_discount_pct"]:
    brand_grp[col] = brand_grp[col].round(2)

top30_by_products = brand_grp.sort_values("product_count", ascending=False).head(30)
top30_by_discount = brand_grp.sort_values("avg_discount_pct", ascending=False).head(30)

print("\nTop 10 Brands by SKU Count:")
print(
    top30_by_products.head(10)[
        ["Brand", "product_count", "category_count", "avg_mrp", "avg_discount_pct"]
    ].to_string(index=False)
)

brand_result = {
    "top_by_products":        top30_by_products.to_dict(orient="records"),
    "top_by_discount":        top30_by_discount.to_dict(orient="records"),
    "total_branded_products": int(len(branded)),
    "single_product_brands":  int((brand_grp["product_count"] == 1).sum()),
}
save_json(brand_result, "brand_analysis.json")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — DISCOUNT & PRICE DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────
print(SEPARATOR)
print("  STEP 6 — DISCOUNT & PRICE DISTRIBUTIONS")
print(SEPARATOR)

disc_bins   = [0, 5, 10, 20, 30, 40, 50, 60, 75, 100]
disc_labels = ["0-5%", "5-10%", "10-20%", "20-30%", "30-40%",
               "40-50%", "50-60%", "60-75%", "75-100%"]

df["DiscBucket"] = pd.cut(df["DiscountPct"], bins=disc_bins,
                          labels=disc_labels, include_lowest=True)
disc_dist = (
    df["DiscBucket"].value_counts()
    .reindex(disc_labels).fillna(0).astype(int)
    .reset_index()
)
disc_dist.columns = ["bucket", "count"]
print("\nDiscount Distribution:")
print(disc_dist.to_string(index=False))
save_json(disc_dist.to_dict(orient="records"), "discount_distribution.json")

price_labels = ["Rs0-100", "Rs101-300", "Rs301-600", "Rs601-1K", "Rs1K-5K", "Rs5K+"]
price_dist = (
    df["PriceRange"].value_counts()
    .reindex(price_labels).fillna(0).astype(int)
    .reset_index()
)
price_dist.columns = ["range", "count"]
print("\nPrice Range Distribution:")
print(price_dist.to_string(index=False))
save_json(price_dist.to_dict(orient="records"), "price_distribution.json")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — ANOMALY DETECTION (3 KEY RISKS)
# ─────────────────────────────────────────────────────────────────────────────
print(SEPARATOR)
print("  STEP 7 — ANOMALY DETECTION (3 KEY RISKS)")
print(SEPARATOR)

# Risk 1: Extreme discounts (>=70%)
extreme = (
    df[df["DiscountPct"] >= 70]
    [["Name", "Brand", "Category", "SubCategory", "Price", "DiscountedPrice", "DiscountPct"]]
    .sort_values("DiscountPct", ascending=False)
    .head(30)
)
print(f"\nRisk 1 — Extreme Discounts (>=70%): {len(extreme)} products")
print(extreme.head(5).to_string(index=False))

# Risk 2: No-brand products
no_brand = (
    df[df["IsNoBrand"]]
    [["Name", "Category", "SubCategory", "Price", "DiscountPct"]]
    .head(30)
)
print(f"\nRisk 2 — No-Brand Products: {int(df['IsNoBrand'].sum())} total (showing 5)")
print(no_brand.head(5).to_string(index=False))

# Risk 3: Zero-discount products
zero_disc = (
    df[df["IsZeroDiscount"]]
    [["Name", "Brand", "Category", "Price"]]
    .head(30)
)
print(f"\nRisk 3 — Zero-Discount Products: {int(df['IsZeroDiscount'].sum())} total")
print(zero_disc.to_string(index=False))

# Premium + high discount
premium_high = (
    df[(df["Price"] > 2000) & (df["DiscountPct"] > 50)]
    [["Name", "Brand", "Category", "Price", "DiscountedPrice", "DiscountPct"]]
    .sort_values("DiscountPct", ascending=False)
)

anomaly_result = {
    "extreme_discount_products":      extreme.to_dict(orient="records"),
    "zero_discount_products":         zero_disc.to_dict(orient="records"),
    "no_brand_products":              no_brand.to_dict(orient="records"),
    "premium_high_discount_products": premium_high.to_dict(orient="records"),
    "extreme_discount_count":         len(extreme),
    "zero_discount_count":            int(df["IsZeroDiscount"].sum()),
    "no_brand_count":                 int(df["IsNoBrand"].sum()),
}
save_json(anomaly_result, "anomalies.json")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — OPPORTUNITY IDENTIFICATION (3 KEY OPPORTUNITIES)
# ─────────────────────────────────────────────────────────────────────────────
print(SEPARATOR)
print("  STEP 8 — OPPORTUNITY IDENTIFICATION (3 KEY OPPORTUNITIES)")
print(SEPARATOR)

global_avg_disc = df["DiscountPct"].mean()

# Opportunity 1: Single-brand subcategories
sub_brand = (
    df[df["Brand"] != ""]
    .groupby("SubCategory")["Brand"].nunique()
    .reset_index()
)
sub_brand.columns = ["SubCategory", "brand_count"]
single_brand = sub_brand[sub_brand["brand_count"] == 1].sort_values("SubCategory")
print(f"\nOpportunity 1 — Single-brand subcategories: {len(single_brand)}")
print(single_brand.to_string(index=False))

# Opportunity 2: Under-promoted categories
cat_disc = df.groupby("Category")["DiscountPct"].mean().reset_index()
cat_disc.columns = ["Category", "avg_discount_pct"]
low_disc = (
    cat_disc[cat_disc["avg_discount_pct"] < global_avg_disc * 0.6]
    .sort_values("avg_discount_pct")
)
print(f"\nOpportunity 2 — Under-promoted categories (avg below {round(global_avg_disc*0.6,1)}%):")
print(low_disc.to_string(index=False))

# Opportunity 3: Premium segment gaps
sub_summary = (
    df[df["Brand"] != ""]
    .groupby("SubCategory")
    .agg(avg_mrp=("Price", "mean"), brand_count=("Brand", "nunique"))
    .reset_index()
)
premium_gap = (
    sub_summary[
        (sub_summary["avg_mrp"] > sub_summary["avg_mrp"].median())
        & (sub_summary["brand_count"] <= 2)
    ]
    .sort_values("avg_mrp", ascending=False)
    .head(15)
)
print(f"\nOpportunity 3 — Premium segment gaps (high avg price, <=2 brands):")
print(premium_gap.to_string(index=False))

# Top promo-overloaded subcategories
sub_disc = df.groupby("SubCategory")["DiscountPct"].mean().reset_index()
sub_disc.columns = ["SubCategory", "avg_discount_pct"]
high_promo = sub_disc.sort_values("avg_discount_pct", ascending=False).head(10)

opp_result = {
    "single_brand_subcategories": single_brand.to_dict(orient="records"),
    "low_discount_categories": [
        {**d, "avg_discount_pct": round(d["avg_discount_pct"], 2)}
        for d in low_disc.to_dict(orient="records")
    ],
    "high_promo_subcategories": [
        {**d, "avg_discount_pct": round(d["avg_discount_pct"], 2)}
        for d in high_promo.to_dict(orient="records")
    ],
    "premium_gap_subcategories": [
        {**d, "avg_mrp": round(d["avg_mrp"], 2)}
        for d in premium_gap.to_dict(orient="records")
    ],
    "global_avg_discount_pct": round(global_avg_disc, 2),
}
save_json(opp_result, "opportunities.json")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — TOP PRODUCTS
# ─────────────────────────────────────────────────────────────────────────────
print(SEPARATOR)
print("  STEP 9 — TOP PRODUCTS")
print(SEPARATOR)

top_discounted = (
    df.sort_values("DiscountPct", ascending=False)
    .head(20)[["Name", "Brand", "Category", "Price", "DiscountedPrice", "DiscountPct"]]
)
highest_mrp = (
    df.sort_values("Price", ascending=False)
    .head(20)[["Name", "Brand", "Category", "Price", "DiscountedPrice", "DiscountPct"]]
)

print("\nTop 10 Most Discounted Products:")
print(top_discounted.head(10).to_string(index=False))

print("\nTop 10 Highest MRP Products:")
print(highest_mrp.head(10).to_string(index=False))

save_json(
    {
        "top_discounted": top_discounted.to_dict(orient="records"),
        "highest_mrp":    highest_mrp.to_dict(orient="records"),
    },
    "top_products.json",
)
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — 5 KEY FINDINGS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(SEPARATOR)
print("  STEP 10 — 5 KEY FINDINGS")
print(SEPARATOR)

pc_count = cat_grp[cat_grp["Category"] == "Personal Care"]["product_count"].values
pf_count = cat_grp[cat_grp["Category"] == "Packaged Food"]["product_count"].values
hk_count = cat_grp[cat_grp["Category"] == "Home & Kitchen"]["product_count"].values

print(f"""
Finding 1 — Heavily Promotional Catalogue
  Average discount : {kpis['avg_discount_pct']}%
  Maximum discount : {kpis['max_discount_pct']}%
  DMart is a heavily promotion-driven retailer.

Finding 2 — Concentrated Category Mix
  Personal Care    : {pc_count[0] if len(pc_count) else 'N/A':,} SKUs
  Packaged Food    : {pf_count[0] if len(pf_count) else 'N/A':,} SKUs
  Home & Kitchen   : {hk_count[0] if len(hk_count) else 'N/A':,} SKUs
  3 categories hold ~63% of all SKUs.

Finding 3 — High Discount Exposure
  SKUs with >=40%  : {kpis['high_discount_products']:,} ({kpis['high_discount_pct_of_total']}% of catalogue)
  Significant margin exposure across the portfolio.

Finding 4 — Brand Coverage Gaps
  Unbranded SKUs   : {kpis['no_brand_products']:,}
  Single-brand subcats: {len(single_brand)}
  Data quality and assortment risks are material.

Finding 5 — Price Skew Towards Value
  Median MRP       : Rs {kpis['median_mrp']}
  Mean MRP         : Rs {kpis['avg_mrp']}
  Strong value-tier skew; limited premium range.
""")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 — 5 RECOMMENDED ACTIONS
# ─────────────────────────────────────────────────────────────────────────────
print(SEPARATOR)
print("  STEP 11 — 5 RECOMMENDED ACTIONS")
print(SEPARATOR)

actions = [
    (
        "Immediate", "Fix Extreme Discount Anomalies",
        f"Audit all {len(extreme)} products with >=70% discount. "
        "Implement a hard 60% discount ceiling in the catalogue system.",
        "Impact: 5/5 | Effort: 1/5",
    ),
    (
        "Immediate", "Assign Brand Data to All Unbranded SKUs",
        f"All {kpis['no_brand_products']} unbranded products must have a brand attributed "
        "within 30 days. Establish a data-entry SLA.",
        "Impact: 4/5 | Effort: 2/5",
    ),
    (
        "Short-Term", "Add Competing Brands to Single-Brand Subcategories",
        f"Prioritise the {len(single_brand)} subcategories with only one brand. "
        "Target at least one additional competing brand each.",
        "Impact: 4/5 | Effort: 3/5",
    ),
    (
        "Short-Term", "Launch Promotions in Under-Discounted Categories",
        f"Run targeted 2-week campaigns in the {len(low_disc)} categories averaging "
        f"below {round(global_avg_disc * 0.6, 1)}% discount.",
        "Impact: 3/5 | Effort: 2/5",
    ),
    (
        "Strategic", "Introduce a Tiered Discount Governance Policy",
        "Formalise: max 40% regular, 60% clearance, with senior sign-off above "
        "thresholds across the full 5,187-SKU portfolio.",
        "Impact: 5/5 | Effort: 4/5",
    ),
]

for i, (horizon, title, desc, score) in enumerate(actions, 1):
    print(f"\nAction {i} [{horizon}] — {title}")
    print(f"  {desc}")
    print(f"  {score}")

print()
print(SEPARATOR)
print("  ANALYSIS COMPLETE")
print(SEPARATOR)
print("  All JSON outputs written to outputs/")
print("  Open frontend/dashboard.html in a browser to view the dashboard.")
print(SEPARATOR)
