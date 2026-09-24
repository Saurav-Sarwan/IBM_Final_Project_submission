"""
DMart Product Catalogue Analysis - Backend Engine
==================================================
Performs data cleaning, feature engineering, KPI computation,
category/brand aggregation, discount analysis, anomaly detection,
and opportunity identification. Outputs all computed data to
outputs/ as JSON files consumed by the frontend dashboard.
"""

import pandas as pd
import numpy as np
import json
import os
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
CSV_PATH = "DMart.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_json(obj, filename: str) -> None:
    """Serialise obj to JSON and write to outputs/<filename>."""
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    print(f"  [saved] {path}")


# ─────────────────────────────────────────────
# 2. DATA LOADING & CLEANING
# ─────────────────────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, on_bad_lines="skip")

    # Normalise column names
    df.columns = df.columns.str.strip()

    # Coerce numeric columns
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["DiscountedPrice"] = pd.to_numeric(df["DiscountedPrice"], errors="coerce")

    # Drop rows where both prices are null or zero
    df = df[(df["Price"] > 0) & (df["DiscountedPrice"] > 0)].copy()

    # Fix negative or reversed pricing (discounted > price → swap)
    mask = df["DiscountedPrice"] > df["Price"]
    df.loc[mask, ["Price", "DiscountedPrice"]] = df.loc[
        mask, ["DiscountedPrice", "Price"]
    ].values

    # Feature engineering
    df["DiscountAmount"] = df["Price"] - df["DiscountedPrice"]
    df["DiscountPct"] = (df["DiscountAmount"] / df["Price"] * 100).round(2)
    df["PriceRange"] = pd.cut(
        df["Price"],
        bins=[0, 100, 300, 600, 1000, 5000, np.inf],
        labels=["₹0–100", "₹101–300", "₹301–600", "₹601–1K", "₹1K–5K", "₹5K+"],
    )

    # Clean brand
    df["Brand"] = df["Brand"].fillna("").str.strip()
    df["BrandLabel"] = df["Brand"].replace("", "No Brand")

    # Clean category / subcategory
    df["Category"] = df["Category"].fillna("Unknown").str.strip()
    df["SubCategory"] = df["SubCategory"].fillna("Unknown").str.strip()

    # Anomaly flags
    df["IsHighDiscount"] = df["DiscountPct"] >= 40
    df["IsZeroDiscount"] = df["DiscountPct"] == 0
    df["IsNoBrand"] = df["Brand"] == ""

    return df


# ─────────────────────────────────────────────
# 3. KPI SUMMARY
# ─────────────────────────────────────────────
def compute_kpis(df: pd.DataFrame) -> dict:
    total_products = len(df)
    total_brands = df[df["Brand"] != ""]["Brand"].nunique()
    total_categories = df["Category"].nunique()
    total_subcategories = df["SubCategory"].nunique()
    avg_price = round(df["Price"].mean(), 2)
    avg_discount_pct = round(df["DiscountPct"].mean(), 2)
    high_discount_count = int(df["IsHighDiscount"].sum())
    no_brand_count = int(df["IsNoBrand"].sum())
    zero_discount_count = int(df["IsZeroDiscount"].sum())
    max_discount_pct = round(df["DiscountPct"].max(), 2)
    median_price = round(df["Price"].median(), 2)

    return {
        "total_products": total_products,
        "total_brands": total_brands,
        "total_categories": total_categories,
        "total_subcategories": total_subcategories,
        "avg_mrp": avg_price,
        "median_mrp": median_price,
        "avg_discount_pct": avg_discount_pct,
        "max_discount_pct": max_discount_pct,
        "high_discount_products": high_discount_count,
        "high_discount_pct_of_total": round(high_discount_count / total_products * 100, 1),
        "no_brand_products": no_brand_count,
        "zero_discount_products": zero_discount_count,
    }


# ─────────────────────────────────────────────
# 4. CATEGORY ANALYSIS
# ─────────────────────────────────────────────
def category_analysis(df: pd.DataFrame) -> list:
    grp = (
        df.groupby("Category")
        .agg(
            product_count=("Name", "count"),
            brand_count=("Brand", lambda x: x[x != ""].nunique()),
            avg_mrp=("Price", "mean"),
            avg_discounted=("DiscountedPrice", "mean"),
            avg_discount_pct=("DiscountPct", "mean"),
            max_discount_pct=("DiscountPct", "max"),
            high_discount_count=("IsHighDiscount", "sum"),
        )
        .reset_index()
    )
    grp["avg_mrp"] = grp["avg_mrp"].round(2)
    grp["avg_discounted"] = grp["avg_discounted"].round(2)
    grp["avg_discount_pct"] = grp["avg_discount_pct"].round(2)
    grp["max_discount_pct"] = grp["max_discount_pct"].round(2)
    grp["high_discount_pct"] = (
        grp["high_discount_count"] / grp["product_count"] * 100
    ).round(1)
    grp = grp.sort_values("product_count", ascending=False)
    return grp.to_dict(orient="records")


# ─────────────────────────────────────────────
# 5. SUBCATEGORY ANALYSIS
# ─────────────────────────────────────────────
def subcategory_analysis(df: pd.DataFrame) -> list:
    grp = (
        df.groupby(["Category", "SubCategory"])
        .agg(
            product_count=("Name", "count"),
            brand_count=("Brand", lambda x: x[x != ""].nunique()),
            avg_mrp=("Price", "mean"),
            avg_discount_pct=("DiscountPct", "mean"),
            max_discount_pct=("DiscountPct", "max"),
        )
        .reset_index()
    )
    grp["avg_mrp"] = grp["avg_mrp"].round(2)
    grp["avg_discount_pct"] = grp["avg_discount_pct"].round(2)
    grp["max_discount_pct"] = grp["max_discount_pct"].round(2)
    grp = grp.sort_values("product_count", ascending=False)
    return grp.to_dict(orient="records")


# ─────────────────────────────────────────────
# 6. BRAND ANALYSIS
# ─────────────────────────────────────────────
def brand_analysis(df: pd.DataFrame, top_n: int = 30) -> dict:
    branded = df[df["Brand"] != ""]
    grp = (
        branded.groupby("Brand")
        .agg(
            product_count=("Name", "count"),
            category_count=("Category", "nunique"),
            avg_mrp=("Price", "mean"),
            avg_discount_pct=("DiscountPct", "mean"),
            max_discount_pct=("DiscountPct", "max"),
            high_discount_count=("IsHighDiscount", "sum"),
        )
        .reset_index()
    )
    grp["avg_mrp"] = grp["avg_mrp"].round(2)
    grp["avg_discount_pct"] = grp["avg_discount_pct"].round(2)
    grp["max_discount_pct"] = grp["max_discount_pct"].round(2)
    top_by_products = (
        grp.sort_values("product_count", ascending=False).head(top_n).to_dict(orient="records")
    )
    top_by_discount = (
        grp.sort_values("avg_discount_pct", ascending=False).head(top_n).to_dict(orient="records")
    )
    return {
        "top_by_products": top_by_products,
        "top_by_discount": top_by_discount,
        "total_branded_products": int(len(branded)),
        "single_product_brands": int((grp["product_count"] == 1).sum()),
    }


# ─────────────────────────────────────────────
# 7. DISCOUNT DISTRIBUTION
# ─────────────────────────────────────────────
def discount_distribution(df: pd.DataFrame) -> dict:
    bins = [0, 5, 10, 20, 30, 40, 50, 60, 75, 100]
    labels = [
        "0–5%", "5–10%", "10–20%", "20–30%",
        "30–40%", "40–50%", "50–60%", "60–75%", "75–100%",
    ]
    df["DiscBucket"] = pd.cut(df["DiscountPct"], bins=bins, labels=labels, include_lowest=True)
    dist = (
        df["DiscBucket"].value_counts().reindex(labels).fillna(0).astype(int).reset_index()
    )
    dist.columns = ["bucket", "count"]
    return dist.to_dict(orient="records")


# ─────────────────────────────────────────────
# 8. PRICE RANGE DISTRIBUTION
# ─────────────────────────────────────────────
def price_range_distribution(df: pd.DataFrame) -> list:
    dist = (
        df["PriceRange"].value_counts()
        .reindex(["₹0–100", "₹101–300", "₹301–600", "₹601–1K", "₹1K–5K", "₹5K+"])
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    dist.columns = ["range", "count"]
    return dist.to_dict(orient="records")


# ─────────────────────────────────────────────
# 9. ANOMALY DETECTION
# ─────────────────────────────────────────────
def detect_anomalies(df: pd.DataFrame) -> dict:
    # Extreme high discounts (≥70%)
    extreme_discount = (
        df[df["DiscountPct"] >= 70][
            ["Name", "Brand", "Category", "SubCategory", "Price", "DiscountedPrice", "DiscountPct"]
        ]
        .sort_values("DiscountPct", ascending=False)
        .head(30)
        .to_dict(orient="records")
    )

    # Zero discount products
    zero_discount = (
        df[df["IsZeroDiscount"]][
            ["Name", "Brand", "Category", "SubCategory", "Price"]
        ]
        .head(30)
        .to_dict(orient="records")
    )

    # No-brand products
    no_brand = (
        df[df["IsNoBrand"]][
            ["Name", "Category", "SubCategory", "Price", "DiscountPct"]
        ]
        .head(30)
        .to_dict(orient="records")
    )

    # Premium products (price > 2000) with very high discounts (>50%)
    premium_high_disc = (
        df[(df["Price"] > 2000) & (df["DiscountPct"] > 50)][
            ["Name", "Brand", "Category", "Price", "DiscountedPrice", "DiscountPct"]
        ]
        .sort_values("DiscountPct", ascending=False)
        .to_dict(orient="records")
    )

    return {
        "extreme_discount_products": extreme_discount,
        "zero_discount_products": zero_discount,
        "no_brand_products": no_brand,
        "premium_high_discount_products": premium_high_disc,
        "extreme_discount_count": len(extreme_discount),
        "zero_discount_count": len(zero_discount),
        "no_brand_count": len(no_brand),
    }


# ─────────────────────────────────────────────
# 10. OPPORTUNITIES
# ─────────────────────────────────────────────
def identify_opportunities(df: pd.DataFrame) -> dict:
    # Subcategories with only 1 brand → assortment gap
    sub_brand = df[df["Brand"] != ""].groupby("SubCategory")["Brand"].nunique().reset_index()
    sub_brand.columns = ["SubCategory", "brand_count"]
    single_brand_subcats = sub_brand[sub_brand["brand_count"] == 1].sort_values(
        "SubCategory"
    ).to_dict(orient="records")

    # Categories with below-average discount → pricing review opportunity
    cat_disc = df.groupby("Category")["DiscountPct"].mean().reset_index()
    cat_disc.columns = ["Category", "avg_discount_pct"]
    avg_global = df["DiscountPct"].mean()
    low_disc_cats = (
        cat_disc[cat_disc["avg_discount_pct"] < avg_global * 0.6]
        .sort_values("avg_discount_pct")
        .to_dict(orient="records")
    )

    # Top 10 subcategories by avg discount (promo over-reliance)
    sub_disc = df.groupby("SubCategory")["DiscountPct"].mean().reset_index()
    sub_disc.columns = ["SubCategory", "avg_discount_pct"]
    high_promo_subcats = (
        sub_disc.sort_values("avg_discount_pct", ascending=False).head(10).to_dict(orient="records")
    )

    # Subcategories with high avg price & low brand count → premium gap
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
        .to_dict(orient="records")
    )

    return {
        "single_brand_subcategories": single_brand_subcats,
        "low_discount_categories": [
            {**d, "avg_discount_pct": round(d["avg_discount_pct"], 2)} for d in low_disc_cats
        ],
        "high_promo_subcategories": [
            {**d, "avg_discount_pct": round(d["avg_discount_pct"], 2)} for d in high_promo_subcats
        ],
        "premium_gap_subcategories": [
            {**d, "avg_mrp": round(d["avg_mrp"], 2)} for d in premium_gap
        ],
        "global_avg_discount_pct": round(avg_global, 2),
    }


# ─────────────────────────────────────────────
# 11. TOP PRODUCTS
# ─────────────────────────────────────────────
def top_products(df: pd.DataFrame) -> dict:
    top_discounted = (
        df.sort_values("DiscountPct", ascending=False)
        .head(20)[["Name", "Brand", "Category", "Price", "DiscountedPrice", "DiscountPct"]]
        .to_dict(orient="records")
    )
    highest_mrp = (
        df.sort_values("Price", ascending=False)
        .head(20)[["Name", "Brand", "Category", "Price", "DiscountedPrice", "DiscountPct"]]
        .to_dict(orient="records")
    )
    return {
        "top_discounted": top_discounted,
        "highest_mrp": highest_mrp,
    }


# ─────────────────────────────────────────────
# 12. ORCHESTRATION
# ─────────────────────────────────────────────
def run_analysis() -> None:
    print("=" * 55)
    print("  DMart Product Catalogue Analysis - Backend Engine")
    print("=" * 55)

    print("\n[1/8] Loading and cleaning data ...")
    df = load_and_clean(CSV_PATH)
    print(f"      {len(df):,} valid product records loaded.")

    print("\n[2/8] Computing KPIs ...")
    kpis = compute_kpis(df)
    save_json(kpis, "kpis.json")

    print("\n[3/8] Category analysis ...")
    cat = category_analysis(df)
    save_json(cat, "category_analysis.json")

    print("\n[4/8] Subcategory analysis ...")
    subcat = subcategory_analysis(df)
    save_json(subcat, "subcategory_analysis.json")

    print("\n[5/8] Brand analysis ...")
    brand = brand_analysis(df)
    save_json(brand, "brand_analysis.json")

    print("\n[6/8] Discount and price distributions ...")
    disc_dist = discount_distribution(df)
    save_json(disc_dist, "discount_distribution.json")
    price_dist = price_range_distribution(df)
    save_json(price_dist, "price_distribution.json")

    print("\n[7/8] Anomaly detection ...")
    anomalies = detect_anomalies(df)
    save_json(anomalies, "anomalies.json")

    print("\n[8/8] Opportunity identification ...")
    opps = identify_opportunities(df)
    save_json(opps, "opportunities.json")

    top = top_products(df)
    save_json(top, "top_products.json")

    print("\n[DONE] Analysis complete. All outputs written to /outputs/")
    print("       Launch frontend/dashboard.html in a browser to view results.")


if __name__ == "__main__":
    run_analysis()
