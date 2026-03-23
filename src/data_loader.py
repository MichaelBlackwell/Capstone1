import os
import pandas as pd
import numpy as np


def load_and_prepare(csv_path="sales_data.csv"):
    """Load sales_data.csv, parse dates, extract time features, and clean data."""
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), csv_path)

    df = pd.read_csv(csv_path, parse_dates=["Date"])

    # Extract time features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.day_name()

    # Add quarter for convenience
    df["Quarter"] = df["Date"].dt.quarter

    # Verify and fix dtypes
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df["Customer_Age"] = pd.to_numeric(df["Customer_Age"], errors="coerce")
    df["Customer_Satisfaction"] = pd.to_numeric(df["Customer_Satisfaction"], errors="coerce")

    # Report and handle nulls
    null_counts = df.isnull().sum()
    if null_counts.any():
        print("Null values found:")
        print(null_counts[null_counts > 0])
        df = df.dropna(subset=["Date", "Sales"])
        df["Customer_Age"] = df["Customer_Age"].fillna(df["Customer_Age"].median())
        df["Customer_Satisfaction"] = df["Customer_Satisfaction"].fillna(df["Customer_Satisfaction"].median())

    return df


def compute_summaries(df):
    """Compute pre-aggregated summaries across multiple dimensions."""
    summaries = {}

    # --- Sales by time period ---
    summaries["monthly_sales"] = (
        df.groupby(["Year", "Month"])["Sales"]
        .agg(["sum", "mean", "count"])
        .reset_index()
        .rename(columns={"sum": "Total_Sales", "mean": "Avg_Sales", "count": "Num_Transactions"})
    )

    summaries["quarterly_sales"] = (
        df.groupby(["Year", "Quarter"])["Sales"]
        .agg(["sum", "mean", "count"])
        .reset_index()
        .rename(columns={"sum": "Total_Sales", "mean": "Avg_Sales", "count": "Num_Transactions"})
    )

    # --- Sales by product ---
    summaries["by_product"] = (
        df.groupby("Product")
        .agg(
            Total_Sales=("Sales", "sum"),
            Avg_Sales=("Sales", "mean"),
            Num_Transactions=("Sales", "count"),
            Avg_Satisfaction=("Customer_Satisfaction", "mean"),
        )
        .reset_index()
    )

    # --- Sales by region ---
    summaries["by_region"] = (
        df.groupby("Region")
        .agg(
            Total_Sales=("Sales", "sum"),
            Avg_Sales=("Sales", "mean"),
            Num_Transactions=("Sales", "count"),
            Avg_Satisfaction=("Customer_Satisfaction", "mean"),
        )
        .reset_index()
    )

    # --- Customer demographics: age bins ---
    age_bins = [0, 25, 35, 45, 55, 100]
    age_labels = ["18-25", "26-35", "36-45", "46-55", "56+"]
    df["Age_Group"] = pd.cut(df["Customer_Age"], bins=age_bins, labels=age_labels, right=True)

    summaries["age_bins"] = (
        df.groupby("Age_Group", observed=False)
        .agg(
            Count=("Customer_Age", "count"),
            Avg_Sales=("Sales", "mean"),
            Avg_Satisfaction=("Customer_Satisfaction", "mean"),
        )
        .reset_index()
    )

    # --- Customer demographics: gender split ---
    summaries["gender_split"] = (
        df.groupby("Customer_Gender")
        .agg(
            Count=("Customer_Gender", "count"),
            Avg_Sales=("Sales", "mean"),
            Avg_Satisfaction=("Customer_Satisfaction", "mean"),
        )
        .reset_index()
    )

    # --- Statistical measures ---
    numeric_cols = ["Sales", "Customer_Age", "Customer_Satisfaction"]

    overall_stats = df[numeric_cols].agg(["mean", "median", "std", "min", "max"]).T
    overall_stats.index.name = "Metric"
    overall_stats = overall_stats.reset_index()

    stats_by_product = (
        df.groupby("Product")[numeric_cols]
        .agg(["mean", "median", "std", "min", "max"])
    )
    stats_by_product.columns = ["_".join(col) for col in stats_by_product.columns]
    stats_by_product = stats_by_product.reset_index()

    stats_by_region = (
        df.groupby("Region")[numeric_cols]
        .agg(["mean", "median", "std", "min", "max"])
    )
    stats_by_region.columns = ["_".join(col) for col in stats_by_region.columns]
    stats_by_region = stats_by_region.reset_index()

    summaries["stats"] = {
        "overall": overall_stats,
        "by_product": stats_by_product,
        "by_region": stats_by_region,
    }

    return summaries


def get_data(csv_path="sales_data.csv"):
    """Load data and compute all summaries. Returns (df, summaries)."""
    df = load_and_prepare(csv_path)
    summaries = compute_summaries(df)
    return df, summaries


if __name__ == "__main__":
    df, summaries = get_data()
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSummary keys: {list(summaries.keys())}")

    print("\n--- Monthly Sales (first 5 rows) ---")
    print(summaries["monthly_sales"].head())

    print("\n--- Sales by Product ---")
    print(summaries["by_product"])

    print("\n--- Sales by Region ---")
    print(summaries["by_region"])

    print("\n--- Age Group Distribution ---")
    print(summaries["age_bins"])

    print("\n--- Gender Split ---")
    print(summaries["gender_split"])

    print("\n--- Overall Stats ---")
    print(summaries["stats"]["overall"])
