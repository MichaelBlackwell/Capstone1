import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Sales trends over time
# ---------------------------------------------------------------------------

def plot_monthly_sales(summaries):
    """Line chart of monthly sales trends."""
    df = summaries["monthly_sales"].copy()
    df["Period"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01")
    fig = px.line(
        df, x="Period", y="Total_Sales",
        title="Monthly Sales Trends",
        labels={"Total_Sales": "Total Sales ($)", "Period": "Month"},
        markers=True,
    )
    fig.update_layout(hovermode="x unified")
    return fig


def plot_quarterly_sales(summaries):
    """Bar chart of quarterly sales."""
    df = summaries["quarterly_sales"].copy()
    df["Period"] = df["Year"].astype(str) + " Q" + df["Quarter"].astype(str)
    fig = px.bar(
        df, x="Period", y="Total_Sales",
        title="Quarterly Sales",
        labels={"Total_Sales": "Total Sales ($)", "Period": "Quarter"},
        color="Total_Sales",
        color_continuous_scale="Blues",
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


# ---------------------------------------------------------------------------
# 2. Product performance comparison
# ---------------------------------------------------------------------------

def plot_product_sales(summaries):
    """Bar chart comparing total sales by product."""
    df = summaries["by_product"]
    fig = px.bar(
        df, x="Product", y="Total_Sales",
        title="Total Sales by Product",
        labels={"Total_Sales": "Total Sales ($)"},
        color="Product",
        text="Total_Sales",
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# 3. Regional analysis
# ---------------------------------------------------------------------------

def plot_region_sales(summaries):
    """Bar chart comparing total sales by region."""
    df = summaries["by_region"]
    fig = px.bar(
        df, x="Region", y="Total_Sales",
        title="Total Sales by Region",
        labels={"Total_Sales": "Total Sales ($)"},
        color="Region",
        text="Total_Sales",
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# 4. Customer demographics
# ---------------------------------------------------------------------------

def plot_age_distribution(df):
    """Histogram of customer age distribution."""
    fig = px.histogram(
        df, x="Customer_Age", nbins=20,
        title="Customer Age Distribution",
        labels={"Customer_Age": "Age", "count": "Number of Customers"},
        color_discrete_sequence=["#636EFA"],
    )
    fig.update_layout(bargap=0.05)
    return fig


def plot_gender_split(summaries):
    """Pie chart of gender split by transaction count."""
    df = summaries["gender_split"]
    fig = px.pie(
        df, values="Count", names="Customer_Gender",
        title="Customer Gender Split",
        color_discrete_sequence=["#EF553B", "#636EFA"],
    )
    fig.update_traces(textinfo="label+percent+value")
    return fig


# ---------------------------------------------------------------------------
# 5. Customer satisfaction
# ---------------------------------------------------------------------------

def plot_satisfaction_by_product(df):
    """Box plot of customer satisfaction scores by product."""
    fig = px.box(
        df, x="Product", y="Customer_Satisfaction",
        title="Customer Satisfaction by Product",
        labels={"Customer_Satisfaction": "Satisfaction Score"},
        color="Product",
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_satisfaction_by_region(df):
    """Box plot of customer satisfaction scores by region."""
    fig = px.box(
        df, x="Region", y="Customer_Satisfaction",
        title="Customer Satisfaction by Region",
        labels={"Customer_Satisfaction": "Satisfaction Score"},
        color="Region",
    )
    fig.update_layout(showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# Convenience: all charts at once
# ---------------------------------------------------------------------------

def get_all_figures(df, summaries):
    """Return a dict of all visualization figures."""
    return {
        "monthly_sales": plot_monthly_sales(summaries),
        "quarterly_sales": plot_quarterly_sales(summaries),
        "product_sales": plot_product_sales(summaries),
        "region_sales": plot_region_sales(summaries),
        "age_distribution": plot_age_distribution(df),
        "gender_split": plot_gender_split(summaries),
        "satisfaction_by_product": plot_satisfaction_by_product(df),
        "satisfaction_by_region": plot_satisfaction_by_region(df),
    }


if __name__ == "__main__":
    from src.data_loader import get_data

    df, summaries = get_data()
    figures = get_all_figures(df, summaries)

    print(f"Generated {len(figures)} charts:")
    for name, fig in figures.items():
        print(f"  - {name}: {fig.layout.title.text}")

    # Save one as HTML to verify rendering
    figures["monthly_sales"].write_html("test_chart.html")
    print("\nSaved test_chart.html — open in browser to verify.")
