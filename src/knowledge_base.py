import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.data_loader import get_data

load_dotenv()

VECTOR_STORE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store")


def _fmt_df(df, max_rows=None):
    """Format a DataFrame as a readable string."""
    if max_rows:
        return df.head(max_rows).to_string(index=False)
    return df.to_string(index=False)


def build_documents(df, summaries):
    """Convert aggregated summaries and raw data descriptions into LangChain Documents."""
    documents = []

    # 1. Dataset overview
    date_min = df["Date"].min().strftime("%Y-%m-%d")
    date_max = df["Date"].max().strftime("%Y-%m-%d")
    products = ", ".join(sorted(df["Product"].unique()))
    regions = ", ".join(sorted(df["Region"].unique()))

    overview_text = (
        f"Dataset Overview:\n"
        f"The sales dataset contains {len(df)} transactions from {date_min} to {date_max}.\n"
        f"Products: {products}\n"
        f"Regions: {regions}\n"
        f"Customer ages range from {int(df['Customer_Age'].min())} to {int(df['Customer_Age'].max())}.\n"
        f"Customer genders: {', '.join(sorted(df['Customer_Gender'].unique()))}\n"
        f"Customer satisfaction scores range from {df['Customer_Satisfaction'].min():.2f} to {df['Customer_Satisfaction'].max():.2f}."
    )
    documents.append(Document(page_content=overview_text, metadata={"dimension": "overview"}))

    # 2. Monthly sales trends
    monthly = summaries["monthly_sales"]
    monthly_text = (
        f"Monthly Sales Summary:\n"
        f"Sales data aggregated by month across all years.\n\n"
        f"{_fmt_df(monthly)}"
    )
    documents.append(Document(page_content=monthly_text, metadata={"dimension": "monthly_sales"}))

    # 3. Quarterly sales trends
    quarterly = summaries["quarterly_sales"]
    quarterly_text = (
        f"Quarterly Sales Summary:\n"
        f"Sales data aggregated by quarter across all years.\n\n"
        f"{_fmt_df(quarterly)}"
    )
    documents.append(Document(page_content=quarterly_text, metadata={"dimension": "quarterly_sales"}))

    # 4. Product performance
    by_product = summaries["by_product"]
    top_product = by_product.loc[by_product["Total_Sales"].idxmax(), "Product"]
    product_text = (
        f"Sales by Product:\n"
        f"Performance breakdown for each product. "
        f"The top-selling product is {top_product}.\n\n"
        f"{_fmt_df(by_product)}"
    )
    documents.append(Document(page_content=product_text, metadata={"dimension": "by_product"}))

    # 5. Regional performance
    by_region = summaries["by_region"]
    top_region = by_region.loc[by_region["Total_Sales"].idxmax(), "Region"]
    region_text = (
        f"Sales by Region:\n"
        f"Performance breakdown for each region. "
        f"The top-selling region is {top_region}.\n\n"
        f"{_fmt_df(by_region)}"
    )
    documents.append(Document(page_content=region_text, metadata={"dimension": "by_region"}))

    # 6. Customer demographics — age groups
    age_text = (
        f"Customer Demographics — Age Groups:\n"
        f"Customers binned into age groups with average sales and satisfaction.\n\n"
        f"{_fmt_df(summaries['age_bins'])}"
    )
    documents.append(Document(page_content=age_text, metadata={"dimension": "age_demographics"}))

    # 7. Customer demographics — gender split
    gender_text = (
        f"Customer Demographics — Gender Split:\n"
        f"Sales and satisfaction breakdown by customer gender.\n\n"
        f"{_fmt_df(summaries['gender_split'])}"
    )
    documents.append(Document(page_content=gender_text, metadata={"dimension": "gender_demographics"}))

    # 8. Overall statistical measures
    overall_stats = summaries["stats"]["overall"]
    stats_text = (
        f"Overall Statistical Measures:\n"
        f"Mean, median, standard deviation, min, and max for key numeric columns.\n\n"
        f"{_fmt_df(overall_stats)}"
    )
    documents.append(Document(page_content=stats_text, metadata={"dimension": "overall_stats"}))

    # 9. Stats by product
    stats_product = summaries["stats"]["by_product"]
    stats_product_text = (
        f"Statistical Measures by Product:\n"
        f"Detailed statistics (mean, median, std, min, max) for Sales, Customer_Age, "
        f"and Customer_Satisfaction broken down by product.\n\n"
        f"{_fmt_df(stats_product)}"
    )
    documents.append(Document(page_content=stats_product_text, metadata={"dimension": "stats_by_product"}))

    # 10. Stats by region
    stats_region = summaries["stats"]["by_region"]
    stats_region_text = (
        f"Statistical Measures by Region:\n"
        f"Detailed statistics (mean, median, std, min, max) for Sales, Customer_Age, "
        f"and Customer_Satisfaction broken down by region.\n\n"
        f"{_fmt_df(stats_region)}"
    )
    documents.append(Document(page_content=stats_region_text, metadata={"dimension": "stats_by_region"}))

    return documents


def create_vector_store(documents, persist_dir=VECTOR_STORE_DIR):
    """Embed documents and store in a FAISS vector store. Persists to disk."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(persist_dir)
    print(f"Vector store saved to {persist_dir} ({len(documents)} documents)")
    return vector_store


def load_vector_store(persist_dir=VECTOR_STORE_DIR):
    """Load a previously persisted FAISS vector store from disk."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.load_local(
        persist_dir, embeddings, allow_dangerous_deserialization=True
    )
    print(f"Vector store loaded from {persist_dir}")
    return vector_store


def get_or_create_vector_store(persist_dir=VECTOR_STORE_DIR):
    """Load vector store from disk if it exists, otherwise build and persist it."""
    if os.path.exists(os.path.join(persist_dir, "index.faiss")):
        return load_vector_store(persist_dir)

    df, summaries = get_data()
    documents = build_documents(df, summaries)
    return create_vector_store(documents, persist_dir)


if __name__ == "__main__":
    df, summaries = get_data()
    documents = build_documents(df, summaries)

    print(f"Built {len(documents)} documents:\n")
    for doc in documents:
        dim = doc.metadata["dimension"]
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  [{dim}] {preview}...")

    print("\nCreating FAISS vector store...")
    vs = create_vector_store(documents)

    print("\nTesting retrieval — query: 'What product has the highest sales?'")
    results = vs.similarity_search("What product has the highest sales?", k=3)
    for i, r in enumerate(results, 1):
        print(f"\n  Result {i} [{r.metadata['dimension']}]:")
        print(f"  {r.page_content[:120]}...")
