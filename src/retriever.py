import os
import re
from typing import List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd
from dotenv import load_dotenv
from langchain.schema import BaseRetriever, Document

from src.data_loader import get_data
from src.knowledge_base import get_or_create_vector_store

load_dotenv()

# Maps quarter aliases to integers
QUARTER_MAP = {"q1": 1, "q2": 2, "q3": 3, "q4": 4}


def _parse_query_intent(query: str) -> dict:
    """Extract structured intent (product, region, year, quarter, month) from a natural-language query."""
    q = query.lower()
    intent = {}

    # Year
    year_match = re.search(r"\b(20\d{2})\b", q)
    if year_match:
        intent["year"] = int(year_match.group(1))

    # Quarter
    quarter_match = re.search(r"\b(q[1-4])\b", q)
    if quarter_match:
        intent["quarter"] = QUARTER_MAP[quarter_match.group(1)]

    # Month (name)
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    for name, num in months.items():
        if name in q:
            intent["month"] = num
            break

    # Product
    product_match = re.search(r"widget\s*[a-d]", q)
    if product_match:
        intent["product"] = product_match.group(0).title()

    # Region
    for region in ["north", "south", "east", "west"]:
        if region in q:
            intent["region"] = region.title()
            break

    # Gender
    if "female" in q:
        intent["gender"] = "Female"
    elif "male" in q:
        intent["gender"] = "Male"

    return intent


def _run_pandas_query(df: pd.DataFrame, intent: dict) -> str | None:
    """Run a live pandas query based on parsed intent. Returns a text summary or None."""
    if not intent:
        return None

    filtered = df.copy()

    if "year" in intent:
        filtered = filtered[filtered["Year"] == intent["year"]]
    if "quarter" in intent:
        filtered = filtered[filtered["Quarter"] == intent["quarter"]]
    if "month" in intent:
        filtered = filtered[filtered["Month"] == intent["month"]]
    if "product" in intent:
        filtered = filtered[filtered["Product"] == intent["product"]]
    if "region" in intent:
        filtered = filtered[filtered["Region"] == intent["region"]]
    if "gender" in intent:
        filtered = filtered[filtered["Customer_Gender"] == intent["gender"]]

    if filtered.empty:
        return None

    # Build a descriptive filter label
    filters = []
    for key in ["product", "region", "year", "quarter", "month", "gender"]:
        if key in intent:
            filters.append(f"{key.title()}={intent[key]}")
    label = ", ".join(filters)

    total = filtered["Sales"].sum()
    avg = filtered["Sales"].mean()
    count = len(filtered)
    avg_sat = filtered["Customer_Satisfaction"].mean()

    text = (
        f"Live Data Query Result (filters: {label}):\n"
        f"  Transactions: {count}\n"
        f"  Total Sales: {total:,.0f}\n"
        f"  Average Sales: {avg:,.2f}\n"
        f"  Average Customer Satisfaction: {avg_sat:.2f}\n"
    )

    # Add top product/region breakdowns if not already filtered
    if "product" not in intent and count > 0:
        top = filtered.groupby("Product")["Sales"].sum().sort_values(ascending=False)
        text += f"  Sales by Product: {dict(top)}\n"

    if "region" not in intent and count > 0:
        top = filtered.groupby("Region")["Sales"].sum().sort_values(ascending=False)
        text += f"  Sales by Region: {dict(top)}\n"

    return text


class HybridRetriever(BaseRetriever):
    """Retriever that combines FAISS vector search with live pandas queries."""

    vector_store: object
    df: pd.DataFrame
    k: int = 3

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs = []

        # 1. Live pandas query for precise numbers
        intent = _parse_query_intent(query)
        pandas_result = _run_pandas_query(self.df, intent)
        if pandas_result:
            docs.append(Document(
                page_content=pandas_result,
                metadata={"source": "live_pandas_query"},
            ))

        # 2. FAISS vector similarity search for pre-computed context
        vector_docs = self.vector_store.similarity_search(query, k=self.k)
        docs.extend(vector_docs)

        return docs


def get_retriever(k=3):
    """Build and return a HybridRetriever ready for use in chains."""
    df, _ = get_data()
    vector_store = get_or_create_vector_store()
    return HybridRetriever(vector_store=vector_store, df=df, k=k)


if __name__ == "__main__":
    retriever = get_retriever()

    test_queries = [
        "What are total sales in Q1 2023?",
        "Which product has the highest sales?",
        "How do female customers compare to male customers?",
        "What is the average satisfaction in the North region?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        results = retriever.invoke(query)
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", doc.metadata.get("dimension", "unknown"))
            print(f"\n  [{i}] Source: {source}")
            print(f"  {doc.page_content[:150]}...")
