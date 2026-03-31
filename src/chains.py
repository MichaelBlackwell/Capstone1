import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import StrOutputParser

from src.retriever import get_retriever
from src.knowledge_base import get_or_create_vector_store
from src.data_loader import get_data

load_dotenv()


def get_llm(model="llama-3.3-70b-versatile", temperature=0.3):
    """Return a ChatGroq instance."""
    return ChatGroq(model=model, temperature=temperature)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """You are a business intelligence analyst. Using the data context below, write a concise
executive summary of the sales dataset. Cover overall performance, top products, regional
highlights, customer demographics, and notable trends. Use specific numbers.

Data Context:
{context}

Executive Summary:"""
)

QA_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful business data assistant. Answer the user's question using ONLY the
data context provided. If the context contains specific numbers, cite them. If you cannot
answer from the context, say so.

Data Context:
{context}

Question: {question}

Answer:"""
)

ANALYSIS_PROMPT = ChatPromptTemplate.from_template(
    """You are a senior data analyst. Using the data context below, produce a deeper analysis
addressing the user's topic. Include trend identification, segment comparisons, and
actionable recommendations. Support every claim with numbers from the context.

Data Context:
{context}

Analysis Topic: {question}

Detailed Analysis:"""
)


# ---------------------------------------------------------------------------
# Chain builders
# ---------------------------------------------------------------------------

def build_qa_chain(model="llama-3.3-70b-versatile"):
    """Build a RetrievalQA chain linking the hybrid retriever to the QA prompt and LLM."""
    llm = get_llm(model)
    retriever = get_retriever(k=3)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT},
    )
    return chain


def build_summary_chain(model="llama-3.3-70b-versatile"):
    """Build a chain that generates an executive summary from all key data documents."""
    llm = get_llm(model)

    def run(_input=None):
        # Gather context from all summary dimensions via the vector store
        df, summaries = get_data()
        retriever = get_retriever(k=10)
        docs = retriever.invoke("overall sales performance summary trends products regions demographics")
        context = "\n\n".join(doc.page_content for doc in docs)

        chain = SUMMARY_PROMPT | llm | StrOutputParser()
        return chain.invoke({"context": context})

    return run


def build_analysis_chain(model="llama-3.3-70b-versatile"):
    """Build a chain for deeper analysis on a user-specified topic."""
    llm = get_llm(model)
    retriever = get_retriever(k=4)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": ANALYSIS_PROMPT},
    )
    return chain


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def ask(question: str, model="llama-3.3-70b-versatile") -> str:
    """Ask a question and get a grounded answer."""
    chain = build_qa_chain(model)
    result = chain.invoke({"query": question})
    return result["result"]


def summarize(model="llama-3.3-70b-versatile") -> str:
    """Generate an executive summary."""
    chain = build_summary_chain(model)
    return chain()


def analyze(topic: str, model="llama-3.3-70b-versatile") -> str:
    """Produce a deeper analysis on a given topic."""
    chain = build_analysis_chain(model)
    result = chain.invoke({"query": topic})
    return result["result"]


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: Q&A Chain")
    print("=" * 60)
    answer = ask("What product has the highest total sales and by how much?")
    print(answer)

    print("\n" + "=" * 60)
    print("TEST 2: Executive Summary")
    print("=" * 60)
    summary = summarize()
    print(summary)

    print("\n" + "=" * 60)
    print("TEST 3: Deep Analysis")
    print("=" * 60)
    analysis = analyze("Compare regional sales performance and identify underperforming areas")
    print(analysis)
