import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.data_loader import load_and_prepare, compute_summaries
from src.visualizations import (
    plot_monthly_sales,
    plot_quarterly_sales,
    plot_product_sales,
    plot_region_sales,
    plot_age_distribution,
    plot_gender_split,
    plot_satisfaction_by_product,
    plot_satisfaction_by_region,
)
from src.memory import ChatSession
from src.chains import summarize, analyze
from src.evaluation import evaluate, GROUND_TRUTH

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="InsightForge", page_icon="📊", layout="wide")
st.title("📊 InsightForge — AI-Powered Business Intelligence")


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = load_and_prepare()
    summaries = compute_summaries(df)
    return df, summaries


df_full, summaries_full = load_data()


# ---------------------------------------------------------------------------
# Sidebar — dataset overview & filters
# ---------------------------------------------------------------------------
st.sidebar.header("Dataset Filters")

date_min = df_full["Date"].min().date()
date_max = df_full["Date"].max().date()
date_range = st.sidebar.date_input("Date range", value=(date_min, date_max), min_value=date_min, max_value=date_max)

products = st.sidebar.multiselect("Products", options=sorted(df_full["Product"].unique()), default=sorted(df_full["Product"].unique()))
regions = st.sidebar.multiselect("Regions", options=sorted(df_full["Region"].unique()), default=sorted(df_full["Region"].unique()))

# Apply filters
df = df_full.copy()
if len(date_range) == 2:
    df = df[(df["Date"].dt.date >= date_range[0]) & (df["Date"].dt.date <= date_range[1])]
df = df[df["Product"].isin(products)]
df = df[df["Region"].isin(regions)]
summaries = compute_summaries(df)

# Model selector
st.sidebar.markdown("---")
st.sidebar.subheader("AI Model")
model = st.sidebar.selectbox("Choose LLM", ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"], index=0)

# Reset chat session when model changes
if "current_model" not in st.session_state:
    st.session_state.current_model = model
if st.session_state.current_model != model:
    st.session_state.current_model = model
    st.session_state.chat_session = ChatSession(model=model)
    st.session_state.messages = []

# Sidebar stats
st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Overview")
st.sidebar.metric("Total Transactions", f"{len(df):,}")
st.sidebar.metric("Date Range", f"{date_min} to {date_max}")
st.sidebar.metric("Products", len(products))
st.sidebar.metric("Regions", len(regions))


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Dashboard", "💬 Chat with Data", "🧠 AI Insights", "✅ Evaluation"])


# ---------------------------------------------------------------------------
# Tab 1 — Dashboard
# ---------------------------------------------------------------------------
with tab1:
    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
    col2.metric("Avg Satisfaction", f"{df['Customer_Satisfaction'].mean():.2f} / 5")

    top_product = summaries["by_product"].loc[summaries["by_product"]["Total_Sales"].idxmax(), "Product"] if not summaries["by_product"].empty else "N/A"
    col3.metric("Top Product", top_product)

    top_region = summaries["by_region"].loc[summaries["by_region"]["Total_Sales"].idxmax(), "Region"] if not summaries["by_region"].empty else "N/A"
    col4.metric("Top Region", top_region)

    st.markdown("---")

    # Charts row 1: trends
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_monthly_sales(summaries), use_container_width=True)
    with c2:
        st.plotly_chart(plot_quarterly_sales(summaries), use_container_width=True)

    # Charts row 2: product & region
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(plot_product_sales(summaries), use_container_width=True)
    with c4:
        st.plotly_chart(plot_region_sales(summaries), use_container_width=True)

    # Charts row 3: demographics
    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(plot_age_distribution(df), use_container_width=True)
    with c6:
        st.plotly_chart(plot_gender_split(summaries), use_container_width=True)

    # Charts row 4: satisfaction
    c7, c8 = st.columns(2)
    with c7:
        st.plotly_chart(plot_satisfaction_by_product(df), use_container_width=True)
    with c8:
        st.plotly_chart(plot_satisfaction_by_region(df), use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2 — Chat with Data
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Chat with Your Data")
    st.caption("Ask questions about the sales dataset. The AI uses RAG to ground answers in real data.")

    # Initialize chat session in session state
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = ChatSession(model=model)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = st.session_state.chat_session.ask(prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.chat_session.clear()
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Tab 3 — AI Insights
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("AI-Generated Insights")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Executive Summary")
        if st.button("Generate Executive Summary", key="gen_summary"):
            with st.spinner("Generating summary..."):
                summary_text = summarize(model=model)
            st.session_state["exec_summary"] = summary_text

        if "exec_summary" in st.session_state:
            st.markdown(st.session_state["exec_summary"])

    with col_b:
        st.markdown("#### Deep Analysis")
        topic = st.text_input("Analysis topic:", placeholder="e.g., Compare regional performance trends")
        if st.button("Run Analysis", key="gen_analysis") and topic:
            with st.spinner("Running analysis..."):
                analysis_text = analyze(topic, model=model)
            st.session_state["analysis"] = analysis_text

        if "analysis" in st.session_state:
            st.markdown(st.session_state["analysis"])


# ---------------------------------------------------------------------------
# Tab 4 — Evaluation
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Model Evaluation — QAEvalChain")
    st.caption("Tests the RAG pipeline against 15 ground-truth Q&A pairs with known answers.")

    if st.button("Run Evaluation", key="run_eval"):
        status_box = st.empty()
        progress = st.progress(0, text="Starting evaluation...")
        total_q = len(GROUND_TRUTH)

        # Phase 1: Generate predictions
        from src.chains import ask, get_llm
        predictions = []
        for i, qa in enumerate(GROUND_TRUTH):
            status_box.info(f"Predicting [{i+1}/{total_q}]: {qa['question'][:60]}...")
            try:
                pred = ask(qa["question"], model=model)
                predictions.append({"question": qa["question"], "result": pred})
                status_box.success(f"Prediction [{i+1}/{total_q}] done ({len(pred)} chars)")
            except Exception as e:
                status_box.error(f"Prediction [{i+1}/{total_q}] FAILED: {e}")
                predictions.append({"question": qa["question"], "result": f"ERROR: {e}"})
            progress.progress((i + 1) / (total_q * 2), text=f"Predicting {i+1}/{total_q}...")
            import time; time.sleep(2)

        # Phase 2: Grade predictions
        from langchain.evaluation.qa import QAEvalChain
        status_box.info("Building eval chain...")
        eval_llm = get_llm(model, temperature=0)
        eval_chain = QAEvalChain.from_llm(eval_llm)

        detailed = []
        correct = 0
        for i, (qa, pred) in enumerate(zip(GROUND_TRUTH, predictions)):
            status_box.info(f"Grading [{i+1}/{total_q}]: {qa['question'][:60]}...")
            try:
                example = [{"query": qa["question"], "answer": qa["answer"]}]
                res = eval_chain.evaluate(example, [pred])
                grade = res[0]["results"].strip().upper()
                status_box.success(f"Grade [{i+1}/{total_q}]: {grade}")
            except Exception as e:
                status_box.error(f"Grade [{i+1}/{total_q}] FAILED: {e}")
                grade = "ERROR"
            is_correct = "CORRECT" in grade
            if is_correct:
                correct += 1
            detailed.append({
                "question": qa["question"],
                "ground_truth": qa["answer"],
                "prediction": pred["result"],
                "grade": grade,
                "correct": is_correct,
            })
            progress.progress((total_q + i + 1) / (total_q * 2), text=f"Grading {i+1}/{total_q}...")
            import time; time.sleep(2)

        progress.empty()
        status_box.empty()
        total = len(GROUND_TRUTH)
        accuracy = correct / total if total > 0 else 0
        results = {
            "total": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": accuracy,
            "detailed": detailed,
        }
        st.session_state["eval_results"] = results

    if "eval_results" in st.session_state:
        results = st.session_state["eval_results"]

        # Summary metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Questions", results["total"])
        m2.metric("Correct", results["correct"])
        m3.metric("Accuracy", f"{results['accuracy']:.0%}")

        # Detailed results table
        st.markdown("---")
        detail_rows = []
        for i, d in enumerate(results["detailed"], 1):
            detail_rows.append({
                "#": i,
                "Status": "✅" if d["correct"] else "❌",
                "Question": d["question"],
                "Expected Answer": d["ground_truth"],
                "Model Answer": d["prediction"],
                "Grade": d["grade"],
            })
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
