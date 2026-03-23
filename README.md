# InsightForge — AI-Powered Business Intelligence Assistant

A conversational BI tool that analyzes sales data using LangChain, OpenAI, and RAG, served through a Streamlit dashboard.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-key-here
```

## Build the Knowledge Base

Run once to embed the data summaries into a FAISS vector store:

```bash
python -m src.knowledge_base
```

## Launch the App

```bash
python -m streamlit run app.py
```

Opens at `http://localhost:8501` with 4 tabs:

- **Dashboard** — KPIs and interactive charts with sidebar filters
- **Chat with Data** — Ask natural-language questions (supports follow-ups)
- **AI Insights** — One-click executive summary and deep analysis
- **Evaluation** — Run 15 ground-truth Q&A tests against the RAG pipeline

## Project Structure

```
app.py                  # Streamlit UI
src/
  data_loader.py        # Data loading & preparation
  knowledge_base.py     # FAISS vector store creation
  retriever.py          # Hybrid retriever (vector + live pandas)
  chains.py             # LLM prompt templates & chains
  memory.py             # Conversation memory
  evaluation.py         # QAEvalChain evaluation
  visualizations.py     # Plotly charts
sales_data.csv          # Source dataset (2,500 rows)
report/
  capstone_report.md    # Written report
```
