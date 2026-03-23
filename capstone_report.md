# InsightForge: An AI-Powered Business Intelligence Assistant

## 1. Introduction & Problem Statement

Modern businesses generate vast amounts of sales data, yet extracting actionable insights remains a manual, time-consuming process that often requires specialized data analysts. Small and mid-size organizations frequently lack the resources to maintain dedicated analytics teams, leaving valuable patterns hidden in their data.

**InsightForge** addresses this gap by providing an AI-powered business intelligence assistant that enables non-technical users to explore, analyze, and derive insights from sales data through natural language conversation. The system combines Retrieval-Augmented Generation (RAG) with pre-computed analytics to deliver accurate, data-grounded responses via an intuitive Streamlit interface.

### Objectives

1. Build a RAG pipeline that grounds LLM responses in real sales data to minimize hallucination
2. Support multi-turn conversational analysis with memory so users can ask follow-up questions
3. Provide interactive visualizations covering sales trends, product performance, regional analysis, and customer demographics
4. Evaluate system accuracy against a set of ground-truth question-answer pairs
5. Deliver the solution as a self-contained web application accessible to business users

### Dataset

The project uses a synthetic sales dataset (`sales_data.csv`) containing 2,500 transactions with the following attributes:

| Column | Type | Description |
|--------|------|-------------|
| Date | datetime | Transaction date (2022-01-01 to 2028-10-28) |
| Product | categorical | Widget A, B, C, or D |
| Region | categorical | North, South, East, or West |
| Sales | integer | Transaction amount ($100–$999) |
| Customer_Age | integer | Customer age (18–69) |
| Customer_Gender | categorical | Male or Female |
| Customer_Satisfaction | float | Satisfaction score (1.0–5.0) |

---

## 2. Literature Review

This project draws on research from four domains: business intelligence systems, sales data analytics, time-series prediction, and AI-driven business innovation.

### 2.1 Business Intelligence Concepts and Approaches

The systematic review by the authors of "Review Study: Business Intelligence Concepts and Approaches" classifies BI approaches into three categories: managerial, technical, and system enablers. The paper emphasizes that effective BI systems must integrate data warehousing, analytics, and user-facing reporting tools into a cohesive pipeline. InsightForge follows this framework by implementing a structured data preparation layer, an analytics engine (LLM + RAG), and a presentation layer (Streamlit dashboard). The review's finding that BI adoption improves decision-making quality directly motivates our natural-language interface design, which lowers the barrier to accessing analytical insights.

### 2.2 Walmart's Sales Data Analysis — A Big Data Analytics Perspective

This study applies Spark and MapReduce to Walmart's sales datasets to understand customer behavior and predict sales across geographical locations. The authors demonstrate that aggregating sales by region, time period, and product category reveals patterns invisible in raw transaction logs. InsightForge adopts this dimensional analysis approach in its data preparation step, pre-computing summaries by product, region, time period, and customer demographics. However, where the Walmart study relies on big data infrastructure, InsightForge achieves similar analytical coverage using pandas on a smaller dataset, making it accessible without distributed computing resources.

### 2.3 Time Series Data Prediction using IoT and Machine Learning

Research on predicting time-series data using machine learning regression models informs our approach to temporal sales analysis. The study demonstrates that structured feature extraction from time-series data (extracting year, month, day, and derived features) significantly improves model performance. InsightForge applies this principle by parsing transaction dates into Year, Month, Day, Weekday, and Quarter features, enabling the RAG system to answer temporal queries (e.g., "What are Q1 2023 sales?") with precision. While InsightForge does not implement predictive modeling, the feature engineering approach directly supports accurate historical trend analysis.

### 2.4 AI-Driven Business Model Innovation

The systematic review of 180 articles on AI and business model innovation establishes that AI transforms value creation by automating insight extraction and enabling data-driven decision-making. The authors identify conversational AI as a key enabler for democratizing access to business analytics. InsightForge embodies this vision by wrapping a RAG-powered LLM in a conversational interface, allowing business users to query their data without writing SQL or Python. The paper's emphasis on grounding AI outputs in verifiable data aligns with our hybrid retriever design, which combines vector search with live pandas computations to ensure factual accuracy.

---

## 3. System Architecture & Design Decisions

### 3.1 Architecture Overview

```
                    +------------------+
                    |   Streamlit UI   |
                    |   (app.py)       |
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
     +--------v---+  +------v------+  +----v--------+
     |  Dashboard |  | Chat (RAG)  |  | AI Insights |
     |  Tab 1     |  | Tab 2       |  | Tab 3       |
     +--------+---+  +------+------+  +----+--------+
              |              |              |
              |     +--------v--------+     |
              |     | Conversational   |    |
              |     | Retrieval Chain  |    |
              |     | + Memory         |    |
              |     +--------+--------+    |
              |              |             |
              |     +--------v--------+    |
              |     | Hybrid Retriever |   |
              |     +---+--------+----+   |
              |         |        |        |
         +----v----+ +--v---+ +-v------+ |
         | Plotly   | |FAISS | |Pandas  | |
         | Charts   | |Vector| |Live    | |
         |          | |Store | |Query   | |
         +----+----+ +--+---+ +-+------+ |
              |         |       |         |
              +----+----+---+---+---------+
                   |        |
            +------v---+ +--v-----------+
            | Data     | | Knowledge    |
            | Loader   | | Base Builder |
            +------+---+ +--+-----------+
                   |         |
               +---v---------v---+
               | sales_data.csv  |
               +-----------------+
```

### 3.2 Key Design Decisions

**RAG over fine-tuning.** Rather than fine-tuning an LLM on the sales data (which would be expensive and inflexible), we use Retrieval-Augmented Generation. The LLM receives relevant data context at query time, ensuring answers are grounded in actual numbers and the system can adapt to new data without retraining.

**Hybrid retriever.** A pure vector search approach would retrieve semantically relevant summary documents but might miss precise numerical answers. Our hybrid retriever first attempts a live pandas query (parsing the user's question for filters like year, quarter, product, region) and then appends vector search results. This ensures the LLM always has access to exact figures.

**Pre-computed summaries as documents.** Converting aggregated statistics into natural-language document chunks (one per analytical dimension) makes them semantically searchable. This is more effective than embedding raw CSV rows, which would lack the analytical context needed for meaningful retrieval.

**FAISS over hosted databases.** FAISS provides a lightweight, file-based vector store that requires no external services. This simplifies deployment and keeps the project self-contained.

**ConversationalRetrievalChain for memory.** LangChain's `ConversationalRetrievalChain` automatically condenses follow-up questions into standalone queries using chat history, then retrieves fresh context for each turn. This avoids the context window bloat that would occur from appending all prior retrievals to every query.

**GPT-3.5-turbo default with GPT-4 option.** The sidebar model selector allows users to choose between cost efficiency (GPT-3.5-turbo) and higher quality (GPT-4) depending on the complexity of their analysis needs.

### 3.3 Module Structure

| Module | Purpose |
|--------|---------|
| `src/data_loader.py` | Load CSV, parse dates, extract features, compute summaries |
| `src/knowledge_base.py` | Convert summaries to LangChain Documents, embed with OpenAI, store in FAISS |
| `src/retriever.py` | Hybrid retriever: FAISS vector search + live pandas queries |
| `src/chains.py` | Prompt templates and LLM chains (Q&A, Summary, Analysis) |
| `src/memory.py` | Conversation memory and `ChatSession` class |
| `src/evaluation.py` | Ground-truth Q&A pairs and QAEvalChain evaluation |
| `src/visualizations.py` | Plotly chart functions (8 visualizations) |
| `app.py` | Streamlit UI with 4 tabs |

---

## 4. Implementation Details

### 4.1 Data Preparation (Step 1)

`data_loader.py` loads `sales_data.csv` and performs the following transformations:

- Parses the `Date` column to datetime format
- Extracts `Year`, `Month`, `Day`, `Weekday`, and `Quarter` features
- Coerces numeric columns to proper types, handling any nulls by filling with median values
- Computes 7 pre-aggregated summary DataFrames:
  - Monthly and quarterly sales (sum, mean, count)
  - Sales by product and region (sum, mean, count, avg satisfaction)
  - Customer demographics by age group (5 bins) and gender
  - Statistical measures (mean, median, std, min, max) overall and per product/region

### 4.2 Knowledge Base Creation (Step 2)

`knowledge_base.py` converts the summaries into 10 LangChain `Document` objects, each representing one analytical dimension:

1. Dataset overview
2. Monthly sales trends
3. Quarterly sales trends
4. Product performance
5. Regional performance
6. Age group demographics
7. Gender split demographics
8. Overall statistics
9. Statistics by product
10. Statistics by region

Documents are embedded using OpenAI's `text-embedding-3-small` model and stored in a FAISS vector store persisted to the `vector_store/` directory.

### 4.3 Custom Retriever (Step 3)

`retriever.py` implements a `HybridRetriever` that combines two retrieval strategies:

1. **Live pandas query**: Parses the user's question with regex to extract filters (year, quarter, month, product, region, gender) and runs an aggregation on the raw DataFrame. Returns precise numbers like "Total Sales in Q1 2023: 46,086."
2. **FAISS similarity search**: Retrieves the top-k most relevant pre-computed summary documents.

The pandas result (if any) is prepended to the vector results, ensuring the LLM always has access to exact figures alongside broader context.

### 4.4 Prompt Engineering & Chains (Step 4)

`chains.py` defines three prompt templates:

- **Summary prompt**: Instructs the LLM to produce an executive summary covering overall performance, top products, regional highlights, demographics, and trends.
- **Q&A prompt**: Directs the LLM to answer questions using only the provided data context, citing specific numbers.
- **Analysis prompt**: Asks the LLM to produce trend identification, segment comparisons, and actionable recommendations with data-backed claims.

Each prompt feeds into a `RetrievalQA` chain that links the hybrid retriever to the LLM.

### 4.5 Memory Integration (Step 5)

`memory.py` adds `ConversationBufferMemory` to enable multi-turn conversations. The `ConversationalRetrievalChain`:

1. Takes the user's follow-up question and chat history
2. Uses a condense prompt to rephrase it as a standalone question
3. Retrieves fresh context for the rephrased question
4. Generates a response incorporating conversation context

The `ChatSession` class wraps this into a simple `ask()` / `clear()` interface for the Streamlit chat tab.

### 4.6 Model Evaluation (Step 6)

`evaluation.py` defines 15 ground-truth Q&A pairs with exact answers derived from the dataset. The evaluation pipeline:

1. Runs each question through the full RAG QA chain
2. Uses LangChain's `QAEvalChain` to grade each prediction against the ground truth
3. Reports accuracy and per-question pass/fail status

### 4.7 Data Visualizations (Step 7)

`visualizations.py` provides 8 interactive Plotly charts:

1. Monthly sales trends (line chart with markers)
2. Quarterly sales (color-scaled bar chart)
3. Total sales by product (bar chart with labels)
4. Total sales by region (bar chart with labels)
5. Customer age distribution (histogram)
6. Customer gender split (pie chart)
7. Customer satisfaction by product (box plot)
8. Customer satisfaction by region (box plot)

### 4.8 Streamlit Application (Step 8)

`app.py` provides the user interface with:

- **Sidebar**: Date range filter, product/region multiselects, AI model selector (GPT-3.5-turbo / GPT-4), and dataset overview metrics
- **Tab 1 — Dashboard**: 4 KPI cards and all 8 interactive visualizations, all responsive to sidebar filters
- **Tab 2 — Chat with Data**: Conversational interface with message history and memory-backed follow-up support
- **Tab 3 — AI Insights**: One-click executive summary and custom deep analysis
- **Tab 4 — Evaluation**: Run the 15-question QAEvalChain evaluation and view results in a detailed table

---

## 5. Results

### 5.1 Sample Insights

**Executive Summary excerpt (generated by the system):**
> Total sales amount to $1,383,220 across 2,500 transactions. Widget A is the top-selling product at $375,235, followed by Widget B ($346,062). The West region leads with $361,383 in total sales. Female customers show slightly higher average sales ($558.96) compared to male customers ($547.56). The 18-25 age group has the highest average transaction value at $572.29.

**Sample Q&A interactions:**
- Q: "What product has the highest total sales?" → A: "Widget A has the highest total sales at 375,235."
- Q: "How does it compare to the lowest-selling product?" → A: "Widget A ($375,235) outperforms Widget D ($326,854) by $48,381."
- Q: "What are total sales in Q1 2023?" → A: "Total sales in Q1 2023 are 46,086 from 90 transactions."

### 5.2 Evaluation Metrics

The QAEvalChain evaluation across 15 ground-truth questions produced the following results:

| Metric | Value |
|--------|-------|
| Total Questions | 15 |
| Correct Answers | 15 |
| Accuracy | **100%** |

All 15 questions covering products, regions, time periods, demographics, and statistical measures were answered correctly. The hybrid retriever design — combining live pandas queries with vector search — was a key factor in achieving perfect accuracy, as it ensures precise numerical data is always available to the LLM.

### 5.3 Visualizations

The dashboard provides 8 interactive charts covering:
- Temporal trends (monthly line chart, quarterly bar chart)
- Product and regional comparisons (bar charts)
- Customer demographics (age histogram, gender pie chart)
- Satisfaction distributions (box plots by product and region)

All charts respond dynamically to sidebar filters, allowing users to explore subsets of the data interactively.

---

## 6. Conclusion & Future Work

### Conclusion

InsightForge demonstrates that combining RAG with pre-computed analytics and live data queries creates a robust business intelligence assistant. The system achieves 100% accuracy on ground-truth evaluations while providing an intuitive conversational interface that requires no technical expertise to operate. Key architectural decisions — the hybrid retriever, dimensional document design, and conversational memory — work together to deliver grounded, contextual, and accurate insights.

### Future Work

1. **Predictive analytics**: Integrate time-series forecasting models (e.g., Prophet, ARIMA) to answer forward-looking questions like "What will Q1 2029 sales look like?"
2. **Dynamic data upload**: Allow users to upload their own CSV files through the Streamlit interface rather than relying on a fixed dataset
3. **Enhanced visualization generation**: Have the LLM automatically generate relevant charts alongside text answers in the chat interface
4. **Expanded evaluation**: Increase the ground-truth set to 50+ questions covering edge cases, ambiguous queries, and multi-step reasoning
5. **Authentication and multi-tenancy**: Add user authentication so multiple teams can maintain separate conversation histories and datasets
6. **Cost optimization**: Implement response caching for frequently asked questions to reduce API calls

---

## References

1. "Review Study: Business Intelligence Concepts and Approaches." (See `docs/BI_approaches.md`)
2. "Walmart's Sales Data Analysis — A Big Data Analytics Perspective." (See `docs/Walmarts_sales_data_analysis.md`)
3. "Time Series Data Prediction using IoT and Machine Learning Technique." (See `docs/Time_Series_IoT_ML.md`)
4. "AI-Driven Business Model Innovation: A Systematic Review and Research Agenda." (See `docs/AI_business_model_innovation.md`)
