import tempfile
import csv
import streamlit as st
import pandas as pd
import duckdb

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage



import re

def clean_sql(sql: str) -> str:
    # Remove ```sql ... ``` blocks
    sql = re.sub(r"```sql", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"```", "", sql)
    return sql.strip()   


# ------------------------------------
# File preprocessing
# ------------------------------------
def preprocess_and_save(file):
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file format")
            return None, None, None

        for col in df.columns:
            if "date" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp:
            df.to_csv(temp.name, index=False, quoting=csv.QUOTE_ALL)
            return temp.name, df.columns.tolist(), df

    except Exception as e:
        st.error(e)
        return None, None, None

# ------------------------------------
# DuckDB Setup
# ------------------------------------
conn = duckdb.connect(database=":memory:")

def run_sql(query: str):
    try:
        return conn.execute(query).df().to_markdown()
    except Exception as e:
        return f"SQL Error: {e}"

@tool
def duckdb_sql(query: str) -> str:
    """Run SQL queries on uploaded_data table"""
    return run_sql(query)

# ------------------------------------
# LangGraph State
# ------------------------------------
class AgentState(TypedDict):
    query: str
    response: Optional[str]

# ------------------------------------
# LLM Node (Groq)
# ------------------------------------
def llm_node(state: AgentState):
    llm = ChatGroq(
        groq_api_key=st.session_state.groq_key,
        model_name="meta-llama/llama-4-maverick-17b-128e-instruct"
    )

    prompt = f"""
    You are a data analyst.

    Rules:
    - Output ONLY raw SQL
    - DO NOT use markdown
    - DO NOT wrap SQL in ``` blocks

    Table: uploaded_data

    User Question:
    {state['query']}
    """

    res = llm.invoke([HumanMessage(content=prompt)])
    return {"response": res.content}

# ------------------------------------
# SQL Execution Node
# ------------------------------------
def sql_node(state: AgentState):
    sql = clean_sql(state["response"])
    result = run_sql(sql)
    return {"response": result}

# ------------------------------------
# Router
# ------------------------------------
def router(state: AgentState):
    if "select" in state["response"].lower():
        return "sql"
    return END

# ------------------------------------
# Build Graph
# ------------------------------------
graph = StateGraph(AgentState)

graph.add_node("llm", llm_node)
graph.add_node("sql", sql_node)

graph.set_entry_point("llm")
graph.add_conditional_edges("llm", router, {"sql": "sql", END: END})
graph.add_edge("sql", END)

app_graph = graph.compile()

# ------------------------------------
# Streamlit UI
# ------------------------------------
st.set_page_config("📊 LangGraph Data Analyst", layout="wide")
st.title("📊 Data Analyst Agent (LangGraph + Groq)")

with st.sidebar:
    st.header("🔑 Groq API Key")
    groq_key = st.text_input("Enter Groq API Key", type="password")
    if groq_key:
        st.session_state.groq_key = groq_key

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file and "groq_key" in st.session_state:
    temp_path, columns, df = preprocess_and_save(uploaded_file)

    if temp_path:
        conn.execute("DROP TABLE IF EXISTS uploaded_data")
        conn.execute(f"""
            CREATE TABLE uploaded_data AS
            SELECT * FROM read_csv_auto('{temp_path}')
        """)

        st.subheader("📄 Uploaded Data")
        st.dataframe(df)

        user_query = st.text_area("Ask a question about the data")

        if st.button("Submit Query"):
            with st.spinner("Analyzing..."):
                result = app_graph.invoke({"query": user_query})
                st.markdown("### ✅ Result")
                st.markdown(result["response"])
