import streamlit as st
import pandas as pd
import tempfile
import duckdb
from graph_agent import build_graph

DB_PATH = "data.duckdb"

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="LangGraph Data Analyst Agent",
    layout="wide"
)

st.title("📊 Data Analyst Agent (LangGraph)")

# -----------------------------
# Sidebar – API Key
# -----------------------------
with st.sidebar:
    st.header("🔐 API Configuration")

    GROQ_API_KEY = st.text_input(
        "Enter GROQ API Key",
        type="password"
    )

    if GROQ_API_KEY:
        st.session_state["groq_key"] = GROQ_API_KEY
        st.success("API key saved")
    else:
        st.warning("Please enter your OpenAI API key")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file and "groq_key" in st.session_state:

    # Load file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # ✅ NORMALIZE COLUMN NAMES (PUT IT HERE)
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    st.subheader("📄 Uploaded Data")
    st.dataframe(df, use_container_width=True)

    # Save temp CSV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name

    # Load into DuckDB
    if uploaded_file and "table_loaded" not in st.session_state:
        conn = duckdb.connect(DB_PATH)
        conn.execute("DROP TABLE IF EXISTS uploaded_data")
        conn.execute("""
            CREATE TABLE uploaded_data AS
            SELECT * FROM read_csv_auto(?)
        """, [temp_path])
        conn.close()

        st.session_state["table_loaded"] = True

    # Build LangGraph
    graph = build_graph(st.session_state["groq_key"])

    # -----------------------------
    # Query Section
    # -----------------------------
    st.subheader("💬 Ask a Question")
    user_query = st.text_area("Ask about your data")

    if st.button("Submit Query"):
        if user_query.strip() == "":
            st.warning("Please enter a question")
        else:
            with st.spinner("Analyzing..."):
                result = graph.invoke({
                    "question": user_query
                })

            st.subheader("🧠 Answer")
            st.markdown(result["answer"])

            st.subheader("📄 Generated SQL")
            st.code(result["sql_query"], language="sql")

            st.subheader("📊 Query Result")
            st.dataframe(result["result"], use_container_width=True)

else:
    st.info("Upload a file and provide an API key to start.")
