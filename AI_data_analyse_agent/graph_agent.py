from typing import TypedDict, Any
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
import duckdb
import os

# -----------------------------
# Graph State
# -----------------------------
class AgentState(TypedDict):
    question: str
    sql_query: str
    result: Any
    answer: str

# -----------------------------
# Graph Builder
# -----------------------------
def build_graph(GROQ_API_KEY: str):

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0.3
    )

    # -----------------------------
    # Node 1: NL → SQL
    # -----------------------------
    def generate_sql(state: AgentState):
        prompt = f"""
        You are an expert data analyst.
        Table name: uploaded_data

        Convert the user's question into a DuckDB SQL query.
        Return ONLY SQL. Do not use markdown.

        Question:
        {state['question']}
        """
        raw_sql = llm.invoke(prompt).content

        #  CRITICAL FIX
        sql = (
            raw_sql
            .replace("```sql", "")
            .replace("```", "")
            .strip()
        )

        return {"sql_query": sql}

    # -----------------------------
    # Node 2: Execute SQL
    # -----------------------------
    def execute_sql(state: AgentState):
        conn = duckdb.connect("data.duckdb")
        df = conn.execute(state["sql_query"]).df()
        conn.close()
        return {"result": df}
    # -----------------------------
    # Node 3: Explain Result
    # -----------------------------
    def explain_result(state: AgentState):
        prompt = f"""
        Explain the following SQL result in clear, simple language.

        Result:
        {state['result'].head().to_markdown()}
        """
        answer = llm.invoke(prompt).content
        return {"answer": answer}

    # -----------------------------
    # Build Graph
    # -----------------------------
    builder = StateGraph(AgentState)

    builder.add_node("sql_generator", generate_sql)
    builder.add_node("sql_executor", execute_sql)
    builder.add_node("explainer", explain_result)

    builder.set_entry_point("sql_generator")
    builder.add_edge("sql_generator", "sql_executor")
    builder.add_edge("sql_executor", "explainer")
    builder.add_edge("explainer", END)

    return builder.compile()
