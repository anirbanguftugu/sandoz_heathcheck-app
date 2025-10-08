# app.py
import os
import re
import html
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import Tool
from io import BytesIO
# dotenv not used for Streamlit Cloud; reading env directly

# -----------------------------
# Streamlit Page Config and Styles
# -----------------------------
st.set_page_config(page_title="SANDOZ RE HEALTH CHECK CHAT BOT", page_icon="💬", layout="wide")

WHITE_BG = "#FFFFFF"
DEEP_BLUE = "#0D47A1"
BLACK = "#000000"

st.markdown(
    f"""
    <style>
        .main {{
            background-color: {WHITE_BG};
        }}
        .chat-wrapper {{
            max-height: 60vh;
            overflow-y: auto;
            padding: 12px 16px;
            border: 1px solid #E0E0E0;
            border-radius: 8px;
            background: transparent;
        }}
        .msg-user {{
            color: {DEEP_BLUE};
            background: #E8F0FE;
            border-left: 4px solid {DEEP_BLUE};
            padding: 10px 12px;
            border-radius: 6px;
            margin: 8px 0;
            font-weight: 500;
        }}
        .msg-bot {{
            color: {BLACK};
            background: #FFFFFF;
            border-left: 4px solid #9E9E9E;
            padding: 10px 12px;
            border-radius: 6px;
            margin: 8px 0;
        }}
        .msg-meta {{
            font-size: 12px;
            color: #757575;
            margin-bottom: 4px;
        }}
        .small-note {{
            color: #666;
            font-size: 12px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("SANDOZ RE HEALTH CHECK CHAT BOT ✨")
st.caption("Ask questions about your data. 🧠")

# -----------------------------
# NOTE: Removed top-right Download trigger per request. (Download moved to left sidebar)
# -----------------------------

# Sidebar controls (left) — download button moved here (with emojis)
with st.sidebar:
    try:
        _here = os.path.dirname(__file__)
        _logo_path = os.path.join(_here, "SANDOZ_LOGO.png")
        if not os.path.exists(_logo_path):
            _logo_path = "SANDOZ_LOGO.png"
        st.image(_logo_path, width=250)
    except Exception:
        st.write("")

    st.header("⚙️ Controls")
    theme_choice = st.radio("Theme", options=["Light", "Dark"], horizontal=True)
    reset = st.button("🔄 Reset chat (clear memory)")
    st.markdown("---")

    # Download button moved to left sidebar with emojis
    if st.button("💾 ⤢ Download ⬇️", help="Open full-screen download panel"):
        st.session_state["show_download_overlay"] = True

    show_refined = st.checkbox("🔎 Show refined query", value=False, key="show_refined_query")
    st.markdown("""---\n**Tips**\n- Keep queries related to the dataset columns/values.\n- Use specific filters like country, role_name, etc.\n- You can scroll to see past conversation.""")

if st.session_state.get("show_refined_query") and st.session_state.get("last_refined_query"):
    with st.expander("Refined query", expanded=False):
        st.code(st.session_state["last_refined_query"], language="markdown")

if theme_choice == "Dark":
    st.markdown(
        """
        <style>
            .main { background-color: #121212; }
            .chat-wrapper { background: transparent; border-color: #333; }
            .msg-user { background: #0B1E3A; color: #BBDEFB; border-left-color: #2196F3; }
            .msg-bot { background: #2A2A2A; color: #ECECEC; border-left-color: #616161; }
            .msg-meta { color: #BDBDBD; }
            .small-note { color: #B0B0B0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------
# Session state initialization
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "welcomed" not in st.session_state:
    st.session_state.welcomed = False
if "show_download_overlay" not in st.session_state:
    st.session_state.show_download_overlay = False
if reset:
    st.session_state.messages = []
    st.session_state.initialized = False
    st.session_state.welcomed = False
    st.success("Chat and memory have been reset. ✅")

# -----------------------------
# Build core objects (use base code logic)
# -----------------------------
DEFAULT_CSV_PATH = r"Updated_data.csv"
CSV_PATH = os.environ.get("CSV_PATH", DEFAULT_CSV_PATH)

df = None
csv_load_error = None
try:
    df = pd.read_csv(CSV_PATH, low_memory=False)
except Exception as e:
    csv_load_error = str(e)

if df is None:
    st.error(
        "Could not load the dataset. Please ensure the CSV exists at the expected path or set 'CSV_PATH' in environment.\n\n"
        f"Tried: {CSV_PATH}\n\nError: {csv_load_error}"
    )
    st.stop()

# Normalize columns (preserve base normalization semantics but robust)
try:
    if "Team/Business Unit" in df.columns:
        df = df.rename(columns={"Team/Business Unit": "team_business_unit"})
    import re as _re
    def _norm(c: str) -> str:
        c = c.strip().lower()
        c = _re.sub(r"[^0-9a-z]+", "_", c)
        c = _re.sub(r"_+", "_", c).strip("_")
        return c
    df.columns = [_norm(c) for c in df.columns]
except Exception:
    pass

# SQL Engine (in-memory) and save
engine = create_engine("sqlite:///:memory:")
df.to_sql("my_table", engine, index=False, if_exists="replace")

# Windowed memory (k=5) - same as base
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True,
)

# -----------------------------
# Utility functions (from base)
# -----------------------------
def clean_query(query: str) -> str:
    # Remove markdown formatting
    query = query.strip().replace("```", "")
    if query.lower().startswith("sql"):
        query = query[3:].strip()
    query = query.replace("`", "").strip()

    # Remove ONLY trailing semicolons (not quotes!)
    while query.endswith(";"):
        query = query[:-1].strip()

    return query

def run_sql_query(query: str):
    query = query.replace("your_table_name", "my_table")
    query = clean_query(query)
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
            columns = result.keys()
            if not rows:
                return "⚠️ No results found for your query."
            output = ["📑 Columns: " + ", ".join(columns)]
            for row in rows[:20]:
                output.append("🔹 " + ", ".join(str(r) for r in row))
            if len(rows) > 20:
                output.append(f"... ⚠️ Showing only first 20 of {len(rows)} rows.")
            return "\n".join(output)
    except Exception as e:
        return f"❌ SQL execution error: {str(e)}. Check column names and table schema."

# Candidate categorical columns (align with base)
_candidate_cat_cols = [
    "country",
    "role_name",
    "cycle",
    "year",
    "currency_type",
    "team_business_unit",
    "product_name",
    "metric/sales-non_sales",
]
categorical_columns = [c for c in _candidate_cat_cols if c in df.columns]
categorical_values = {col: df[col].astype(str).unique().tolist() for col in categorical_columns}

# Define the SQL tool (same description as base)
sql_tool = Tool(
    name="SQL Database",
    func=run_sql_query,
    description=(
        "Use this tool to answer questions about the CSV data stored in SQL. "
        "If user ask about budget utilization, explain the calculation in your answer first and then provide the output table. "
        "The table name is 'my_table'. "
        "Columns available: " + ", ".join(df.columns) + ". "
        "Categorical values are: " + ", ".join([f"{k}: {v}" for k, v in categorical_values.items()]) + ". "
        "Additional Instructions: "
        "- Remember 'Sales' and 'Targets' are absolute numbers even if they are in decimals. "
        "- 'Attainment' and 'Achievement' are decimals (0–1). Convert to % when showing. "
        "- Aggregations: SUM() for absolute metrics, AVG() for ratios unless SUM() explicitly requested. "
        "- Budget Utilization = SUM(Product_level_Payout) / SUM(Product_Target_Pay). "
        "- Always use exact column names and SQLite syntax."
    ),
)

# OPENAI key (use env)
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.warning("OPENAI_API_KEY is not set in environment. Set it to enable the assistant. 🔑")
else:
    os.environ["OPENAI_API_KEY"] = api_key

# Initialize LLM and agent (matching base)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
sql_agent = initialize_agent(
    tools=[sql_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
)

# -----------------------------
# Validation & refine (base behavior)
# -----------------------------
def is_query_relevant(query: str) -> bool:
    col_list = ", ".join(df.columns)
    val_list = ", ".join([f"{k}: {len(v)} values" for k, v in categorical_values.items()])

    validation_prompt = f"""
    You are a smart data assistant.
    The dataset has these columns: {col_list}.
    Example categorical values: {val_list}.
    The user asked: "{query}"

    Decide if this query is meaningful and can be answered using the dataset.
    If not, respond only: "This query is not relevant. Please rephrase your question."
    Otherwise, respond YES.
    """

    response = llm.invoke(validation_prompt)
    if hasattr(response, "content"):
        response_text = response.content.strip().lower()
    else:
        response_text = str(response).strip().lower()
    return response_text.startswith("yes")

def refine_query_with_context(query: str, history: list) -> str:
    history_text = "\n".join([f"User: {h}" for h in history[-5:]])
    prompt = f"""
    You are a query rewriting assistant.
    Conversation history:
    {history_text}

    Current user question: "{query}"

    Task:
    1. Rewrite the user question into a complete, contextually clear query
       that can be directly answered from the dataset. Make it elaborate and detailed as much as possible so that the agent will understand easily. FOCUS ON USER R.
    2. MUST normalize any mention of time cycles (quarters or trimesters) into the exact tokens used in the dataset:
   - Quarters → format **Qn-YYYY** (e.g., Q1-2024, Q2-2025). Accept input variants such as "Q1", "quarter 1", "qtr 1", "first quarter of 2024", "quarter 2 2025", etc., and convert them to **Qn-YYYY**.
   - Trimesters (1st/2nd/3rd trimester / first trimester / second trimester, etc.) → format **Cn-YYYY** where C1 = 1st trimester, C2 = 2nd trimester, C3 = 3rd trimester (e.g., C1-2025, C2-2024). Accept input variants such as "1st trimester", "second trimester", "third trim", "trimester 2 2025", etc., and convert them to **Cn-YYYY**.

    3. If the user question is completely irrelevant or nonsensical or user is talking about anything which is not the part of data column or values. Try to follow the below few shot prompt example, if not there, try to rephrase query on your own with details., 
       reply exactly: "This query is not relevant. Please rephrase your question."
    4. If the user query is regarding any currency for any particular country mentioned by user, ask agent clearly to mention the relevant currency to be used. You can guide the agent by mentioning the currency in the rewritten query.
    5. In the rewritten query always mention the agent to present the user requested in relevant table, which will be useful to draw insights. For example: If user ask for budget utilization, ask agent to create outoput tabel where payout, taget pay and %budget utilization is mentioned. And as payout and target pay is a currency value, ask to add relevant currency.
    6. If user query refer 'geography' word, then in refine query mention this as terr_id.
    7. If user mention 'field force', field force size', 'headcount','rep' word without mentioning role, then refer this as emp_id.
    8. When user want any output in employee level, in refine query ask agent specificaly for emp id and emp_name.
    9. If user mention 'component', 'molecule', 'promo grid', then user is actually referring 'Product Name', so in refine query mention product name.
    10. If user is using 'quota', 'goal','goals' in his query, user is actually referring 'Targets', so modify the refine query accordingly.
    11. If user is using 'wt', 'weight', 'product weight', 'weightage', then user is actually referring product wt column, so modify the refine query accordingly.
    12. If user is using 'payout factor' or 'payout percentage', then user is referring 'achievement' column.
    16. If user is asking for 'Earnings', then use the formula:
        Earnings = Achievement * Product Target_pay
    17. If user is asking about focus product, then user is actually want to know product with highest 'Product wt'
    18. If user is asking about weight of 'non sales' or 'sales' component, 'Metric/sales-non sales' column has 'Sales' and 'Non-Sales' component categorization, so use this column in refined query.
    ### 🔹 FEW-SHOT EXAMPLES

    Example A
    User: "how many countries do we have"
    Refined: "Provide the number of distinct countries in the dataset (count distinct country). Present the result in a table with columns: metric, value."

    **Example 1 — Sales (Quarter + Trimester reference)**
    User: "give me total sales role wise for india in 1st qtr 2025 and 2nd trimister 2024"  
    Refined: "Provide total sales grouped by role_name for India for Q1-2025 and C2-2024.  
    Present the results in a table with columns: role_name, cycle, sales."

    ---

    **Example 2 — Sales by Geography**
    User: "show total sales geography wise in Q3 2024"  
    Refined: "Provide total sales grouped by terr_id for Q3-2024.  
    Present the results in a table with columns: terr_id, cycle, sales."

    ---

    **Example 3 — Sales by Product**
    User: "compare sales by component in 2nd trimester 2025"  
    Refined: "Provide total sales grouped by product_name for C2-2025.  
    Present the results in a table with columns: product_name, cycle, sales."

    ---

    **Example 4 — Payout with Currency**
    User: "show payout by role for UK in Q4 2024"  
    Refined: "Provide total payout grouped by role_name for the UK for Q4-2024.  
    Present the results in a table with columns: role_name, cycle, payout (in GBP)."

    ---

    **Example 5 — Payout vs Target**
    User: "compare payout and target pay for india in 3rd quarter 2025"  
    Refined: "Provide payout and target_pay grouped by country for Q3-2025 for India.  
    Present the results in a table with columns: country, cycle, payout (INR), target_pay (INR)."

    ---

    **Example 6 — Budget Utilization**
    User: "show budget utilization by country for Q2 2025"  
    Refined: "Provide payout, target_pay, and %budget_utilization grouped by country for Q2-2025.  
    Present the results in a table with columns: country, cycle, payout (currency), target_pay (currency), budget_utilization(%)."

    ---

    **Example 7 — Field Force Size**
    User: "show field force size by geography for 1st trimester 2025"  
    Refined: "Provide count of emp_id grouped by terr_id for C1-2025.  
    Present the results in a table with columns: terr_id, cycle, headcount."

    ---

    **Example 8 — Employee-Level View**
    User: "show payout for each employee in Q2 2025"  
    Refined: "Provide payout at employee level grouped by emp_id and emp_name for Q2-2025.  
    Present the results in a table with columns: emp_id, emp_name, cycle, payout (currency)."

    ---

    **Example 9 — Product Targets**
    User: "give me target for all molecules in Q1 2024"  
    Refined: "Provide targets grouped by product_name for Q1-2024.  
    Present the results in a table with columns: product_name, cycle, targets."

    ---

    **Example 10 — Achievement and Payout Factor**
    User: "show payout percentage by role for 2nd trimester 2025"  
    Refined: "Provide achievement (payout percentage) grouped by role_name for C2-2025.  
    Present the results in a table with columns: role_name, cycle, achievement(%)."

    ---

    **Example 11 — Product Weightage**
    User: "show product weightage for each component in Q4 2025"  
    Refined: "Provide product_wt grouped by product_name for Q4-2025.  
    Present the results in a table with columns: product_name, cycle, product_wt."

    ---

    **Example 12 — Focus Product**
    User: "show focus product for Q3 2024"  
    Refined: "Identify the product with the highest product_wt for Q3-2024.  
    Present the results in a table with columns: product_name, cycle, product_wt."

    ---

    **Example 13 — Sales vs Non-Sales Weight**
    User: "show weight of non sales component for 1st trimester 2025"  
    Refined: "Provide product_wt for components categorized as 'Non-Sales' for C1-2025.  
    Present the results in a table with columns: component_type, cycle, product_wt."

    ---

    **Example 14 — Quota / Goal Queries**
    User: "compare goals and payout for europe in Q2 2024"  
    Refined: "Provide targets and payout grouped by country for Europe for Q2-2024.  
    Present the results in a table with columns: country, cycle, targets, payout (currency)."

    ---

    **Example 15 — Earnings Calculation**
    User: "show earnings by role for Q1 2025"  
    Refined: "Provide earnings (calculated as achievement * product_target_pay) grouped by role_name for Q1-2025.  
    Present the results in a table with columns: role_name, cycle, earnings (currency)."

    ---

    **Example 16 — Headcount Trend**
    User: "compare field force headcount between Q2 and Q3 2025"  
    Refined: "Provide count of emp_id (headcount) for Q2-2025 and Q3-2025.  
    Present the results in a table with columns: cycle, headcount."

    ---

    **Example 17 — Territory Comparison**
    User: "compare sales between north and south india in Q4 2024"  
    Refined: "Provide total sales grouped by terr_id for North India and South India for Q4-2024.  
    Present the results in a table with columns: terr_id, region, cycle, sales."

    ---

    **Example 18 — Invalid Query Handling**
    User: "how many chocolates were sold in mars"  
    Refined: "This query is not relevant. Please rephrase your question."

    ---

    **Example 19 — Component-Level Earnings**
    User: "show earnings by component in Q3 2024"  
    Refined: "Provide earnings (achievement * product_target_pay) grouped by product_name for Q3-2024.  
    Present the results in a table with columns: product_name, cycle, earnings (currency)."

    ---

    **Example 20 — Role-wise Achievement Comparison**
    User: "compare achievement across roles for 2nd trimester 2025"  
    Refined: "Provide achievement grouped by role_name for C2-2025.  
    Present the results in a table with columns: role_name, cycle, achievement(%)."

    ---

    **Example 21 — Non-Sales Metric**
    User: "show payout for non sales metric in Q2 2025"  
    Refined: "Provide payout for components categorized as 'Non-Sales' in Q2-2025.  
    Present the results in a table with columns: component_type, cycle, payout (currency)."

    ---

    **Example 22 — Role-wise Targets**
    User: "give me role wise quota for Q3 2025"  
    Refined: "Provide targets grouped by role_name for Q3-2025.  
    Present the results in a table with columns: role_name, cycle, targets."

    ---

    **Example 23 — Role-wise Payout & Achievement**
    User: "show payout and achievement by role for C2 2025"  
    Refined: "Provide payout and achievement grouped by role_name for C2-2025.  
    Present the results in a table with columns: role_name, cycle, payout (currency), achievement(%)."

    ---

    **Example 24 — Territory + Product Analysis**
    User: "show sales by geography and product for Q1 2024"  
    Refined: "Provide total sales grouped by terr_id and product_name for Q1-2024.  
    Present the results in a table with columns: terr_id, product_name, cycle, sales."

    ---

    **Example 25 — Missing Year (auto-assume latest year)**
    User: "give me sales by role for Q2"  
    Refined: "Provide total sales grouped by role_name for Q2-2025 (assumed latest dataset year).  
    Present the results in a table with columns: role_name, cycle, sales."

    ---

    **Example 26 — Achievement by Product Category**
    User: "show payout factor by molecule for 3rd trimester 2024"  
    Refined: "Provide achievement grouped by product_name for C3-2024.  
    Present the results in a table with columns: product_name, cycle, achievement(%)."

    ---

    **Example 27 — Goal vs Achievement by Role**
    User: "compare goal and achievement role wise for Q4 2024"  
    Refined: "Provide targets and achievement grouped by role_name for Q4-2024.  
    Present the results in a table with columns: role_name, cycle, targets, achievement(%)."

    ---

    **Example 28 — Currency Context**
    User: "show payout by product for Japan in Q1 2025"  
    Refined: "Provide payout grouped by product_name for Japan for Q1-2025, with payout values in JPY.  
    Present the results in a table with columns: product_name, cycle, payout (JPY)."

    ---

    **Example 29 — Product Weight by Sales Type**
    User: "show product weight for sales components in C3 2025"  
    Refined: "Provide product_wt for components categorized as 'Sales' for C3-2025.  
    Present the results in a table with columns: component_type, cycle, product_wt."

    ---

    **Example 30 — Cross-Year Comparison**
    User: "compare total sales between Q2 2024 and Q2 2025"  
    Refined: "Provide total sales for Q2-2024 and Q2-2025.  
    Present the results in a table with columns: cycle, sales."

    Return only the rewritten question or the 'not relevant' message.
    """
    response = llm.invoke(prompt)
    refined_response = response.content.strip() if hasattr(response, "content") else str(response).strip()
    # keep compatibility with Streamlit UI
    st.session_state["last_refined_query"] = refined_response
    # base logic printed too
    print(refined_response)
    return refined_response

def ask_csv_agent(user_question: str) -> str:
    user_history = [m.get("content", "") for m in st.session_state.messages if m.get("role") == "user"]
    refined_question = refine_query_with_context(user_question, user_history)
    if "not relevant" in refined_question.lower():
        return "⚠️ This query is not relevant. Please rephrase your question."
    try:
        resp = sql_agent.run(f"{refined_question} [Original user input: {user_question}]")
        return resp
    except Exception as e:
        return f"❌ Agent execution error: {str(e)}"

# -----------------------------
# Download helpers (copied from provided streamlit code)
# -----------------------------
def to_csv_bytes(df_export: pd.DataFrame) -> bytes:
    return df_export.to_csv(index=False).encode("utf-8")

def to_excel_bytes(df_export: pd.DataFrame) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        df_export.to_excel(writer, index=False, sheet_name="filtered_data")
    bio.seek(0)
    return bio.read()

def _existing_col(df_in: pd.DataFrame, candidates):
    for c in candidates:
        if c in df_in.columns:
            return c
    return None

# -------------------------------------------------
# Download Overlay (full-screen) with filters & download
# -------------------------------------------------
if st.session_state.get("show_download_overlay"):
    def render_filtered_download_panel():
        st.markdown("---")
        st.subheader("📥 Download filtered data")
        st.caption("Select at least one filter. Options cascade based on earlier selections.")
        if st.button("✖ Close", help="Close download panel"):
            st.session_state.show_download_overlay = False
            st.rerun()

        ordered_filters = [
            ("🌎 Country", ["country"]),
            ("📆 Cycle", ["cycle"]),
            ("🏢 Business line", [
                "team_business_unit",
                "business_line",
                "business_unit",
                "business_team_bus",
                "team_business",
            ]),
            ("🧪 Product", ["product_name", "product", "product_r", "product_y"]),
            ("👤 Role", ["role_name", "role"]),
        ]

        mapped = []
        for label, cands in ordered_filters:
            col = _existing_col(df, cands)
            if col:
                mapped.append((label, col))

        if not mapped:
            st.info("No relevant categorical columns found in this dataset.")
            return

        if "download_filters" not in st.session_state:
            st.session_state.download_filters = {}

        with st.expander("Detected filter-to-column mapping", expanded=False):
            for label, col in mapped:
                st.write(f"{label} → `{col}`")

        working_df = df.copy()
        cols = st.columns(2)
        i = 0
        for label, col in mapped:
            options = sorted(working_df[col].astype(str).dropna().unique().tolist())
            prev_sel = st.session_state.download_filters.get(col, [])
            prev_sel = [v for v in prev_sel if v in options]
            st.session_state.download_filters[col] = prev_sel
            with cols[i % 2]:
                sel = st.multiselect(
                    label=f"Filter by {label}",
                    options=options,
                    default=prev_sel,
                    key=f"dl_{col}",
                    help="Options update based on earlier selections.",
                )
                st.session_state.download_filters[col] = sel
            if sel:
                working_df = working_df[working_df[col].astype(str).isin(sel)]
            i += 1

        at_least_one = any(len(v) > 0 for v in st.session_state.download_filters.values())

        filtered_df = df
        for _, col in mapped:
            vals = st.session_state.download_filters.get(col, [])
            if vals:
                filtered_df = filtered_df[filtered_df[col].astype(str).isin(vals)]

        with st.container():
            if not at_least_one:
                st.warning("Select at least one filter to preview and download filtered data.")
            else:
                if filtered_df.empty:
                    st.error("No data matches the selected filters. Try adjusting your selections.")
                else:
                    st.write(f"Filtered rows: {len(filtered_df):,}")
                    st.dataframe(filtered_df.head(1000), use_container_width=True)
                    if len(filtered_df) > 1000:
                        st.caption(
                            f"Showing first 1,000 rows out of {len(filtered_df):,}. Download will include all filtered rows."
                        )

        can_download = at_least_one and not filtered_df.empty
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                label="⬇️ CSV",
                data=to_csv_bytes(filtered_df) if can_download else b"",
                file_name="filtered_data.csv",
                mime="text/csv",
                disabled=not can_download,
                use_container_width=True,
            )
        with c2:
            st.download_button(
                label="⬇️ Excel",
                data=to_excel_bytes(filtered_df) if can_download else b"",
                file_name="filtered_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                disabled=not can_download,
                use_container_width=True,
            )

    render_filtered_download_panel()
    st.stop()

# -------------------------------------------------
# Chat UI rendering (copied and adapted)
# -------------------------------------------------
chat_ph = st.empty()

# -----------------------------
# Markdown table parser and improved render_chat()
# -----------------------------
def _parse_markdown_table(md: str):
    """
    Parse a GitHub-style markdown table into a pandas.DataFrame.
    Returns DataFrame or None if parsing fails.
    """
    m = re.search(r"(\|.*\|\s*\n\|[ \-:\|]+?\|\s*\n(?:\|.*\|\s*\n?)*)", md, flags=re.M)
    if not m:
        return None
    table_block = m.group(1).strip()

    lines = [ln.strip() for ln in table_block.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None

    def split_row(row):
        row = row.strip()
        if row.startswith("|"):
            row = row[1:]
        if row.endswith("|"):
            row = row[:-1]
        cells = [c.strip() for c in row.split("|")]
        return cells

    header = split_row(lines[0])
    sep_line = lines[1]
    if not re.match(r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?$", sep_line):
        return None

    data_rows = []
    for row in lines[2:]:
        if "|" not in row:
            continue
        cells = split_row(row)
        if len(cells) != len(header):
            if len(cells) < len(header):
                cells = cells + [""] * (len(header) - len(cells))
            else:
                cells = cells[:len(header)]
        data_rows.append(cells)

    try:
        df_parsed = pd.DataFrame(data_rows, columns=header)
        for c in df_parsed.columns:
            try:
                tmp = df_parsed[c].astype(str).str.replace(",", "").replace("", pd.NA)
                df_parsed[c] = pd.to_numeric(tmp)
            except Exception:
                df_parsed[c] = df_parsed[c].astype(str)
        return df_parsed
    except Exception:
        return None


def render_chat():
    if st.session_state.messages:
        c = chat_ph.container()
        c.markdown("<div class='chat-wrapper'>", unsafe_allow_html=True)

        def md_basic(txt: str) -> str:
            s = html.escape(txt)
            s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
            s = s.replace("\n", "<br/>")
            return s

        for m in st.session_state.messages:
            content = m.get("content", "") or ""
            if m["role"] == "user":
                content_html = md_basic(content)
                block = f"<div class='msg-meta'>🙋 You</div><div class='msg-user'>{content_html}</div>"
                c.markdown(block, unsafe_allow_html=True)
            else:
                # Bot message: try to detect and render markdown table nicely
                df_table = _parse_markdown_table(content)
                if df_table is not None:
                    table_match = re.search(r"(\|.*\|\s*\n\|[ \-:\|]+?\|\s*\n(?:\|.*\|\s*\n?)*)", content, flags=re.M)
                    before = content[:table_match.start()].strip()
                    after = content[table_match.end():].strip()
                    if before:
                        before_html = md_basic(before)
                        block = f"<div class='msg-meta'>🤖 Bot</div><div class='msg-bot'>{before_html}</div>"
                        c.markdown(block, unsafe_allow_html=True)
                    # Render the parsed dataframe in a clean table
                    # Use the container 'c' so the table appears inside the chat area
                    with c:
                        st.dataframe(df_table, use_container_width=True)
                    if after:
                        after_html = md_basic(after)
                        block = f"<div class='msg-meta'>🤖 Bot</div><div class='msg-bot'>{after_html}</div>"
                        c.markdown(block, unsafe_allow_html=True)
                else:
                    content_html = md_basic(content)
                    block = f"<div class='msg-meta'>🤖 Bot</div><div class='msg-bot'>{content_html}</div>"
                    c.markdown(block, unsafe_allow_html=True)

        c.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Welcome message
# -------------------------------------------------
if not st.session_state.welcomed and not st.session_state.messages:
    welcome_text = (
        "👋 Welcome! I’m your Sandoz RE Health Check assistant. Ask me questions about the dataset, "
        "like country/role-wise metrics, attainment, or budget utilization."
    )
    st.session_state.messages.append({"role": "bot", "content": welcome_text})
    st.session_state.welcomed = True

render_chat()

# Input area at the bottom
user_input = st.chat_input("Type your question here… ❓")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    render_chat()

    typing_ph = st.empty()
    typing_ph.markdown("<span class='small-note'>🤖 Bot is typing…</span>", unsafe_allow_html=True)

    answer = ask_csv_agent(user_input)

    full_bot_msg = f"📊 Answer:\n{answer}"
    st.session_state.messages.append({"role": "bot", "content": full_bot_msg})
    typing_ph.empty()
    render_chat()
    render_chat()

st.markdown("<span class='small-note'>📝 Tip: Use the sidebar to reset chat and switch theme.</span>", unsafe_allow_html=True)
