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
from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv  # Disabled for Streamlit Cloud; use st.secrets instead

# -------------------------------------------------
# Streamlit Page Config and Basic Styles
# -------------------------------------------------
st.set_page_config(page_title="SANDOZ RE HEALTH CHECK CHAT BOT", page_icon="💬", layout="wide")

# Custom CSS for colors and scrollable chat area
WHITE_BG = "#FFFFFF"
DEEP_BLUE = "#0D47A1"  # Deep blue for user text
BLACK = "#000000"      # Black for bot text

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

# -------------------------------------------------
# Header and helpful notes
# -------------------------------------------------
st.title("SANDOZ RE HEALTH CHECK CHAT BOT ✨")
st.caption("Ask questions about your data. 🧠")

# Sidebar controls
with st.sidebar:
    # Logo (use absolute path for reliability)
    try:
        _here = os.path.dirname(__file__)
        _logo_path = os.path.join(_here, "SANDOZ_LOGO.png")
        if not os.path.exists(_logo_path):
            _logo_path = "SANDOZ_LOGO.png"
        st.image(_logo_path, width=250)
    except Exception:
        st.write("")

    st.header("⚙️ Controls")
    # Theme toggle
    theme_choice = st.radio("Theme", options=["Light", "Dark"], horizontal=True)
    reset = st.button("🔄 Reset chat (clear memory)")
    st.markdown("""---\n**Tips**\n- Keep queries related to the dataset columns/values.\n- Use specific filters like country, role_name, etc.\n- You can scroll to see past conversation.""")

# Apply theme overrides via CSS after selection
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
# State management
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: {role: "user"|"bot", "content": str}
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "welcomed" not in st.session_state:
    st.session_state.welcomed = False
if reset:
    st.session_state.messages = []
    st.session_state.initialized = False
    st.session_state.welcomed = False
    st.success("Chat and memory have been reset. ✅")

# -------------------------------------------------
# Build core objects (same logic/flow as original)
# -------------------------------------------------
# 1) Load CSV — follow original absolute path; allow fallback if not found.
#    Prefer Streamlit Cloud secrets; .env loading is disabled for cloud deployment.
# load_dotenv()  # intentionally disabled
DEFAULT_CSV_PATH = r"new_demo_data_updated - Copy.csv"
CSV_PATH = st.secrets.get("CSV_PATH", os.environ.get("CSV_PATH", DEFAULT_CSV_PATH))

df = None
csv_load_error = None
try:
    df = pd.read_csv(CSV_PATH, low_memory=False)
except Exception as e:
    csv_load_error = str(e)

if df is None:
    st.error(
        "Could not load the dataset. Please ensure the CSV exists at the expected path or set 'CSV_PATH' in environment or Streamlit secrets.\n\n"
        f"Tried: {CSV_PATH}\n\nError: {csv_load_error}"
    )
    st.stop()

# 2) Normalize columns
try:
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
except Exception:
    pass

# 3) Create in-memory SQLite engine and store table
engine = create_engine("sqlite:///:memory:")
df.to_sql("my_table", engine, index=False, if_exists="replace")

# 4) Windowed memory (last 5 turns)
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True,
)

# 5) Utility function to clean queries (same as original)
def clean_query(query: str) -> str:
    query = query.strip().replace("```", "")
    if query.lower().startswith("sql"):
        query = query[3:].strip()
    query = query.replace("`", "").strip()
    return query

# 6) SQL query runner (same as original)
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

# 7) Categorical values (same logic)
categorical_columns = ["country", "role_name"]
categorical_values = {col: df[col].astype(str).unique().tolist() for col in categorical_columns if col in df.columns}

# 8) Define tool (same information)
sql_tool = Tool(
    name="SQL Database",
    func=run_sql_query,
    description=(
        "Use this tool to answer questions about the CSV data stored in SQL. "
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

# 9) API key setup (avoid hardcoding). Prefer Streamlit secrets; fallback to OS env.
#    .env loading is disabled for Streamlit Cloud; set secrets in .streamlit/secrets.toml
api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
if not api_key:
    st.warning("OPENAI_API_KEY is not set. Please add it to Streamlit secrets or environment to enable the assistant. 🔑")
else:
    os.environ["OPENAI_API_KEY"] = api_key

# 10) Initialize Conversational ReAct Agent (same as original)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
sql_agent = initialize_agent(
    tools=[sql_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
)

# 11) LLM-based Validation (same prompt/logic)
def is_query_relevant(query: str) -> bool:
    col_list = ", ".join(df.columns)
    val_list = ", ".join([f"{k}: {len(v)} values" for k, v in categorical_values.items()])

    validation_prompt = f"""
    You are a smart data assistant.
    The dataset has these columns: {col_list}.
    Example categorical values: {val_list}.
    The user asked: "{query}"

    Decide if this query is meaningful and can be answered using the dataset.
    Answer only YES or NO.
    """

    response = llm.predict(validation_prompt).strip().lower()
    return response.startswith("yes")

# 12) Ask agent safely (same logic)
def ask_csv_agent(user_question: str) -> str:
    if not is_query_relevant(user_question):
        # Remove last user message from memory if invalid
        if memory.chat_memory.messages:
            memory.chat_memory.messages.pop()
        return "⚠️ Please ask a valid question related to the dataset (check column names or values)."
    try:
        resp = sql_agent.run(user_question)
        return resp
    except Exception as e:
        return f"❌ Agent execution error: {str(e)}"

# -------------------------------------------------
# Chat UI
# -------------------------------------------------
# Chat area placeholder and renderer
chat_ph = st.empty()

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
            content_html = md_basic(m["content"]) if m.get("content") else ""
            if m["role"] == "user":
                block = f"<div class='msg-meta'>🙋 You</div><div class='msg-user'>{content_html}</div>"
            else:
                block = f"<div class='msg-meta'>🤖 Bot</div><div class='msg-bot'>{content_html}</div>"
            c.markdown(block, unsafe_allow_html=True)
        c.markdown("</div>", unsafe_allow_html=True)

# Add a welcome message on first load so the chat isn't empty
if not st.session_state.welcomed and not st.session_state.messages:
    welcome_text = (
        "👋 Welcome! I’m your Sandoz RE Health Check assistant. Ask me questions about the dataset, "
        "like country/role-wise metrics, attainment, or budget utilization."
    )
    st.session_state.messages.append({"role": "bot", "content": welcome_text})
    st.session_state.welcomed = True

# Initial render
render_chat()

# Input area at the bottom
user_input = st.chat_input("Type your question here… ❓")

if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Re-render chat immediately so the user's message appears instantly
    render_chat()

    # Tiny typing indicator using a status placeholder
    typing_ph = st.empty()
    typing_ph.markdown("<span class='small-note'>🤖 Bot is typing…</span>", unsafe_allow_html=True)

    # Get agent answer
    answer = ask_csv_agent(user_input)

    # Do not generate insights/follow-ups per request
    full_bot_msg = f"📊 Answer:\n{answer}"

    # Append bot message
    st.session_state.messages.append({"role": "bot", "content": full_bot_msg})

    # Remove typing indicator
    typing_ph.empty()

    # Re-render chat with bot response
    render_chat()

    # Final render without forcing a rerun so content remains visible
    render_chat()

# Footer note (removed memory mention)
st.markdown("<span class='small-note'>📝 Tip: Use the sidebar to reset chat and switch theme.</span>", unsafe_allow_html=True)
