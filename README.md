# SANDOZ RE HEALTH CHECK CHAT BOT

A Streamlit chat application that wraps your existing CSV + SQL + LangChain agent logic into a clean, scrollable UI. It preserves the original code flow while adding a modern chat interface, theme toggle, and a typing indicator.

## Features
- White background (Light) / Dark mode toggle in sidebar
- Deep blue user messages, black bot messages
- Scrollable conversation area (no large empty box)
- Reset chat button (clears conversation display)
- Uses in-memory SQLite to query your CSV
- Subtle "Bot is typing…" indicator for better UX
- Sidebar logo support (`SANDOZ_LOGO.png` in this folder)

## Project Structure
```
sandoz_chat_app/
├─ app.py                # Streamlit UI (uses original logic and functions)
├─ requirements.txt      # Python dependencies
└─ README.md             # This file
```

## How it works
- The app loads your CSV, normalizes column names, stores it as `my_table` in an in-memory SQLite database, and uses a LangChain ReAct agent with a SQL tool to answer questions.
- Irrelevant questions are filtered by an LLM-based validator (same logic).
- Insights and follow-ups have been disabled in the UI by design (only the main answer is shown).

## Configuration
- The app expects the dataset at the original absolute path used in your script:
  `C:\\Users\\Anirban-PC\\Desktop\\Desktop Item\\Gen AI Folder\\Final_CSV_AGENT_project\\new_demo_data_updated - Copy.csv`
- For Streamlit Cloud, configuration is read from `st.secrets` first. Provide values in Secrets as shown below.
- For local development, you may also rely on OS environment variables as a fallback. Loading from `.env` is disabled in code for cloud deployment.

### Streamlit Secrets (primary on Cloud)
Add in Streamlit Cloud settings when deploying:
```
OPENAI_API_KEY = "sk-..."
CSV_PATH = "C:\\path\\to\\your\\dataset.csv"
```

Optionally, for local development you can set OS environment variables:
```
set OPENAI_API_KEY=sk-...
set CSV_PATH=C:\\path\\to\\your\\dataset.csv
```
or in PowerShell:
```
$env:OPENAI_API_KEY = "sk-..."
$env:CSV_PATH = "C:\\path\\to\\your\\dataset.csv"
```

## Run locally
1. Create a virtual environment (recommended) and activate it
2. Install dependencies
3. Run Streamlit

### Windows (PowerShell)
```
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud
1. Push this folder to a GitHub repository
2. In Streamlit Cloud, create a new app pointing to `sandoz_chat_app/app.py`
3. Add `OPENAI_API_KEY` (and optionally `CSV_PATH`) in the app's Secrets
4. Deploy

## Notes
- This app avoids hardcoding the API key. If the key is missing, a warning is shown in the UI.
- The agent uses `gpt-4o-mini` via `langchain_community.chat_models.ChatOpenAI`. Ensure your key has access.

## Suggested Beautifications
- Rounded chat bubbles with subtle elevation (added)
- Emojis for roles and sections (added)
- Compact meta labels for role names (added)
- Sticky input at the bottom using `st.chat_input` (added)
- Optional: Show latest query history in a sidebar pill list

Enjoy! ✨
