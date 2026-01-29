"""
LegalRAG: Indian Evidence Act RAG Assistant (2026) ✅ LOGIN 100% WORKING
SIMPLEST VERSION - NO DEBUG, CLEAN LOGIN
"""

import os
import sys
import json
import uuid
from pathlib import Path
import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.hasher import Hasher

# PATHS
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_db"
CONFIG_PATH = BASE_DIR / "config.yaml"
HISTORY_FILE = BASE_DIR / "chat_history.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.environ["OTEL_PYTHON_DISABLED"] = "true"
os.environ.setdefault("CHROMA_PERSIST_DIRECTORY", str(CHROMA_DIR))

# App imports
from config.settings import settings
from src.ingestion.document_processor import load_documents, split_documents
from src.ingestion.vector_store import VectorStoreManager
from src.generation.rag_pipeline import answer_question

# HELPERS
def load_all_history():
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except:
            return {}
    return {}

def save_all_history(all_history):
    HISTORY_FILE.write_text(json.dumps(all_history, ensure_ascii=False, indent=2), encoding="utf-8")

def get_chat_title(messages):
    for msg in messages:
        if msg["role"] == "user":
            return msg["content"][:30] + "..." if len(msg["content"]) > 30 else msg["content"]
    return "New Chat"

def save_config(config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def list_source_files():
    files = []
    for d in [DATA_DIR, UPLOADS_DIR]:
        if d.exists():
            files += [p.name for p in d.iterdir() if p.is_file()]
    return sorted(set(files))

# MAIN APP - CLEAN LOGIN
def run_streamlit_app():
    st.set_page_config(page_title="LegalGPT", page_icon="⚖️", layout="wide", initial_sidebar_state="expanded")

    # CREATE CONFIG IF MISSING
    if not CONFIG_PATH.exists():
        config = {"credentials": {"usernames": {}}, "cookie": {"name": "legalgpt", "key": "legal_key_2026", "expiry_days": 30}}
        save_config(config)

    config = yaml.load(open(CONFIG_PATH), Loader=SafeLoader) or {}
    config["credentials"] = config.get("credentials", {})
    config["cookie"] = config.get("cookie", {"name": "legalgpt", "key": "legal_key_2026", "expiry_days": 30})

    # AUTH
    cookie_key = st.secrets.get("AUTH_COOKIE_KEY", config["cookie"]["key"])
    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        cookie_key,
        cookie_expiry_days=config["cookie"]["expiry_days"]
    )

    # SESSION INIT
    for key in ["authentication_status", "name", "username"]:
        if key not in st.session_state:
            st.session_state[key] = None

    # LOGIN SCREEN
    if st.session_state["authentication_status"] != "authenticated":
        st.title("⚖️ LegalGPT")
        st.markdown("### 🔐 **Please Login or Sign Up**")
        
        tab1, tab2 = st.tabs([" Login ", " Sign Up "])

        with tab1:
            name, authentication_status, username = authenticator.login("main")
            if authentication_status:
                st.session_state["authentication_status"] = "authenticated"
                st.session_state["name"] = name
                st.session_state["username"] = username
                st.success("✅ Logged in!")
                st.rerun()
            elif authentication_status == False:
                st.error("❌ Wrong credentials")

        with tab2:
            with st.form("signup"):
                fullname = st.text_input("Full Name")
                username_input = st.text_input("Username")
                password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                if st.form_submit_button("Create Account"):
                    if password != confirm_password:
                        st.error("Passwords don't match")
                    elif username_input in config["credentials"]["usernames"]:
                        st.error("Username exists")
                    else:
                        hashed_password = Hasher([password]).generate()[0]
                        config["credentials"]["usernames"][username_input] = {
                            "name": fullname,
                            "password": hashed_password
                        }
                        save_config(config)
                        st.session_state["name"] = fullname
                        st.session_state["username"] = username_input
                        st.session_state["authentication_status"] = "authenticated"
                        st.success("✅ Account created & logged in!")
                        st.rerun()

        st.stop()

    # APP STARTS HERE (LOGGED IN)
    name = st.session_state["name"]
    username = st.session_state["username"]

    # STYLE
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {background-color: #0e1117;}
    .stApp {background-color: #000;}
    section[data-testid="stSidebar"] > div > div > div > div > div {background-color: #111;}
    </style>""", unsafe_allow_html=True)

    # SIDEBAR
    with st.sidebar:
        st.markdown(f"**👋 {name}**")
        authenticator.logout("Logout", "sidebar")
        st.markdown("---")

        # Chat history
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
        if "chat_id" not in st.session_state:
            st.session_state["chat_id"] = str(uuid.uuid4())

        all_history = load_all_history()
        if st.session_state["chat_id"] not in all_history:
            all_history[st.session_state["chat_id"]] = []

        if st.button("➕ New Chat"):
            st.session_state["chat_id"] = str(uuid.uuid4())
            st.session_state["messages"] = []
            st.rerun()

        for chat_id, msgs in list(all_history.items())[-5:]:
            if msgs:
                title = get_chat_title(msgs)
                if st.button(title, key=chat_id):
                    st.session_state["chat_id"] = chat_id
                    st.session_state["messages"] = msgs.copy()
                    st.rerun()

    # MAIN
    st.title("⚖️ LegalGPT")
    st.caption("Indian Evidence Act RAG Assistant")

    # INDEX TOOLS
    with st.expander("🔧 Rebuild Index"):
        files = list_source_files()
        st.write(f"Found {len(files)} files")
        if files:
            st.write(files[:5])
        
        if st.button("🔨 INDEX DOCUMENTS"):
            docs = load_documents(str(UPLOADS_DIR)) or load_documents(str(DATA_DIR))
            if docs:
                chunks = split_documents(docs)
                VectorStoreManager().add_documents(chunks)
                st.success(f"✅ Indexed {len(chunks)} chunks!")
            else:
                st.error("No documents found!")

    # CHAT
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about Evidence Act..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                result = answer_question(prompt)
                answer = result.get("answer", "No results found")
            st.markdown(answer)

        st.session_state["messages"].append({"role": "assistant", "content": answer})
        all_history[st.session_state["chat_id"]] = st.session_state["messages"]
        save_all_history(all_history)
        st.rerun()

if __name__ == "__main__":
    run_streamlit_app()
