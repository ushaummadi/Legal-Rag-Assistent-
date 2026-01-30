"""
LegalRAG: Indian Evidence Act RAG Assistant
Full-Stack Streamlit + Chroma + HuggingFace (2026)
✅ ORIGINAL UI + FIXED LOGIN (persists after refresh)
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

# ---------------------------- PATHS & CONFIG ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_db"
CONFIG_PATH = BASE_DIR / "config.yaml"
HISTORY_FILE = BASE_DIR / "chat_history.json"

for d in [DATA_DIR, UPLOADS_DIR, CHROMA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.environ["OTEL_PYTHON_DISABLED"] = "true"
os.environ.setdefault("CHROMA_PERSIST_DIRECTORY", str(CHROMA_DIR))

# App imports
from config.settings import settings
from src.ingestion.document_processor import load_documents, split_documents
from src.ingestion.vector_store import VectorStoreManager
from src.generation.rag_pipeline import answer_question

# ---------------------------- HELPERS ----------------------------
def load_all_history():
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except:
            return {}
    return {}

def save_all_history(all_history):
    HISTORY_FILE.write_text(
        json.dumps(all_history, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def get_chat_title(messages):
    for msg in messages:
        if msg["role"] == "user":
            return msg["content"][:28] + "..." if len(msg["content"]) > 28 else msg["content"]
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

# ---------------------------- MAIN APP ----------------------------
def run_streamlit_app():
    st.set_page_config(
        page_title="LegalGPT",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Dark Theme CSS
    st.markdown("""
    <style>
    html, body, #root, .stApp, [data-testid="stSidebar"] { background-color: #171717 !important; }
    .stTextInput > div > div > div { background-color: #212121 !important; }
    .stButton > button { background-color: #212121 !important; color: #ececf1 !important; }
    * { border-color: #303030 !important; }
    </style>
    """, unsafe_allow_html=True)

    # Load config
    if not CONFIG_PATH.exists():
        save_config({
            "credentials": {"usernames": {}},
            "cookie": {"name": "legalgpt", "key": "legal_key", "expiry_days": 30}
        })

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.load(f, Loader=SafeLoader) or {}

    # 🔥 FIXED AUTHENTICATION - ALWAYS CALL LOGIN FIRST
    authenticator = stauth.Authenticate(
        config["credentials"],
        config.get("cookie", {}).get("name", "legalgpt_auth"),
        config.get("cookie", {}).get("key", "legal_key"),
        cookie_expiry_days=float(config.get("cookie", {}).get("expiry_days", 30))
    )

    # ⚠️ CRITICAL: ALWAYS CALL LOGIN() FIRST - This reads cookie on refresh!
    name, authentication_status, username = authenticator.login(
        location="sidebar",  # Sidebar for clean layout
        key="main_login"     # Unique key
    )

    # Set session state from authenticator result
    st.session_state["authentication_status"] = authentication_status
    if name:
        st.session_state["name"] = name
    if username:
        st.session_state["username"] = username

    # Show login/signup ONLY if not authenticated
    if authentication_status != "authenticated":
        st.sidebar.markdown("### 🔐 Login Required")
        tab_login, tab_signup = st.tabs(["Login", "Sign up"])

        with tab_login:
            # Login already called above - handle status
            if authentication_status is False:
                st.error("❌ Wrong credentials")

        with tab_signup:
            with st.form("signup_form", clear_on_submit=True):
                new_fullname = st.text_input("Full Name")
                new_user = st.text_input("Username")
                new_pass = st.text_input("Password", type="password")
                if st.form_submit_button("Create Account"):
                    if new_user in config["credentials"]["usernames"]:
                        st.error("Username exists!")
                    else:
                        hashed = Hasher([new_pass]).generate()[0]
                        config["credentials"]["usernames"][new_user] = {
                            "name": new_fullname,
                            "password": hashed
                        }
                        save_config(config)
                        st.success("✅ Account created! Refresh or login.")
                        st.rerun()

        st.stop()  # Stop app until authenticated

    # ✅ AUTHENTICATED - Main App
    name, username = st.session_state["name"], st.session_state["username"]
    
    # Session state init (safe after auth)
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state["messages"] = []

    all_history = load_all_history()
    cur_sid = st.session_state["session_id"]
    if cur_sid not in all_history:
        all_history[cur_sid] = []

    # Sidebar: Chats + Profile
    with st.sidebar:
        if st.button("➕ New Chat", use_container_width=True, type="secondary"):
            st.session_state["session_id"] = str(uuid.uuid4())
            st.session_state["messages"] = []
            st.rerun()

        st.caption("💬 Your Chats")
        for sid in list(all_history.keys())[::-1]:
            msgs = all_history[sid]
            if not msgs: continue
            title = get_chat_title(msgs)
            is_selected = (sid == cur_sid)
            c1, c2 = st.columns([1, 0.15])
            with c1:
                if st.button(title, key=f"chat_{sid}", use_container_width=True,
                           type="primary" if is_selected else "secondary"):
                    st.session_state["session_id"] = sid
                    st.session_state["messages"] = msgs.copy()
                    st.rerun()
            with c2:
                if st.button("✖", key=f"del_{sid}", type="secondary"):
                    del all_history[sid]
                    save_all_history(all_history)
                    st.rerun()

        st.markdown("---")

        # Profile + Logout
        initials = (name or "LG")[:2].upper()
        st.markdown(f"""
        <div style='padding: 15px; border-top: 1px solid #303030;'>
          <div style='display: flex; align-items: center; gap: 12px;'>
            <div style='width: 40px; height: 40px; border-radius: 50%; 
                        background: linear-gradient(135deg, #7b4ec9, #4ec9d8); 
                        display: flex; align-items: center; justify-content: center; 
                        font-weight: bold; color: white; font-size: 16px;'>{initials}</div>
            <div>
              <div style='color: white; font-weight: 600; font-size: 14px;'>{name}</div>
              <div style='color: #a1a1aa; font-size: 12px;'>Free Plan</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚪 Logout", use_container_width=True):
            authenticator.logout(key="main_logout")
            st.rerun()

    # Main Content
    st.title("⚖️ LegalGPT")
    st.caption("Indian Evidence Act • Production RAG System")

    # Admin Tools
    with st.expander("🛠️ Admin Tools / Rebuild Index", expanded=False):
        vsm = VectorStoreManager()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Vectors Indexed", vsm.count())
        with col2:
            files = list_source_files()
            st.caption("Files:")
            st.code("\n".join(files[:8]) if files else "No files")

        if st.button("🔄 FORCE REBUILD INDEX", type="primary", use_container_width=True):
            with st.spinner("Indexing..."):
                docs = load_documents(str(UPLOADS_DIR)) or load_documents(str(DATA_DIR))
                if docs:
                    chunks = split_documents(docs)
                    vsm.add_documents(chunks)
                    st.success(f"✅ {len(chunks)} chunks indexed!")
                    st.rerun()

    # Chat Interface
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("Ask about Evidence Act, CrPC, Constitution..."):
        st.session_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching legal database..."):
                result = answer_question(query)
                answer = result.get("answer", "**No relevant sections found.**")

            st.markdown(answer + "\n\n📚 *Powered by LegalRAG Pipeline*")

        st.session_state["messages"].append({"role": "assistant", "content": answer})
        all_history[cur_sid] = st.session_state["messages"]
        save_all_history(all_history)
        st.rerun()


if __name__ == "__main__":
    run_streamlit_app()
