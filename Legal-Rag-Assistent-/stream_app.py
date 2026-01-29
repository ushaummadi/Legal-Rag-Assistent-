"""
LegalRAG: Indian Evidence Act RAG Assistant (2026)
✅ FIXED: Profile -> Logout (At Bottom of Sidebar)
✅ FEATURES: Original UI, Dark Mode, Auto-Login, Admin Tools
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

# ----------------------------
# 1. PATHS & SETUP
# ----------------------------
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

# App imports (Ensure these files exist in your src folder)
from config.settings import settings
from src.ingestion.document_processor import load_documents, split_documents
from src.ingestion.vector_store import VectorStoreManager
from src.generation.rag_pipeline import answer_question

# ----------------------------
# 2. HELPER FUNCTIONS
# ----------------------------
def load_all_history():
    if HISTORY_FILE.exists():
        try: return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except: return {}
    return {}

def save_all_history(all_history):
    HISTORY_FILE.write_text(json.dumps(all_history, ensure_ascii=False, indent=2), encoding="utf-8")

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
        if d.exists(): files += [p.name for p in d.iterdir() if p.is_file()]
    return sorted(set(files))

# ----------------------------
# 3. MAIN APP
# ----------------------------
def run_streamlit_app():
    st.set_page_config(
        page_title="LegalGPT", 
        page_icon="⚖️", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )

    # 🎨 DARK THEME CSS
    st.markdown("""
    <style>
    /* Main Backgrounds */
    html, body, #root, .stApp, [data-testid="stSidebar"] { background-color: #171717 !important; }
    
    /* Inputs */
    .stTextInput > div > div > div { background-color: #212121 !important; color: #ececf1 !important; }
    
    /* Buttons */
    .stButton > button { background-color: #212121 !important; color: #ececf1 !important; border: 1px solid #303030; }
    .stButton > button:hover { border-color: #7b4ec9 !important; color: #7b4ec9 !important; }
    
    /* Borders & Text */
    * { border-color: #303030 !important; }
    h1, h2, h3, p, span { color: #ececf1 !important; }
    
    /* Sidebar Spacing */
    [data-testid="stSidebar"] > div > div { display: flex; flex-direction: column; height: 100%; }
    </style>
    """, unsafe_allow_html=True)

    # Config Check
    if not CONFIG_PATH.exists():
        save_config({
            "credentials": {"usernames": {}}, 
            "cookie": {"name": "legalgpt", "key": "legal_key", "expiry_days": 30}
        })

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.load(f, Loader=SafeLoader) or {}

    # ----------------------------
    # 🔐 AUTHENTICATION
    # ----------------------------
    cookie_key = st.secrets.get("AUTH_COOKIE_KEY", config.get("cookie", {}).get("key", "fallback_key"))
    authenticator = stauth.Authenticate(
        config["credentials"],
        config.get("cookie", {}).get("name", "legalgpt_auth"),
        cookie_key,
        cookie_expiry_days=float(config.get("cookie", {}).get("expiry_days", 30))
    )

    # Session Keys
    for key in ["authentication_status", "name", "username"]:
        if key not in st.session_state: st.session_state[key] = None

    # 🛑 LOGIN SCREEN
    if st.session_state["authentication_status"] != "authenticated":
        st.sidebar.markdown("---")
        with st.sidebar.expander("👤 Account", expanded=True):
            tab_login, tab_signup = st.tabs(["Login", "Sign up"])

            with tab_login:
                name, auth_status, username = authenticator.login(location="main")
                if auth_status:
                    st.session_state["authentication_status"] = "authenticated"
                    st.session_state["name"] = name
                    st.session_state["username"] = username
                    st.rerun()
                elif auth_status == False:
                    st.error("❌ Wrong credentials")

            with tab_signup:
                with st.form("signup_form", clear_on_submit=True):
                    new_fullname = st.text_input("Full Name")
                    new_user = st.text_input("Username")
                    new_pass = st.text_input("Password", type="password")
                    if st.form_submit_button("Create & Enter"):
                        if new_user in config["credentials"]["usernames"]:
                            st.error("Username exists!")
                        else:
                            hashed = Hasher([new_pass]).generate()[0]
                            config["credentials"]["usernames"][new_user] = {"name": new_fullname, "password": hashed}
                            save_config(config)
                            
                            # Auto Login
                            st.session_state["authentication_status"] = "authenticated"
                            st.session_state["name"] = new_fullname
                            st.session_state["username"] = new_user
                            st.rerun()
        st.stop()

    # ----------------------------
    # 📱 APP UI (LOGGED IN)
    # ----------------------------
    name = st.session_state["name"]
    username = st.session_state["username"]

    # History Init
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state["messages"] = []

    all_history = load_all_history()
    cur_sid = st.session_state["session_id"]
    if cur_sid not in all_history:
        all_history[cur_sid] = []

    # ----------------------------
    # 📂 SIDEBAR
    # ----------------------------
    with st.sidebar:
        # 1. New Chat
        if st.button("➕ New chat", use_container_width=True, type="secondary"):
            st.session_state["session_id"] = str(uuid.uuid4())
            st.session_state["messages"] = []
            st.rerun()

        # 2. Chat List
        st.caption("Your chats")
        for sid in list(all_history.keys())[::-1]:
            msgs = all_history[sid]
            if not msgs: continue
            title = get_chat_title(msgs)
            is_selected = (sid == cur_sid)
            
            c1, c2 = st.columns([1, 0.2])
            with c1:
                if st.button(title, key=f"load_{sid}", use_container_width=True, type=("primary" if is_selected else "secondary")):
                    st.session_state["session_id"] = sid
                    st.session_state["messages"] = msgs.copy()
                    st.rerun()
            with c2:
                if is_selected and st.button("✖", key=f"del_{sid}"):
                    del all_history[sid]
                    st.session_state["session_id"] = str(uuid.uuid4())
                    st.session_state["messages"] = []
                    save_all_history(all_history)
                    st.rerun()
        
        # 3. SPACER - Pushes everything below to bottom
        st.markdown("<div style='flex-grow: 1;'></div>", unsafe_allow_html=True)
        
        # 4. PROFILE SECTION
        initials = (name[:2].upper() if name else "LG")
        st.markdown(f"""
        <div style='margin-bottom: 10px; padding: 10px; border-radius: 8px; background-color: #262626;'>
          <div style='display: flex; align-items: center; gap: 10px;'>
            <div style='width: 32px; height: 32px; border-radius: 50%; background: #7b4ec9; color: #fff; 
                        display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 12px;'>
              {initials}
            </div>
            <div>
              <div style='color: #fff; font-size: 13px; font-weight: 600;'>{name}</div>
              <div style='color: #b4b4b4; font-size: 11px;'>Free Plan</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # 5. LOGOUT BUTTON (Strictly Below Profile)
        if st.button("🚪 Logout", use_container_width=True):
            for key in ["authentication_status", "name", "username"]:
                st.session_state[key] = None
            st.rerun()

    # ----------------------------
    # 💬 MAIN CONTENT
    # ----------------------------
    st.title("⚖️ LegalGPT")
    st.caption("Indian Evidence Act • Production RAG System")

    # Admin Tools
    with st.expander("🛠️ **Admin Tools / Rebuild Index**", expanded=False):
        vsm = VectorStoreManager()
        col1, col2 = st.columns(2)
        with col1: st.metric("📊 Vectors", vsm.count())
        with col2:
            files = list_source_files()
            st.code("\n".join(files[:3]) + "..." if files else "No files", language="text")

        if st.button("🔄 **FORCE REBUILD INDEX**", type="primary", use_container_width=True):
            with st.spinner("⏳ Indexing..."):
                docs = load_documents(str(UPLOADS_DIR)) or load_documents(str(DATA_DIR))
                if docs:
                    chunks = split_documents(docs)
                    vsm.add_documents(chunks)
                    st.success(f"✅ Indexed {len(chunks)} chunks!")
                    st.rerun()

    # Chat Messages
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if query := st.chat_input("Ask about Evidence Act (e.g., 'What is Section 32?')..."):
        st.session_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching..."):
                result = answer_question(query)
                answer = result.get("answer", "No results found.")
            st.markdown(answer + "\n\n📚 *Powered by LegalRAG Pipeline*")

        st.session_state["messages"].append({"role": "assistant", "content": answer})
        all_history[st.session_state["session_id"]] = st.session_state["messages"]
        save_all_history(all_history)
        st.rerun()

if __name__ == "__main__":
    run_streamlit_app()
