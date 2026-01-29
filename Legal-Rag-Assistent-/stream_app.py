"""
LegalRAG: Indian Evidence Act RAG Assistant
Full-Stack Streamlit + Chroma + HuggingFace (2026) ✅ LOGIN FIXED
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
# PATHS (set BEFORE app imports)
# ----------------------------
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
os.environ.setdefault("DOCS_DIR", str(UPLOADS_DIR))
os.environ.setdefault("UPLOADS_DIR", str(UPLOADS_DIR))

# ✅ App imports
from config.settings import settings
from src.ingestion.document_processor import load_documents, split_documents
from src.ingestion.vector_store import VectorStoreManager
from src.generation.rag_pipeline import answer_question

# ----------------------------
# HELPERS
# ----------------------------
def load_all_history():
    if HISTORY_FILE.exists():
        try:
            data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}

def save_all_history(all_history):
    HISTORY_FILE.write_text(
        json.dumps(all_history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def get_chat_title(messages):
    for msg in messages:
        if msg["role"] == "user":
            return msg["content"][:28] + "..." if len(msg["content"]) > 28 else msg["content"]
    return "New Chat"

def save_config(config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

def list_source_files():
    files = []
    if DATA_DIR.exists():
        files += [p.name for p in DATA_DIR.iterdir() if p.is_file()]
    if UPLOADS_DIR.exists():
        files += [p.name for p in UPLOADS_DIR.iterdir() if p.is_file()]
    return sorted(set(files))

def load_docs_for_index():
    docs = load_documents(str(UPLOADS_DIR))
    if not docs:
        docs = load_documents(str(DATA_DIR))
    return docs

# ----------------------------
# MAIN APP ✅ BULLETPROOF AUTH + FORCE LOGIN FIX
# ----------------------------
def run_streamlit_app():
    st.set_page_config(
        page_title="LegalGPT - Evidence Act RAG",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if not CONFIG_PATH.exists():
        st.error("❌ config.yaml not found! Create it or use signup.")
        st.stop()

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.load(f, Loader=SafeLoader) or {}

    config.setdefault("credentials", {}).setdefault("usernames", {})
    config.setdefault("cookie", {})

    # ----------------------------  
    # 🔥 BULLETPROOF AUTHENTICATION + FORCE LOGIN DEBUG
    # ----------------------------
    cookie_name = config["cookie"].get("name", "legalgpt_auth")
    cookie_expiry_days = float(config["cookie"].get("expiry_days", 30))
    cookie_key = (
        st.secrets.get("AUTH_COOKIE_KEY", "")
        if hasattr(st, "secrets")
        else ""
    ) or config["cookie"].get("key", "")

    if not cookie_key:
        st.error("❌ Missing AUTH_COOKIE_KEY in Streamlit Cloud Secrets!")
        st.stop()

    authenticator = stauth.Authenticate(
        config["credentials"],
        cookie_name,
        cookie_key,
        cookie_expiry_days=cookie_expiry_days,
    )

    # ✅ FORCE DEBUG MODE - REMOVE AFTER TESTING
    SHOW_FORCE_LOGIN = st.sidebar.checkbox("💥 DEBUG: Force Login Screen", key="debug_force")

    # ✅ INITIALIZE ALL SESSION KEYS PROPERLY
    auth_keys = ["authentication_status", "name", "username"]
    for key in auth_keys:
        if key not in st.session_state:
            st.session_state[key] = None

    # 🔥 FORCE RESET BUTTON (Always available in sidebar)
    with st.sidebar:
        if st.button("🔄 RESET SESSION", key="reset_session", help="Force login screen"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state["authentication_status"] = None
            st.rerun()

    # 🛑 LOGIN SCREEN (FORCE IF DEBUG OR NOT AUTH)
    if st.session_state["authentication_status"] != "authenticated" or SHOW_FORCE_LOGIN:
        st.sidebar.markdown("---")
        with st.sidebar.expander("👤 Account", expanded=True):
            tab_login, tab_signup = st.tabs(["Login", "Sign up"])

            with tab_login:
                name, authentication_status, username = authenticator.login(
                    location="main", 
                    fields={"Form name": "Login"}
                )
                if authentication_status:
                    st.session_state["authentication_status"] = authentication_status
                    st.session_state["name"] = name
                    st.session_state["username"] = username
                    st.rerun()

            with tab_signup:
                with st.form("signup_form", clear_on_submit=True):
                    new_fullname = st.text_input("Full Name")
                    new_user = st.text_input("Username")
                    new_pass = st.text_input("Password", type="password")
                    new_pass2 = st.text_input("Confirm Password", type="password")
                    
                    if st.form_submit_button("Create Account"):
                        if new_pass != new_pass2:
                            st.error("❌ Passwords don't match!")
                        elif new_user in config["credentials"]["usernames"]:
                            st.error("❌ Username exists!")
                        else:
                            # ✅ CORRECT HASHER SYNTAX
                            hashed = Hasher([new_pass]).generate()[0]
                            config["credentials"]["usernames"][new_user] = {
                                "name": new_fullname, "password": hashed
                            }
                            save_config(config)
                            
                            # ✅ AUTO-LOGIN IMMEDIATELY
                            st.session_state["name"] = new_fullname
                            st.session_state["username"] = new_user
                            st.session_state["authentication_status"] = "authenticated"
                            
                            st.success(f"✅ Welcome {new_fullname}! 👋")
                            st.rerun()
        
        # DEBUG INFO
        with st.sidebar.expander("🐛 Debug Info"):
            st.write(f"Status: {st.session_state.get('authentication_status')}")
            st.write(f"Name: {st.session_state.get('name')}")
            st.write(f"Cookie Key: {'✅' if cookie_key else '❌'}")
        
        # STOP if still not authenticated
        if st.session_state["authentication_status"] != "authenticated":
            st.stop()
    
    # ✅ EXTRACT USER INFO (logged in)
    name = st.session_state["name"]
    username = st.session_state["username"]

    # CSS Styles - Dark Theme
    st.markdown("""
    <style>
    html, body, #root, .stApp, [data-testid="stSidebar"] { background-color: #171717 !important; }
    .stTextInput > div > div > div { background-color: #212121 !important; }
    .stButton > button { background-color: #212121 !important; color: #ececf1 !important; }
    * { border-color: #303030 !important; }
    </style>
    """, unsafe_allow_html=True)

    # Session ID Init
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state["messages"] = []

    # History Load/Save
    all_history = load_all_history()
    cur_sid = st.session_state["session_id"]
    if cur_sid not in all_history:
        all_history[cur_sid] = st.session_state["messages"]
        save_all_history(all_history)

    # 🔥 SIDEBAR - Chats + Profile + Logout
    with st.sidebar:
        if st.button("➕ New chat", use_container_width=True, type="secondary"):
            st.session_state["session_id"] = str(uuid.uuid4())
            st.session_state["messages"] = []
            st.rerun()

        st.caption("Your chats")
        for sid in list(all_history.keys())[::-1]:
            msgs = all_history[sid]
            if not msgs: continue
            title = get_chat_title(msgs)
            is_selected = (sid == st.session_state["session_id"])
            c1, c2 = st.columns([1, 0.2])
            with c1:
                if st.button(title, key=f"load_{sid}", use_container_width=True, 
                           type=("primary" if is_selected else "secondary")):
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
        
        st.markdown("<div style='flex-grow: 1; height: 50vh;'></div>", unsafe_allow_html=True)
        
        # 🔥 SAFE LOGOUT BUTTON
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            for key in auth_keys:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state["authentication_status"] = None
            st.rerun()
        
        # User Profile
        initials = (name[:2].upper() if name else "LG")
        st.markdown(f"""
        <div style='position: sticky; bottom: 20px; width: 100%; padding: 10px 12px;'>
          <div style='display: flex; align-items: center; gap: 10px;'>
            <div style='width: 36px; height: 36px; border-radius: 8px; background: #7b4ec9; color: #fff; 
                        display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 14px;'>
              {initials}
            </div>
            <div>
              <div style='color: #fff; font-size: 14px; font-weight: 600;'>{name}</div>
              <div style='color: #b4b4b4; font-size: 12px;'>Free Plan</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # MAIN CONTENT
    st.title("⚖️ LegalGPT")
    st.caption("Indian Evidence Act • Production RAG System")

    # 🛠️ ADMIN TOOLS / REBUILD INDEX
    with st.expander("🛠️ **Admin Tools / Rebuild Index**", expanded=False):
        vsm = VectorStoreManager()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Vectors Indexed", vsm.count())
        with col2:
            st.caption("Available Files:")
            files = list_source_files()
            if files:
                st.code("\n".join(files[:5]) + ("..." if len(files)>5 else ""), language="text")
            else:
                st.warning("⚠️ Add .txt files to data/uploads/")

        if st.button("🔄 **FORCE REBUILD INDEX NOW**", type="primary", use_container_width=True):
            with st.spinner("⏳ Indexing legal files..."):
                docs = load_docs_for_index()
                if not docs:
                    st.error("❌ No files found! Add txt files to data/uploads/")
                else:
                    chunks = split_documents(docs)
                    vsm.add_documents(chunks)
                    st.success(f"✅ Indexed {len(chunks)} chunks! Total: {vsm.count()}")
                    st.rerun()

    # CHAT INTERFACE
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("Ask about Evidence Act (e.g., 'What is Section 32?', 'murder evidence')..."):
        st.session_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("🔍 Searching legal database..."):
                result = answer_question(query)
                answer = result.get("answer", "No relevant sections found. Try rebuilding index.")
            placeholder.markdown(answer + "\n\n📚 *Powered by LegalRAG Pipeline*")

        st.session_state["messages"].append({"role": "assistant", "content": answer})
        all_history = load_all_history()
        all_history[st.session_state["session_id"]] = st.session_state["messages"]
        save_all_history(all_history)
        st.rerun()

if __name__ == "__main__":
    run_streamlit_app()
