"""
LegalRAG: Indian Evidence Act RAG Assistant
Full-Stack Streamlit + Chroma + HuggingFace (2026) - FIXED LOGIN + RAG
"""
import sys
from pathlib import Path
import streamlit as st
import json
import uuid
import yaml
from yaml.loader import SafeLoader
from streamlit_authenticator.utilities.hasher import Hasher
import streamlit_authenticator as stauth  # ✅ For cookie persistence

# ✅ FIXED PATHS (absolute for Cloud)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_db"
CONFIG_PATH = BASE_DIR / "config.yaml"
HISTORY_FILE = BASE_DIR / "chat_history.json"

# Add to path
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# FIXED IMPORTS
from config.settings import settings
from src.ingestion.document_processor import load_documents, split_documents
from src.ingestion.vector_store import VectorStoreManager
from src.generation.rag_pipeline import answer_question

# --- HELPER FUNCTIONS (unchanged) ---
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

# --- MAIN APP ---
def run_streamlit_app():
    st.set_page_config(
        page_title="LegalGPT - Evidence Act RAG",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if not CONFIG_PATH.exists():
        st.error("❌ config.yaml not found!")
        st.stop()

    # ✅ FIXED: Load config + create authenticator
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.load(f, Loader=SafeLoader)
    
    config.setdefault("credentials", {}).setdefault("usernames", {})
    config.setdefault("cookie", {
        "name": "legalgpt_auth",
        "key": "some_random_string_super_secret_key_change_me",
        "expiry_days": 30
    })

    # ✅ FIXED AUTHENTICATOR WITH COOKIES (persists on refresh!)
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'], 
        config['cookie']['expiry_days']
    )

    # ✅ CRITICAL: Call login() - handles cookie restore automatically
    name, authentication_status, username = authenticator.login('main', 'Login')

    if authentication_status == False:
        st.error('❌ Username/password is incorrect')
        st.stop()
        
    elif authentication_status == None:
        st.warning('👈 Please enter your username and password on the left sidebar')
        st.stop()

    # ✅ SUCCESS - Set session state from authenticator
    st.session_state["authentication_status"] = True
    st.session_state["name"] = name
    st.session_state["username"] = username

    # Logout button (in sidebar later)
    authenticator.logout('Logout', 'main')

    # Your existing CSS (PASTE YOUR FULL CSS HERE - same as before)
    st.markdown("""
        <style>
        /* YOUR EXISTING CSS - paste the full #171717 styling + sidebar buttons here */
        html, body, #root, .stApp { background-color: #171717 !important; }
        /* ... rest of your CSS ... */
        </style>
    """, unsafe_allow_html=True)

    # Session init (unchanged)
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state["messages"] = []

    all_history = load_all_history()
    cur_sid = st.session_state["session_id"]
    if cur_sid not in all_history:
        all_history[cur_sid] = st.session_state["messages"]
        save_all_history(all_history)

    qp = st.query_params
    show_settings = (qp.get("menu") == "settings")

    # ✅ FIXED SIDEBAR WITH DEBUG + PATHS
    with st.sidebar:
        st.markdown("### 🔍 Debug RAG")
        
        # Create dirs
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        
        st.metric("📂 Uploads", len(list(UPLOADS_DIR.glob("*"))))
        st.metric("🗄️ Chroma", len(list(CHROMA_DIR.glob("*"))))
        
        # ✅ TEST VECTOR COUNT
        if st.button("🧪 Test Vectors", key="test_vec"):
            try:
                vsm = VectorStoreManager(persist_dir=str(CHROMA_DIR))
                count = vsm.count()
                st.success(f"✅ {count:,} vectors ready!")
            except Exception as e:
                st.error(f"❌ {str(e)[:80]}")

        # Your chat history buttons (unchanged)
        if st.button("➕ New chat", use_container_width=True, type="secondary"):
            # ... existing logic
            pass

        # Profile footer (your existing)
        initials = (name[:2].upper() if name else "LG")
        st.markdown(f"""
        <div style='position: sticky; bottom: 0; width: 100%; background: #171717; border-top: 1px solid #303030; padding: 10px 12px;'>
          <div style='display: flex; align-items: center; gap: 10px;'>
            <div style='width: 36px; height: 36px; border-radius: 8px; background: #7b4ec9; color: #fff; font-weight: 700; font-size: 14px; display: flex; align-items: center; justify-content: center;'>{initials}</div>
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

    # FIXED SETTINGS
    if show_settings:
        st.subheader("⚙️ Rebuild Index")
        if st.button("🔄 Rebuild Index", use_container_width=True, type="primary"):
            with st.spinner("Indexing..."):
                # ✅ FIXED PATHS
                docs = load_documents(str(UPLOADS_DIR))
                chunks = split_documents(docs)
                vsm = VectorStoreManager(persist_dir=str(CHROMA_DIR))
                vsm.add_documents(chunks)
                st.success(f"✅ {vsm.count():,} vectors indexed!")

    # CHAT (FIXED)
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("Ask about Evidence Act..."):
        st.session_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                # ✅ FIXED: Pass chroma_dir
                result = answer_question(query, chroma_dir=str(CHROMA_DIR))
                answer = result.get("answer", "")
            st.markdown(answer)

        st.session_state["messages"].append({"role": "assistant", "content": answer})
        all_history[st.session_state["session_id"]] = st.session_state["messages"]
        save_all_history(all_history)
        st.rerun()

if __name__ == "__main__":
    run_streamlit_app()
