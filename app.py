import streamlit as st
import sys
import asyncio
from pathlib import Path
from typing import Optional

# ============================================================================
# 1. IMMEDIATE UI RENDER - NO BLOCKING CODE
# ============================================================================

st.set_page_config(
    page_title="Strategic Ops Platform", 
    page_icon="🎮",
    layout="wide"
)

# Render UI instantly
st.title("🎮 Strategic Operations Platform")
st.markdown("*AI-powered Strategy & Operations Associate*")

# Status indicator
status_placeholder = st.empty()
status_placeholder.success("✅ App loaded and ready")

# ============================================================================
# 2. LAZY IMPORT SYSTEM - Only load when needed
# ============================================================================

@st.cache_resource(show_spinner=False)
def get_llm_client() -> Optional['LLMClient']:
    """Initialize LLM client once and cache it"""
    try:
        from llm_client import LLMClient
        return LLMClient(
            gemini_key=st.secrets.get("GEMINI_API_KEY", "demo-key"),
            hf_key=st.secrets.get("HF_API_KEY", "demo-key")
        )
    except Exception as e:
        st.error(f"LLM Client init failed: {e}")
        return None

@st.cache_resource(show_spinner=False)
def get_logger() -> Optional['SheetsLogger']:
    """Initialize logger once with fallback"""
    try:
        from sheets_logger import SheetsLogger
        import io
        import json
        
        # Try to load from secrets first
        service_account_info = st.secrets.get("service_account", None)
        sheet_id = st.secrets.get("SHEET_ID", None)
        
        if service_account_info and sheet_id:
            # Convert to dict if it's a string
            if isinstance(service_account_info, str):
                service_account_info = json.loads(service_account_info)
                
            return SheetsLogger(
                service_account_info=service_account_info,
                sheet_id=sheet_id
            )
        else:
            # Mock logger
            return MockLogger()
            
    except Exception as e:
        st.warning(f"Sheets logging disabled: {e}")
        return MockLogger()

class MockLogger:
    """Fallback logger that does nothing"""
    def log_interaction(self, **kwargs):
        st.toast("📊 Logging skipped (demo mode)", icon="⚠️")

# ============================================================================
# 3. ASYNC RUNNER - For LLM calls
# ============================================================================

def run_async(coroutine):
    """Run async function in Streamlit"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coroutine)

# ============================================================================
# 4. TABS - Main Features
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏢 Competitive Analysis",
    "📈 Opportunity Sizing", 
    "🎯 Strategic Initiatives",
    "📊 OKR Tracker",
    "⚙️ System Health"
])

# ============================================================================
# TAB 1: Competitive Analysis
# ============================================================================

with tab1:
    st.header("Competitive Intelligence")
    
    with st.form("competitor_form"):
        col1, col2 = st.columns(2)
        with col1:
            competitor = st.text_input("Competitor Name", "Mobalytics")
        with col2:
            game = st.selectbox("Game", ["league_of_legends", "valorant", "cs2", "overwatch"])
        
        analyze_btn = st.form_submit_button("🔍 Analyze Competitor", use_container_width=True)
    
    if analyze_btn:
        with st.spinner("Running AI analysis..."):
            try:
                llm = get_llm_client()
                if not llm:
                    st.error("LLM client not available")
                    st.stop()
                
                prompt = f"""
                Analyze {competitor} in the {game} AI coaching market.
                Provide concise JSON with:
                - strengths: [list]
                - weaknesses: [list]
                - market_position: string
                - threat_level: number 1-10
                - opportunities: [list]
                """
                
                result = run_async(
                    llm.generate_content(
                        prompt=prompt,
                        action="competitor_analysis",
                        user="strategic_analyst"
                    )
                )
                
                if result:
                    st.json(result)
                    logger = get_logger()
                    logger.log_interaction(
                        user="analyst",
                        action="competitor_analysis",
                        input_data={"competitor": competitor, "game": game},
                        output_data=result,
                        model=result.get("model", "unknown"),
                        status="success"
                    )
                else:
                    st.error("Analysis failed. Check API keys.")
                    
            except Exception as e:
                st.exception(e)

# ============================================================================
# TAB 2: Opportunity Sizing
# ============================================================================

with tab2:
    st.header("Market Opportunity Sizing (TAM)")
    
    with st.form("tam_form"):
        game_sel = st.selectbox("Game", ["lol", "valorant", "cs2"], key="tam_game")
        arpu = st.number_input("Annual Revenue Per User ($)", 30, 200, 60)
        penetration_rate = st.slider("Market Penetration %", 1, 25, 10) / 100
        
        calc_btn = st.form_submit_button("📊 Calculate TAM", use_container_width=True)
    
    if calc_btn:
        # Mock data (replace with real API calls later)
        player_bases = {"lol": 150_000_000, "valorant": 25_000_000, "cs2": 35_000_000}
        total_players = player_bases.get(game_sel, 10_000_000)
        
        tam_data = {
            "total_players": total_players,
            "serviceable_players": int(total_players * penetration_rate),
            "arpu_usd": arpu,
            "tam_usd": total_players * arpu,
            "sam_usd": int(total_players * penetration_rate * arpu),
            "assumptions": {
                "penetration_rate": penetration_rate,
                "segment": "Competitive ranked players",
                "confidence": "medium"
            }
        }
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Players", f"{tam_data['total_players']:,}")
        col2.metric("TAM", f"${tam_data['tam_usd']:,.0f}")
        col3.metric("SAM (Serviceable)", f"${tam_data['sam_usd']:,.0f}")
        
        st.json(tam_data)
        
        logger = get_logger()
        logger.log_interaction(
            user="analyst",
            action="tam_calculation",
            input_data={"game": game_sel, "arpu": arpu},
            output_data=tam_data,
            model="calculator",
            status="success"
        )

# ============================================================================
# TAB 3: Strategic Initiatives
# ============================================================================

with tab3:
    st.header("Strategic Initiatives")
    
    with st.form("initiative_form"):
        title = st.text_input("Initiative Title", "Expand into Asian Markets")
        hypothesis = st.text_area("Hypothesis", "Asian mobile market is 3x larger than NA/EU...")
        priority = st.slider("Priority (1=low, 10=critical)", 1, 10, 7)
        
        create_btn = st.form_submit_button("📝 Create Initiative", use_container_width=True)
    
    if create_btn:
        initiative = {
            "id": st.secrets.get("INITIATIVE_ID", f"INIT-{st.session_state.get('init_count', 0)}"),
            "title": title,
            "hypothesis": hypothesis[:200] + "...",
            "priority": priority,
            "status": "exploring",
            "created_at": st.session_state.get("session_start", "2025-01-01")
        }
        
        st.json(initiative)
        st.session_state['init_count'] = st.session_state.get('init_count', 0) + 1
        
        logger = get_logger()
        logger.log_interaction(
            user="strategist",
            action="initiative_created",
            input_data={"title": title, "priority": priority},
            output_data=initiative,
            model="manual",
            status="success"
        )

# ============================================================================
# TAB 4: OKR Tracker
# ============================================================================

with tab4:
    st.header("OKR Tracker (v0.2 Preview)")
    st.info("Full OKR tracking with SQLite backend coming in next update!")
    
    # Mock OKRs for demo
    mock_okrs = [
        {"objective": "Launch 3 new game titles", "progress": 67, "status": "🟢 On Track"},
        {"objective": "Secure 2 hardware partnerships", "progress": 25, "status": "🟡 At Risk"},
        {"objective": "Reach 200K users", "progress": 85, "status": "🟢 On Track"}
    ]
    
    for okr in mock_okrs:
        with st.expander(f"{okr['objective']} - {okr['status']}"):
            st.progress(okr['progress'])
            st.write(f"Progress: {okr['progress']}%")

# ============================================================================
# TAB 5: System Health
# ============================================================================

with tab5:
    st.header("System Health Check")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Environment")
        import platform
        st.write("Python:", platform.python_version())
        st.write("Platform:", platform.platform())
        st.write("Streamlit:", st.__version__)
        
        # Check secrets
        st.subheader("Secrets Status")
        required_secrets = ["GEMINI_API_KEY", "HF_API_KEY", "SHEET_ID"]
        for secret in required_secrets:
            if st.secrets.get(secret):
                st.success(f"✅ {secret}")
            else:
                st.warning(f"⚠️ {secret} (using demo)")
    
    with col2:
        st.subheader("Module Status")
        
        # Test imports
        modules = {
            "llm_client": "LLMClient",
            "sheets_logger": "SheetsLogger"
        }
        
        for module_name, class_name in modules.items():
            try:
                module = __import__(module_name)
                st.success(f"✅ {module_name}.{class_name}")
            except Exception as e:
                st.error(f"❌ {module_name}: {e}")

# ============================================================================
# 5. SESSION STATE INITIALIZATION
# ============================================================================

if 'session_start' not in st.session_state:
    st.session_state['session_start'] = st.session_state.get('last_rerun', '2025-01-01')

# ============================================================================
# 6. FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.info("🔄 All interactions are logged and tracked")
st.sidebar.caption("Built for Python 3.11 | trophi.ai POC")
