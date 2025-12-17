
import streamlit as st
import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from llm_client import LLMClient
from sheets_logger import init_logger

# Page config
st.set_page_config(
    page_title="Strategic Operations Platform",
    page_icon="🎮",
    layout="wide"
)

# Initialize session state
if 'logger' not in st.session_state:
    try:
        # For local development
        init_logger(
            service_account_file="service_account.json",
            sheet_id=st.secrets["SHEET_ID"]
        )
        from sheets_logger import logger
        st.session_state.logger = logger
        st.success("✅ Connected to Google Sheets")
    except Exception as e:
        st.warning(f"⚠️ Sheets logging disabled: {e}")
        st.session_state.logger = None

# Initialize LLM client
if 'llm_client' not in st.session_state:
    try:
        st.session_state.llm_client = LLMClient(
            gemini_key=st.secrets["GEMINI_API_KEY"],
            hf_key=st.secrets["HF_API_KEY"]
        )
        st.success("✅ LLM client ready")
    except Exception as e:
        st.error(f"❌ Failed to initialize LLM: {e}")
        st.stop()

# Sidebar
st.sidebar.title("🎮 Strategic Ops Platform")
st.sidebar.markdown("AI-powered Strategy & Operations Associate")

# Main interface
st.title("Welcome to Your AI-Powered Strategy Associate")

# Tabs for different functions
tab1, tab2, tab3, tab4 = st.tabs([
    "Competitive Analysis", 
    "Opportunity Sizing", 
    "Strategic Initiatives",
    "OKR Tracker"
])

with tab1:
    st.header("Competitive Intelligence")
    
    col1, col2 = st.columns(2)
    with col1:
        competitor = st.text_input("Competitor Name", "Mobalytics")
    with col2:
        game = st.selectbox("Game", ["lol", "valorant", "cs2", "overwatch"])
    
    if st.button("Analyze Competitor"):
        with st.spinner("Running AI analysis..."):
            prompt = f"""
            Analyze {competitor} in the {game} AI coaching market.
            Provide: strengths, weaknesses, market position (dominant/challenger/niche), 
            threat level (1-10), and 2-3 opportunity windows.
            Return as JSON.
            """
            
            result = asyncio.run(
                st.session_state.llm_client.generate_content(
                    prompt=prompt,
                    action="competitor_analysis",
                    user=st.session_state.get('user', 'default')
                )
            )
            
            if result:
                st.json(result)
                st.success("Analysis complete!")
            else:
                st.error("Analysis failed. Check logs.")

with tab2:
    st.header("Market Opportunity Sizing")
    
    game_sel = st.selectbox("Select Game", ["lol", "valorant", "cs2"], key="tam_game")
    segment = st.text_input("Target Segment", "Competitive players (ranked Gold+)")
    
    if st.button("Calculate TAM"):
        with st.spinner("Calculating market size..."):
            # Mock data for POC
            player_bases = {"lol": 150_000_000, "valorant": 25_000_000, "cs2": 35_000_000}
            total = player_bases.get(game_sel, 10_000_000)
            
            tam_data = {
                "total_players": total,
                "tam_usd": total * 60,  # $60 ARPU
                "assumptions": {
                    "segment": segment,
                    "arpu": 60,
                    "adoption_rate": 0.15
                }
            }
            
            st.json(tam_data)
            
            # Log to sheets
            if st.session_state.logger:
                st.session_state.logger.log_interaction(
                    user="analyst",
                    action="tam_calculation",
                    input_data={"game": game_sel, "segment": segment},
                    output_data=tam_data,
                    model="calculator",
                    status="success"
                )

with tab3:
    st.header("Strategic Initiatives")
    
    title = st.text_input("Initiative Title", "Expand into Mobile Market")
    hypothesis = st.text_area("Hypothesis", "Mobile gamers represent untapped TAM...")
    priority = st.slider("Priority", 1, 10, 5)
    
    if st.button("Create Initiative"):
        initiative = {
            "id": st.secrets.get("INITIATIVE_ID", "INIT-001"),
            "title": title,
            "hypothesis": hypothesis,
            "priority": priority,
            "status": "exploring"
        }
        
        st.json(initiative)
        
        if st.session_state.logger:
            st.session_state.logger.log_interaction(
                user="strategist",
                action="initiative_created",
                input_data={"title": title},
                output_data=initiative,
                model="manual",
                status="success"
            )

with tab4:
    st.header("OKR Tracker")
    st.info("OKR functionality requires backend integration. Coming in v0.2!")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("All interactions logged to Google Sheets for review.")
