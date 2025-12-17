"""
Streamlit Frontend - Strategic Operations Platform
"""

import streamlit as st
import httpx
import asyncio
import json
from datetime import datetime
import uuid

# Session tracking
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if 'user' not in st.session_state:
    st.session_state.user = "anonymous"

API_URL = "http://localhost:8000"

# ============================================================================
# SIDEBAR - USER & CONFIG
# ============================================================================

with st.sidebar:
    st.header("👤 Configuration")
    user_name = st.text_input("Your Name", "Strategy Analyst")
    st.session_state.user = user_name
    
    api_url = st.text_input("API URL", API_URL)
    
    if st.button("🧪 Test Connection"):
        try:
            resp = httpx.get(f"{api_url}/")
            st.success("✅ Connected" if resp.status_code == 200 else "❌ Failed")
        except:
            st.error("❌ Connection error")
    
    st.markdown("---")
    st.caption(f"Session: {st.session_state.session_id}")

# ============================================================================
# MAIN UI
# ============================================================================

st.title("🎮 Strategic Operations Platform")
st.markdown("*Every action is logged to Google Sheets*")

tab1, tab2, tab3 = st.tabs(["Competitive Intel", "Opportunity Sizing", "OKRs"])

with tab1:
    st.header("Competitive Analysis")
    
    with st.form("comp_analysis"):
        competitor = st.text_input("Competitor", "Mobalytics")
        game = st.selectbox("Game", ["league_of_legends", "valorant", "cs2"])
        submitted = st.form_submit_button("🔍 Analyze")
    
    if submitted:
        with st.spinner("Analyzing..."):
            try:
                response = httpx.post(
                    f"{api_url}/analyze-competitor",
                    params={"competitor": competitor, "game": game},
                    timeout=60.0
                )
                
                if response.status_code == 200:
                    insight = response.json()
                    st.success("✅ Complete")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Threat", f"{insight['threat_level']}/10")
                    col2.metric("Confidence", f"{insight['confidence_score']:.0%}")
                    col3.metric("Position", insight['market_position'])
                    
                    st.json(insight)
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.exception(e)

with tab2:
    st.header("Market Sizing")
    
    if st.button("📊 Calculate TAM"):
        try:
            response = httpx.post(
                f"{api_url}/calculate-tam",
                params={"game": "valorant", "segment": "mobile"},
                timeout=30.0
            )
            
            if response.status_code == 200:
                tam = response.json()
                st.metric("TAM", f"${tam['tam_usd']:,.0f}")
                st.json(tam)
        except Exception as e:
            st.exception(e)

with tab3:
    st.header("OKR Tracker")
    
    objective = st.text_input("Objective", "Launch 3 integrations")
    
    if st.button("📝 Create OKR"):
        okr_data = {
            "objective": objective,
            "owner": user_name,
            "quarter": "Q1-2025",
            "priority": 5,
            "key_results": [{"description": "Ship product", "target": 100, "current": 0, "unit": "%"}],
            "status": "on_track"
        }
        
        try:
            response = httpx.post(f"{api_url}/okrs", json=okr_data)
            if response.status_code == 200:
                st.success("✅ OKR Created")
                st.json(response.json())
        except Exception as e:
            st.exception(e)

st.sidebar.markdown("---")
st.sidebar.info("Powered by Gemini AI + Google Sheets")
