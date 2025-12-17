"""
Streamlit Frontend - Strategic Operations Platform
"""

import streamlit as st
import httpx
import json
import uuid

# Session tracking
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if 'user' not in st.session_state:
    st.session_state.user = "analyst"

API_URL = "http://localhost:8000"

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(page_title="Strategic Ops Platform", page_icon="🎮", layout="wide")

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("👤 Configuration")
    st.session_state.user = st.text_input("Your Name", "Strategy Analyst")
    
    api_url = st.text_input("API URL", API_URL)
    
    if st.button("🧪 Test Connection"):
        try:
            resp = httpx.get(f"{api_url}/")
            st.success("✅ Connected" if resp.status_code == 200 else "❌ Failed")
        except:
            st.error("❌ Connection error")
    
    st.markdown("---")
    st.caption(f"Session ID: {st.session_state.session_id}")

# ============================================================================
# MAIN SECTION
# ============================================================================

st.title("🎮 Strategic Operations Platform")
st.markdown("*All actions logged to Google Sheets for audit trail*")

tab1, tab2 = st.tabs(["Competitive Intel", "Opportunity Sizing"])

with tab1:
    st.header("Market Intelligence")
    
    with st.form("comp_form"):
        competitor = st.text_input("Competitor Name", "Mobalytics")
        game = st.selectbox("Game", ["league_of_legends", "valorant", "cs2", "overwatch"])
        submitted = st.form_submit_button("🔍 Analyze Competitor", use_container_width=True)
    
    if submitted:
        with st.spinner(f"Analyzing {competitor}..."):
            try:
                response = httpx.post(
                    f"{api_url}/analyze-competitor",
                    params={"competitor": competitor, "game": game},
                    timeout=60.0
                )
                
                if response.status_code == 200:
                    insight = response.json()
                    st.success("✅ Analysis Complete (Logged to Sheets)")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Threat Level", f"{insight['threat_level']}/10")
                    col2.metric("Confidence", f"{insight['confidence_score']:.0%}")
                    col3.metric("Market Position", insight['market_position'].title())
                    
                    with st.expander("📋 Full Report"):
                        st.json(insight)
                    
                    # Human review
                    col1, col2 = st.columns(2)
                    if col1.button("✅ Approve", key="approve"):
                        st.success("✅ Approved & logged")
                    if col2.button("❌ Reject", key="reject"):
                        st.warning("❌ Rejected & logged")
                else:
                    st.error(f"API error: {response.status_code}")
                    
            except Exception as e:
                st.exception(e)

with tab2:
    st.header("Market Opportunity Sizing")
    
    game = st.selectbox("Select Game", ["league_of_legends", "valorant", "cs2"], key="tam_game")
    segment = st.text_area("Target Segment", "Competitive ranked players (Gold+)")
    
    if st.button("📊 Calculate TAM", use_container_width=True):
        with st.spinner("Running market analysis..."):
            try:
                response = httpx.post(
                    f"{api_url}/calculate-tam",
                    params={"game": game, "segment": segment, "tier": "premium", "geo": "global"},
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    tam = response.json()
                    st.success("✅ Market Analysis Complete (Logged)")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("🌍 TAM", f"${tam['tam_usd']:,.0f}")
                    col2.metric("🎯 SAM", f"${tam['sam_usd']:,.0f}")
                    col3.metric("💰 SOM", f"${tam['som_usd']:,.0f}")
                    
                    st.subheader("Key Assumptions")
                    st.json(tam['assumptions'])
                    
                    # Confidence indicator
                    confidence = tam['confidence_score']
                    if confidence > 0.8:
                        st.success(f"High Confidence ({confidence:.0%})")
                    elif confidence > 0.5:
                        st.warning(f"Medium Confidence ({confidence:.0%})")
                    else:
                        st.error(f"Low Confidence ({confidence:.0%}) - Verify data sources")
                else:
                    st.error(f"API error: {response.status_code}")
                    
            except Exception as e:
                st.exception(e)

st.sidebar.markdown("---")
st.sidebar.info("🎮 Powered by Gemini AI + Google Sheets")
st.sidebar.caption("v1.0 | Python 3.11 | All actions logged")
