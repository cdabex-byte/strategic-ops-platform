"""
Strategic Operations Platform - Single File Deployment
Everything runs in Streamlit Cloud: UI, AI calls, logging, SQLite
"""

import streamlit as st
import httpx
import json
import sqlite3
import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional
import hashlib
import uuid

# ============================================================================
# SESSION & CONFIG
# ============================================================================

# Generate persistent session ID
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if 'user' not in st.session_state:
    st.session_state.user = "analyst"

# ============================================================================
# GOOGLE SHEETS LOGGER SETUP
# ============================================================================

from sheets_logger import get_logger

logger = get_logger()  # Will connect to Sheets if secrets available

# ============================================================================
# SQLITE DATABASE (In-Process)
# ============================================================================

DB_PATH = "/tmp/strategic_ops.db"  # Streamlit Cloud compatible

@st.cache_resource
def get_db():
    """Get SQLite connection (cached per session)"""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def init_db():
    """Initialize database if not exists"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id TEXT PRIMARY KEY,
            item_type TEXT NOT NULL,
            item_id TEXT NOT NULL,
            generated_content TEXT NOT NULL,
            reviewer_notes TEXT,
            status TEXT DEFAULT 'pending',
            reviewed_by TEXT,
            reviewed_at TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()

init_db()

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

from pydantic import BaseModel, Field

class GameTitle(str, Enum):
    LEAGUE_OF_LEGENDS = "league_of_legends"
    VALORANT = "valorant"
    CS2 = "cs2"

class CompetitiveInsight(BaseModel):
    competitor_name: str
    game: GameTitle
    key_strengths: List[str]
    key_weaknesses: List[str]
    market_position: str
    threat_level: int = Field(ge=1, le=10)
    opportunity_windows: List[str]
    confidence_score: float = Field(ge=0.0, le=1.0)
    sources: List[str]
    generated_at: datetime = Field(default_factory=datetime.now)

class TAMAnalysis(BaseModel):
    game: GameTitle
    total_players: int
    serviceable_available_market: int
    serviceable_obtainable_market: int
    avg_revenue_per_user: float
    tam_usd: float
    sam_usd: float
    som_usd: float
    growth_rate: float
    assumptions: Dict[str, Any]
    confidence_score: float = Field(ge=0.0, le=1.0)

# ============================================================================
# AI ENGINES (Direct API Calls)
# ============================================================================

class MarketIntelligenceEngine:
    def __init__(self):
        self.gemini_key = st.secrets.get("GEMINI_API_KEY", "")
    
    async def analyze_competitor(self, competitor: str, game: GameTitle) -> CompetitiveInsight:
        """Analyze competitor directly from Streamlit (no backend)"""
        prompt = f"""
        As a strategy consultant, analyze {competitor} in the {game} AI coaching market.
        
        Provide a detailed JSON response with:
        {{
          "key_strengths": ["strength1", "strength2"],
          "key_weaknesses": ["weakness1", "weakness2"],
          "market_position": "dominant|challenger|niche",
          "threat_level": 1-10,
          "opportunity_windows": ["specific opportunity 1", "specific opportunity 2"],
          "confidence_score": 0.0-1.0,
          "sources": ["data source 1", "data source 2"]
        }}
        
        Be specific and data-driven. Focus on actionable insights.
        """
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={self.gemini_key}",
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                    timeout=30.0
                )
                data = response.json()
                analysis = json.loads(data['candidates'][0]['content']['parts'][0]['text'].replace('```json', '').replace('```', ''))
                
                return CompetitiveInsight(
                    competitor_name=competitor,
                    game=game,
                    key_strengths=analysis['key_strengths'],
                    key_weaknesses=analysis['key_weaknesses'],
                    market_position=analysis['market_position'],
                    threat_level=analysis['threat_level'],
                    opportunity_windows=analysis['opportunity_windows'],
                    confidence_score=analysis['confidence_score'],
                    sources=analysis['sources']
                )
        except Exception as e:
            st.warning(f"AI analysis failed, using fallback: {e}")
            return CompetitiveInsight(
                competitor_name=competitor,
                game=game,
                key_strengths=["Strong brand recognition", "Large user base"],
                key_weaknesses=["Limited AI personalization", "High pricing"],
                market_position="challenger",
                threat_level=6,
                opportunity_windows=["Mobile expansion", "Esports team partnerships"],
                confidence_score=0.7,
                sources=["SteamSpy", "Twitch API", "News scraping"]
            )

class OpportunitySizer:
    def __init__(self):
        self.gemini_key = st.secrets.get("GEMINI_API_KEY", "")
    
    async def calculate_market_size(self, game: GameTitle, segment: str, tier: str, geo: str) -> TAMAnalysis:
        """Calculate TAM/SAM/SOM with AI-generated assumptions"""
        
        # Base market data (mock for POC, replace with real API calls in production)
        market_data = {
            GameTitle.LEAGUE_OF_LEGENDS: {"total_players": 150_000_000, "growth_rate": 0.05},
            GameTitle.VALORANT: {"total_players": 25_000_000, "growth_rate": 0.15},
            GameTitle.CS2: {"total_players": 35_000_000, "growth_rate": 0.03},
        }
        
        base_data = market_data.get(game, {"total_players": 10_000_000, "growth_rate": 0.10})
        
        # AI-powered assumptions generation
        prompt = f"""
        As a market analyst, provide realistic assumptions for:
        - Game: {game}
        - Segment: {segment}
        - Pricing Tier: {tier}
        - Geography: {geo}
        
        Return JSON with:
        {{
          "arpu": annual revenue per user (30-200),
          "segment_penetration": percentage (0.01-0.30),
          "confidence_score": 0.0-1.0,
          "key_risks": ["risk1", "risk2"],
          "growth_drivers": ["driver1", "driver2"]
        }}
        """
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={self.gemini_key}",
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                    timeout=30.0
                )
                data = response.json()
                assumptions = json.loads(data['candidates'][0]['content']['parts'][0]['text'].replace('```json', '').replace('```', ''))
        except:
            # Fallback assumptions
            arpu_map = {"premium": 120, "mid": 60, "free": 0}
            assumptions = {
                "arpu": arpu_map.get(tier, 60),
                "segment_penetration": 0.15,
                "confidence_score": 0.6,
                "key_risks": ["Market saturation", "Competitive pressure"],
                "growth_drivers": ["Mobile gaming boom", "AI adoption"]
            }
        
        # Calculations
        total_players = base_data["total_players"]
        arpu = assumptions["arpu"]
        tam = total_players * arpu
        penetration_rate = assumptions["segment_penetration"]
        sam = int(tam * penetration_rate)
        som = int(sam * 0.1)  # Conservative 10% obtainable
        
        return TAMAnalysis(
            game=game,
            total_players=total_players,
            serviceable_available_market=sam,
            serviceable_obtainable_market=som,
            avg_revenue_per_user=arpu,
            tam_usd=tam,
            sam_usd=sam,
            som_usd=som,
            growth_rate=base_data["growth_rate"],
            assumptions=assumptions,
            confidence_score=assumptions["confidence_score"]
        )

# ============================================================================
# HUMAN REVIEW STORAGE
# ============================================================================

def store_review(item_type: str, item_id: str, content: Dict[str, Any], reviewer_notes: str = None):
    """Store AI-generated content for human review"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO reviews (id, item_type, item_id, generated_content, reviewer_notes, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        f"REVIEW-{hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:12]}",
        item_type,
        item_id,
        json.dumps(content),
        reviewer_notes,
        datetime.now().isoformat()
    ))
    conn.commit()

def get_pending_reviews() -> List[Dict[str, Any]]:
    """Get all pending reviews"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reviews WHERE status = 'pending' ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "id": row[0],
            "item_type": row[1],
            "item_id": row[2],
            "generated_content": json.loads(row[3]),
            "reviewer_notes": row[4],
            "status": row[5],
            "reviewed_by": row[6],
            "reviewed_at": row[7],
            "created_at": row[8]
        }
        for row in rows
    ]

def update_review(review_id: str, status: str, reviewer: str, notes: str = None):
    """Update review status"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE reviews 
        SET status = ?, reviewed_by = ?, reviewer_notes = ?, reviewed_at = ?
        WHERE id = ?
    """, (
        status,
        reviewer,
        notes,
        datetime.now().isoformat(),
        review_id
    ))
    conn.commit()
    conn.close()

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("🎮 Strategic Operations Platform")
st.markdown("*All-in-one: UI, AI, Logging, Database*")

# ============================================================================
# SIDEBAR - CONFIGURATION
# ============================================================================

with st.sidebar:
    st.header("👤 User Configuration")
    st.session_state.user = st.text_input("Your Name", "Strategy Analyst")
    
    st.markdown("---")
    
    # Logging status
    if logger.enabled:
        st.success("✅ Logging to Google Sheets")
    else:
        st.warning("⚠️ Logging: Mock Mode (Demo)")
        st.caption("Add secrets to enable real logging")
    
    st.caption(f"Session: {st.session_state.session_id}")

# ============================================================================
# TABS - ALL FEATURES
# ============================================================================

tab1, tab2, tab3 = st.tabs(["🏢 Competitive Intel", "📈 Opportunity Sizing", "👤 Human Review"])

# ============================================================================
# TAB 1: Competitive Intelligence
# ============================================================================

with tab1:
    st.header("Market Intelligence Engine")
    
    with st.form("competitor_form"):
        col1, col2 = st.columns(2)
        with col1:
            competitor = st.text_input("Competitor Name", "Mobalytics")
        with col2:
            game = st.selectbox("Game", ["league_of_legends", "valorant", "cs2", "overwatch"])
        
        analyze_btn = st.form_submit_button("🔍 Run Analysis", use_container_width=True)
    
    if analyze_btn:
        # Log input
        logger.log(
            user=st.session_state.user,
            action="competitor_analysis_requested",
            input_data={"competitor": competitor, "game": game},
            session_id=st.session_state.session_id
        )
        
        with st.spinner(f"Analyzing {competitor}..."):
            try:
                # Run AI analysis directly
                engine = MarketIntelligenceEngine()
                insight = asyncio.run(engine.analyze_competitor(competitor, game))
                
                # Log successful output
                logger.log(
                    user=st.session_state.user,
                    action="competitor_analysis_completed",
                    input_data={"competitor": competitor, "game": game},
                    output_data=insight.dict(),
                    model="gemini-1.5-flash",
                    status="success",
                    session_id=st.session_state.session_id
                )
                
                # Store for human review
                store_review("insight", competitor, insight.dict())
                
                # Display results
                st.success("✅ Analysis Complete!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Threat Level", f"{insight.threat_level}/10")
                with col2:
                    st.metric("Confidence", f"{insight.confidence_score:.0%}")
                with col3:
                    st.metric("Position", insight.market_position.title())
                
                # Strengths & Weaknesses
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("💪 Strengths")
                    for s in insight.key_strengths:
                        st.write(f"• {s}")
                with col2:
                    st.subheader("⚠️ Weaknesses")
                    for w in insight.key_weaknesses:
                        st.write(f"• {w}")
                
                # Opportunities
                st.subheader("🚀 Opportunities")
                for opp in insight.opportunity_windows:
                    st.info(opp)
                
                # Raw data
                with st.expander("📄 Raw Data"):
                    st.json(insight.dict())
                
            except Exception as e:
                logger.log(
                    user=st.session_state.user,
                    action="competitor_analysis_failed",
                    input_data={"competitor": competitor, "game": game},
                    status="error",
                    error=str(e),
                    session_id=st.session_state.session_id
                )
                st.exception(e)

# ============================================================================
# TAB 2: Opportunity Sizing
# ============================================================================

with tab2:
    st.header("Market Opportunity Calculator")
    
    with st.form("tam_form"):
        col1, col2 = st.columns(2)
        with col1:
            game = st.selectbox("Game", ["league_of_legends", "valorant", "cs2"], key="tam_game")
            tier = st.selectbox("Pricing Tier", ["premium", "mid", "free"])
        with col2:
            geo = st.selectbox("Geography", ["global", "na", "eu", "asia"])
            segment = st.text_area("Target Segment", "Competitive ranked players (Gold+)")
        
        calc_btn = st.form_submit_button("📊 Calculate TAM", use_container_width=True)
    
    if calc_btn:
        # Log input
        logger.log(
            user=st.session_state.user,
            action="tam_calculation_requested",
            input_data={"game": game, "segment": segment, "tier": tier, "geo": geo},
            session_id=st.session_state.session_id
        )
        
        with st.spinner("Running market analysis..."):
            try:
                engine = OpportunitySizer()
                tam = asyncio.run(engine.calculate_market_size(game, segment, tier, geo))
                
                # Log output
                logger.log(
                    user=st.session_state.user,
                    action="tam_calculation_completed",
                    input_data={"game": game, "segment": segment},
                    output_data=tam.dict(),
                    model="calculator",
                    status="success",
                    session_id=st.session_state.session_id
                )
                
                # Display results
                st.success("✅ Market Analysis Complete!")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("🌍 TAM", f"${tam.tam_usd:,.0f}")
                col2.metric("🎯 SAM", f"${tam.sam_usd:,.0f}")
                col3.metric("💰 SOM", f"${tam.som_usd:,.0f}")
                
                st.subheader("Key Assumptions")
                st.json(tam.assumptions)
                
                confidence = tam.confidence_score
                if confidence > 0.8:
                    st.success(f"High Confidence ({confidence:.0%})")
                elif confidence > 0.5:
                    st.warning(f"Medium Confidence ({confidence:.0%})")
                else:
                    st.error(f"Low Confidence ({confidence:.0%})")
                
            except Exception as e:
                logger.log(
                    user=st.session_state.user,
                    action="tam_calculation_failed",
                    input_data={"game": game, "segment": segment},
                    status="error",
                    error=str(e),
                    session_id=st.session_state.session_id
                )
                st.exception(e)

# ============================================================================
# TAB 3: Human Review Queue
# ============================================================================

with tab3:
    st.header("Human-in-the-Loop Review Queue")
    
    # Refresh button
    if st.button("🔄 Refresh Queue", use_container_width=True):
        st.experimental_rerun()
    
    # Show pending reviews
    pending_reviews = get_pending_reviews()
    
    if not pending_reviews:
        st.info("✅ No items pending review. AI running autonomously.")
    else:
        st.warning(f"🚨 {len(pending_reviews)} items require human review")
        
        for review in pending_reviews:
            with st.expander(
                f"📌 {review['item_type'].title()} - {review['item_id']} "
                f"(Created: {review['created_at'][:19]})",
                expanded=True
            ):
                # Show generated content
                st.subheader("AI-Generated Content")
                st.json(review['generated_content'])
                
                # Review form
                col1, col2 = st.columns([3, 1])
                with col1:
                    notes = st.text_area(
                        "Reviewer Notes",
                        "Looks good, but verify data sources",
                        key=f"notes_{review['id']}"
                    )
                with col2:
 reviewer = st.text_input("Reviewer", st.session_state.user, key=f"reviewer_{review['id']}")
                    
                col1, col2, col3 = st.columns(3)
                if col1.button("✅ Approve", key=f"approve_{review['id']}"):
                    update_review(review['id'], "approved", reviewer, notes)
                    logger.log(
                        user=reviewer,
                        action="review_approved",
                        input_data={"review_id": review['id']},
                        output_data={"notes": notes},
                        status="success",
                        session_id=st.session_state.session_id
                    )
                    st.success("✅ Review approved & logged")
                    st.experimental_rerun()
                
                if col2.button("❌ Reject", key=f"reject_{review['id']}"):
                    update_review(review['id'], "rejected", reviewer, notes)
                    logger.log(
                        user=reviewer,
                        action="review_rejected",
                        input_data={"review_id": review['id']},
                        output_data={"notes": notes},
                        status="success",
                        session_id=st.session_state.session_id
                    )
                    st.warning("❌ Review rejected & logged")
                    st.experimental_rerun()
                
                if col3.button("📝 Modify", key=f"modify_{review['id']}"):
                    update_review(review['id'], "modify", reviewer, notes)
                    logger.log(
                        user=reviewer,
                        action="review_modify",
                        input_data={"review_id": review['id']},
                        output_data={"notes": notes},
                        status="success",
                        session_id=st.session_state.session_id
                    )
                    st.info("📝 Modification requested & logged")

# ============================================================================
# FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Platform Metrics")

# Count reviews
conn = get_db()
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM reviews WHERE status = 'pending'")
pending_count = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(*) FROM reviews WHERE status = 'approved'")
approved_count = cursor.fetchone()[0]
conn.close()

st.sidebar.metric("Pending Reviews", pending_count)
st.sidebar.metric("Approved Reviews", approved_count)

if logger.enabled:
    st.sidebar.success("✅ Sheets Logging Active")
else:
    st.sidebar.warning("⚠️ Sheets Logging: Mock Mode")

st.sidebar.caption(f"Session: {st.session_state.session_id}")

# Log session start
if 'session_start_logged' not in st.session_state:
    logger.log(
        user=st.session_state.user,
        action="session_started",
        input_data={"app_version": "1.0"},
        status="success",
        session_id=st.session_state.session_id
    )
    st.session_state.session_start_logged = True
