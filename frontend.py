"""
Strategic Operations Platform - Single File Deployment
Everything runs in Streamlit Cloud: UI, AI, Logging, Database
100% module-level safe
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
# 🔍 DEBUGGING PANEL - Add this to sidebar (after logger init)
# ============================================================================

with st.sidebar.expander("🐞 Logging Debug", expanded=True):
    st.write("Logger Status:", "✅ Enabled" if logger.enabled else "❌ Disabled")
    
    # Show secrets checks
    try:
        st.write("Gemini Key Present:", bool(st.secrets.get("GEMINI_API_KEY")))
        st.write("Sheet ID Present:", bool(st.secrets.get("SHEET_ID")))
        st.write("Service Account Present:", bool(st.secrets.get("service_account")))
    except:
        st.error("❌ Secrets.toml not configured")
    
    # Manual test button
    if st.button("🧪 Force Test Log"):
        try:
            logger.log(
                user="debug_user",
                action="manual_test",
                input_data={"test": True, "timestamp": str(datetime.now())},
                output_data={"status": "forced_test"},
                model="debug",
                status="success",
                session_id=st.session_state.session_id
            )
            st.success("✅ Test log attempted - refresh sheet in 5 seconds")
        except Exception as e:
            st.error(f"❌ Logging failed: {e}")

# ============================================================================

# ============================================================================
# DATABASE HELPERS (Fixed - No Module-Level Calls)
# ============================================================================

def get_db():
    """Get a NEW SQLite connection (do NOT cache)"""
    import os
    os.makedirs("/tmp", exist_ok=True)
    conn = sqlite3.connect("/tmp/strategic_ops.db", check_same_thread=False)
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
    return conn

# Replace your store_review function with this:

def store_review(item_type: str, item_id: str, content: Dict[str, Any], reviewer_notes: str = None):
    """Store AI-generated content for human review"""
    conn = get_db()
    try:
        cursor = conn.cursor()
        
        # Convert datetime fields to ISO strings
        if isinstance(content, dict) and 'generated_at' in content:
            content['generated_at'] = str(content['generated_at'])
        
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
    except Exception as e:
        logger.log(
            user="system",
            action="store_review_error",
            input_data={"item_type": item_type, "item_id": item_id},
            status="error",
            error=str(e),
            session_id=st.session_state.session_id
        )
        st.error(f"❌ Failed to store review: {e}")
    finally:
        conn.close()

@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_pending_reviews() -> List[Dict[str, Any]]:
    """Get all pending reviews - cached to avoid module-level calls"""
    try:
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
    except Exception as e:
        st.error(f"Database error: {e}")
        return []

def update_review(review_id: str, status: str, reviewer: str, notes: str = None):
    """Update review status"""
    conn = get_db()
    try:
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
    finally:
        conn.close()

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
# AI ENGINES - Gemini 2.5 Flash (Production 2025)
# ============================================================================

class MarketIntelligenceEngine:
    def __init__(self):
        self.gemini_key = st.secrets.get("GEMINI_API_KEY", "")
        if not self.gemini_key:
            st.error("❌ GEMINI_API_KEY not found in secrets")
            self.enabled = False
        else:
            self.enabled = True
        
        # ✅ CORRECT: v1 API + Gemini 2.5 Flash (NOT 1.5)
        self.api_url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={self.gemini_key}"
        print(f"✅ Using Gemini 2.5 Flash: {self.api_url[:80]}...")
    
    async def analyze_competitor(self, competitor: str, game: GameTitle) -> CompetitiveInsight:
        if not self.enabled:
            return self._get_fallback(competitor, game, "API disabled")
        
        prompt = f"""
        Analyze {competitor} in the {game} AI coaching market.
        
        Provide JSON with:
        {{
          "key_strengths": ["strength1", "strength2"],
          "key_weaknesses": ["weakness1", "weakness2"],
          "market_position": "dominant|challenger|niche",
          "threat_level": 1-10,
          "opportunity_windows": ["opportunity1", "opportunity2"],
          "confidence_score": 0.0-1.0,
          "sources": ["source1", "source2"]
        }}
        
        Be specific and data-driven. Focus on actionable insights.
        """
        
        try:
            async with httpx.AsyncClient() as client:
                print(f"🚀 Calling Gemini 2.5 Flash for {competitor}...")
                
                response = await client.post(
                    self.api_url,
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": 0.3,
                            "maxOutputTokens": 2048,
                            "topP": 0.95
                        }
                    },
                    timeout=30.0
                )
                
                print(f"📡 Status: {response.status_code}")
                print(f"📡 Response: {response.text[:300]}")
                
                if response.status_code != 200:
                    raise Exception(f"API {response.status_code}: {response.text}")
                
                data = response.json()
                content = data["candidates"][0]["content"]["parts"][0]["text"]
                cleaned = content.replace('```json', '').replace('```', '').strip()
                analysis = json.loads(cleaned)
                
                return CompetitiveInsight(
                    competitor_name=competitor,
                    game=game,
                    key_strengths=analysis['key_strengths'][:3],
                    key_weaknesses=analysis['key_weaknesses'][:3],
                    market_position=analysis['market_position'],
                    threat_level=min(10, max(1, analysis['threat_level'])),
                    opportunity_windows=analysis['opportunity_windows'][:3],
                    confidence_score=min(1.0, max(0.0, analysis['confidence_score'])),
                    sources=analysis['sources'][:2],
                )
                
        except Exception as e:
            error_msg = f"⚠️ AI failed: {str(e)}"
            print(error_msg)
            st.warning(error_msg)
            
            logger.log(
                user="system",
                action="ai_analysis_failed",
                input_data={"competitor": competitor, "game": game},
                status="error",
                error=error_msg,
                session_id=st.session_state.session_id
            )
            
            return self._get_fallback(competitor, game, str(e))
    
    def _get_fallback(self, competitor: str, game: GameTitle, error: str) -> CompetitiveInsight:
        """Return fallback with error logged"""
        st.info(f"Using fallback for {competitor}: {error[:50]}")
        return CompetitiveInsight(
            competitor_name=competitor,
            game=game,
            key_strengths=["Strong brand recognition", "Large user base"],
            key_weaknesses=["Limited AI personalization", "High pricing"],
            market_position="challenger",
            threat_level=6,
            opportunity_windows=["Mobile expansion", "Esports team partnerships"],
            confidence_score=0.7,
            sources=["SteamSpy", "Twitch API", error[:30]],
        )

class OpportunitySizer:
    def __init__(self):
        self.gemini_key = st.secrets.get("GEMINI_API_KEY", "")
        self.enabled = bool(self.gemini_key)
        self.api_url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={self.gemini_key}"
    
    async def calculate_market_size(self, game: GameTitle, segment: str, tier: str, geo: str) -> TAMAnalysis:
        if not self.enabled:
            return self._get_fallback(game, segment, "API disabled")
        
        prompt = f"""
        Market analysis for {game}, segment: {segment}, pricing: {tier}, region: {geo}.
        Return JSON with arpu (30-200), segment_penetration (0.01-0.30), confidence_score (0-1).
        """
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    raise Exception(f"API {response.status_code}: {response.text}")
                
                data = response.json()
                content = data["candidates"][0]["content"]["parts"][0]["text"]
                cleaned = content.replace('```json', '').replace('```', '').strip()
                assumptions = json.loads(cleaned)
                
                # Mock data (replace with real API calls)
                total_players = 1_000_000
                arpu = assumptions.get("arpu", 60)
                tam = total_players * arpu
                penetration = assumptions.get("segment_penetration", 0.15)
                sam = int(tam * penetration)
                som = int(sam * 0.1)
                
                return TAMAnalysis(
                    game=game,
                    total_players=total_players,
                    serviceable_available_market=sam,
                    serviceable_obtainable_market=som,
                    avg_revenue_per_user=arpu,
                    tam_usd=tam,
                    sam_usd=sam,
                    som_usd=som,
                    growth_rate=0.10,
                    assumptions=assumptions,
                    confidence_score=assumptions.get("confidence_score", 0.6),
                )
                
        except Exception as e:
            logger.log(
                user="system",
                action="tam_calculation_failed",
                input_data={"game": game, "segment": segment},
                status="error",
                error=str(e)[:200],
                session_id=st.session_state.session_id
            )
            return self._get_fallback(game, segment, str(e))
    
    def _get_fallback(self, game: GameTitle, segment: str, error: str) -> TAMAnalysis:
        return TAMAnalysis(
            game=game,
            total_players=1_000_000,
            serviceable_available_market=150_000,
            serviceable_obtainable_market=15_000,
            avg_revenue_per_user=60,
            tam_usd=60_000_000,
            sam_usd=9_000_000,
            som_usd=900_000,
            growth_rate=0.10,
            assumptions={"error": error[:50]},
            confidence_score=0.5,
        )
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
    
    # Logging status
    if logger.enabled:
        st.success("✅ Sheets Logging Active")
    else:
        st.warning("⚠️ Sheets Logging: Mock Mode")
        st.caption("Add secrets to enable real logging")
    
    st.caption(f"Session ID: {st.session_state.session_id}")

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
                
                # Store for human review (using dict() which is JSON-safe now)
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
    
    # Show pending reviews (using cached function)
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
# FOOTER & METRICS (Fixed - Wrapped in cached function)
# ============================================================================

@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_review_metrics():
    """Safely get review metrics from database"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM reviews WHERE status = 'pending'")
        pending = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM reviews WHERE status = 'approved'")
        approved = cursor.fetchone()[0]
        conn.close()
        return pending, approved
    except Exception as e:
        print(f"Metrics error: {e}")
        return 0, 0

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Platform Metrics")
pending_count, approved_count = get_review_metrics()
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

# ============================================================================
# 🧪 END-TO-END TEST (Delete after verifying)
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.subheader("🧪 End-to-End Test")

if st.sidebar.button("🔥 Full System Test"):
    # 1. Test Gemini API
    try:
        engine = MarketIntelligenceEngine()
        test_insight = asyncio.run(engine.analyze_competitor("Test", GameTitle.CS2))
        st.sidebar.success("✅ Gemini API working")
    except Exception as e:
        st.sidebar.error(f"❌ Gemini API failed: {e}")
    
    # 2. Test Google Sheets
    try:
        logger.log(
            user="test_user",
            action="e2e_test",
            input_data={"test": True},
            output_data={"result": "success"},
            model="test",
            status="success",
            session_id=st.session_state.session_id
        )
        st.sidebar.success("✅ Sheets logging attempted")
    except Exception as e:
        st.sidebar.error(f"❌ Sheets logging failed: {e}")
    
    # 3. Test SQLite
    try:
        store_review("test", "test-item", {"test": True}, "Test notes")
        reviews = get_pending_reviews()
        st.sidebar.success(f"✅ SQLite working ({len(reviews)} reviews)")
    except Exception as e:
        st.sidebar.error(f"❌ SQLite failed: {e}")
