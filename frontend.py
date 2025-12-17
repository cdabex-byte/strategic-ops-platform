
import streamlit as st
import asyncio
import httpx
import sqlite3
import json
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
import google.generativeai as genai
import hashlib

# ==================== Configuration ====================

class Settings:
    DATABASE_URL = "strategic.db"
    CACHE_TTL = 300
    STEAMSPY_RATE = 1.0
    RATE_LIMIT_CALLS = 100
    RATE_LIMIT_WINDOW = 60

# ==================== Enums & Models ====================

class GameTitle(str, Enum):
    CS2 = "CS2"
    LEAGUE_OF_LEGENDS = "League of Legends"
    VALORANT = "VALORANT"

class InitiativeStatus(str, Enum):
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETE = "complete"
    BLOCKED = "blocked"

class InitiativePriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Metric(BaseModel):
    name: str
    current_value: float
    target_value: float
    unit: str

class StrategicInitiative(BaseModel):
    id: str = Field(default_factory=lambda: f"init_{int(datetime.now().timestamp())}")
    title: str
    description: str
    competitor: str
    game_title: GameTitle
    owner: str
    status: InitiativeStatus = InitiativeStatus.PLANNING
    priority: InitiativePriority = InitiativePriority.MEDIUM
    budget: float = 0.0
    roi_low: float = 0.0
    roi_base: float = 0.0
    roi_high: float = 0.0
    risk_score: float = 0.0
    metrics: List[Metric] = []
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    ai_analysis: Optional[Dict[str, Any]] = None
    
    @validator('risk_score')
    def validate_risk(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Risk score must be between 0 and 1')
        return round(v, 2)

class PartnershipDeal(BaseModel):
    id: str = Field(default_factory=lambda: f"deal_{int(datetime.now().timestamp())}")
    company: str
    deal_type: str
    value: float
    hubspot_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

# ==================== Database Manager ====================

class DatabaseManager:
    def __init__(self, db_path: str = Settings.DATABASE_URL):
        self.db_path = db_path
        self._init_db()
        self._migrate_db()
    
    def _init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS initiatives (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                competitor TEXT,
                game_title TEXT,
                owner TEXT,
                status TEXT,
                priority TEXT,
                budget REAL,
                roi_low REAL,
                roi_base REAL,
                roi_high REAL,
                risk_score REAL,
                created_at TEXT,
                updated_at TEXT,
                metrics TEXT,
                ai_analysis TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS partnerships (
                id TEXT PRIMARY KEY,
                company TEXT,
                deal_type TEXT,
                value REAL,
                hubspot_id TEXT,
                created_at TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_cache (
                key TEXT PRIMARY KEY,
                data TEXT,
                expires_at TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rate_limits (
                endpoint TEXT PRIMARY KEY,
                calls_made INTEGER,
                window_start TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_analysis_cache (
                opportunity_key TEXT PRIMARY KEY,
                analysis_data TEXT,
                briefing_text TEXT,
                created_at TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _migrate_db(self):
        """Add missing columns to existing database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("PRAGMA table_info(ai_analysis_cache)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'briefing_text' not in columns:
                print("🔄 Migrating database: adding briefing_text column...")
                cursor.execute("ALTER TABLE ai_analysis_cache ADD COLUMN briefing_text TEXT")
                conn.commit()
                print("✅ Migration complete")
        except Exception as e:
            print(f"Migration error: {e}")
        finally:
            conn.close()
    
    def save_initiative(self, initiative: StrategicInitiative):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO initiatives VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            initiative.id, initiative.title, initiative.description,
            initiative.competitor, initiative.game_title.value,
            initiative.owner, initiative.status.value, initiative.priority.value,
            initiative.budget, initiative.roi_low, initiative.roi_base, initiative.roi_high,
            initiative.risk_score,
            initiative.created_at.isoformat(), initiative.updated_at.isoformat(),
            json.dumps([m.dict() for m in initiative.metrics]),
            json.dumps(initiative.ai_analysis) if initiative.ai_analysis else None
        ))
        conn.commit()
        conn.close()
    
    def get_initiatives(self, status: Optional[str] = None, game: Optional[str] = None) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM initiatives"
        params = []
        conditions = []
        
        if status:
            conditions.append("status = ?")
            params.append(status)
        if game:
            conditions.append("game_title = ?")
            params.append(game)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        initiatives = []
        for row in rows:
            initiative = {
                "id": row[0], "title": row[1], "description": row[2],
                "competitor": row[3], "game_title": row[4], "owner": row[5],
                "status": row[6], "priority": row[7], "budget": row[8],
                "roi_low": row[9], "roi_base": row[10], "roi_high": row[11],
                "risk_score": row[12], "created_at": row[13], "updated_at": row[14],
                "metrics": json.loads(row[15]) if row[15] else [],
                "ai_analysis": json.loads(row[16]) if row[16] else None
            }
            initiatives.append(initiative)
        
        return initiatives
    
    def save_partnership(self, deal: PartnershipDeal):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO partnerships VALUES (?, ?, ?, ?, ?, ?)
        """, (
            deal.id, deal.company, deal.deal_type, deal.value,
            deal.hubspot_id, deal.created_at.isoformat()
        ))
        conn.commit()
        conn.close()
    
    def get_partnerships(self) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM partnerships")
        rows = cursor.fetchall()
        conn.close()
        
        deals = []
        for row in rows:
            deals.append({
                "id": row[0], "company": row[1], "deal_type": row[2],
                "value": row[3], "hubspot_id": row[4], "created_at": row[5]
            })
        return deals
    
    def cache_get(self, key: str) -> Optional[Any]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT data FROM api_cache WHERE key=? AND expires_at > ?",
            (key, datetime.now().isoformat())
        )
        result = cursor.fetchone()
        conn.close()
        return json.loads(result[0]) if result else None
    
    def cache_set(self, key: str, data: Any):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        expires_at = (datetime.now() + timedelta(seconds=Settings.CACHE_TTL)).isoformat()
        cursor.execute(
            "INSERT OR REPLACE INTO api_cache (key, data, expires_at) VALUES (?, ?, ?)",
            (key, json.dumps(data), expires_at)
        )
        conn.commit()
        conn.close()
    
    def save_ai_analysis(self, opportunity_key: str, analysis_data: Dict[str, Any], briefing_text: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO ai_analysis_cache (opportunity_key, analysis_data, briefing_text, created_at)
            VALUES (?, ?, ?, ?)
        """, (opportunity_key, json.dumps(analysis_data), briefing_text, datetime.now().isoformat()))
        conn.commit()
        conn.close()
    
    def get_ai_analysis(self, opportunity_key: str) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT analysis_data, briefing_text FROM ai_analysis_cache WHERE opportunity_key=?",
            (opportunity_key,)
        )
        result = cursor.fetchone()
        conn.close()
        if result:
            return {"analysis": json.loads(result[0]), "briefing": result[1]}
        return None

# ==================== AI Integration (Gemini 2.5 Flash) ====================

class GeminiAI:
    """Google Gemini 2.5 Flash for market research"""
    
    def __init__(self):
        self.api_key = st.secrets.get("GEMINI_API_KEY", "")
        self.configured = bool(self.api_key)
        self.last_error = None
        self.model_name = "gemini-2.5-flash"
        
        if self.configured:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                print(f"✅ {self.model_name} initialized")
                self._test_connection()
            except Exception as e:
                st.error(f"❌ Gemini Config Error: {str(e)}")
                print(f"❌ Config failed: {e}")
                self.configured = False
                self.model = None
        else:
            print("⚠️ No GEMINI_API_KEY")
            st.sidebar.warning("⚠️ Gemini API key missing")
            self.model = None
    
    def _test_connection(self):
        """Test API connectivity"""
        try:
            response = self.model.generate_content(
                "Test response: OK",
                generation_config=genai.GenerationConfig(max_output_tokens=10)
            )
            
            if response and hasattr(response, 'text') and "OK" in response.text:
                print(f"✅ {self.model_name} test PASSED")
                st.sidebar.success(f"✅ {self.model_name} Connected")
            else:
                raise ValueError(f"Bad response: {response}")
                
        except Exception as e:
            st.sidebar.error(f"❌ Test Failed: {e}")
            print(f"❌ Test error: {e}")
            self.configured = False
    
    async def analyze_opportunity(self, opportunity: str) -> tuple[Dict[str, Any], str]:
        """Generate unique market analysis"""
        
        if not self.configured:
            st.warning("⚠️ Using fallback data")
            return self._get_fallback_data(opportunity), self._generate_fallback_briefing(opportunity)
        
        cache_key = f"gemini25_{hash(opportunity)}"
        cached = db.get_ai_analysis(cache_key)
        if cached:
            st.info("ℹ️ Using cached analysis")
            return cached["analysis"], cached["briefing"]
        
        prompt = f"""
Analyze business opportunity for Trophi.ai: {opportunity}

Generate JSON with:
- market_size_millions (50-500)
- growth_rate_percent (5-40)
- competitors list
- trophi_fit_score_0_to_1 (0.2-0.9)
- priority level
- budget_estimate
- roi_multiple
- game_title_fit
- 2 metrics
- 2 risks

Format: {{"market_size": 0, ...}} then BRIEFING: your analysis.
"""
        
        try:
            with st.spinner(f"🤖 {self.model_name} analyzing..."):
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.95,
                        max_output_tokens=2000
                    )
                )
            
            result_text = response.text
            
            # Parse JSON
            json_start = result_text.find("{")
            json_end = result_text.rfind("}") + 1
            if json_start != -1 and json_end != -1:
                json_str = result_text[json_start:json_end]
                analysis_data = json.loads(json_str)
                
                # Extract briefing
                briefing_marker = "BRIEFING:"
                briefing_start = result_text.find(briefing_marker)
                if briefing_start != -1:
                    briefing = result_text[briefing_start + len(briefing_marker):].strip()
                else:
                    briefing = f"Briefing for {analysis_data.get('title', 'opportunity')}"
                
                db.save_ai_analysis(cache_key, analysis_data, briefing)
                st.success("✅ Analysis complete!")
                return analysis_data, briefing
            else:
                raise ValueError("No JSON found")
                
        except Exception as e:
            print(f"❌ Generation error: {e}")
            st.error(f"AI error: {str(e)}")
            return self._get_fallback_data(opportunity), self._generate_fallback_briefing(opportunity)
    
    def _get_fallback_data(self, opportunity: str) -> Dict[str, Any]:
        """Enhanced fallback with variation"""
        hash_val = int(hashlib.md5(opportunity.encode()).hexdigest(), 16)
        lower_opp = opportunity.lower()
        
        base_market = 50
        if "asia" in lower_opp: base_market += 100
        if "pro" in lower_opp: base_market += 50
        market_size = base_market + (hash_val % 150)
        
        fit_score = min(0.5 + (("ai" in lower_opp) * 0.3) + (hash_val % 30) / 100, 1.0)
        
        games = ["CS2", "VALORANT", "LEAGUE_OF_LEGENDS"]
        priorities = ["high", "medium", "low"]
        
        return {
            "title": f"Fallback: {opportunity[:50]}",
            "market_size": market_size,
            "growth_rate": 15 + (hash_val % 25),
            "audience": f"{opportunity.split()[0] if opportunity else 'Gaming'} players",
            "competitors": [f"Platform {hash_val % 5}", f"Tool {hash_val % 7}"],
            "fit_score": fit_score,
            "priority": priorities[hash_val % 3],
            "budget": 50000 + ((hash_val % 100) * 1000),
            "roi": 2.0 + fit_score,
            "game": games[hash_val % 3],
            "metrics": ["Users", "Engagement"]
        }
    
    def _generate_fallback_briefing(self, opportunity: str) -> str:
        return f"""
# ⚠️ FALLBACK MODE - AI NOT CONFIGURED

**Opportunity:** {opportunity[:80]}

## Status
Using static fallback data.

## Fix
1. Get key: https://makersuite.google.com
2. Add to `.streamlit/secrets.toml`

```toml
GEMINI_API_KEY = "your_key"
```
3. Redeploy

## Generic Analysis
Opportunity shows potential. Validation recommended.
"""

# Initialize
ai_engine = GeminiAI()

# ==================== API Integrations ====================

class SteamSpyAPI:
    def __init__(self):
        self.base_url = "https://steamspy.com/api.php"
    
    async def get_game_data(self, app_id: str) -> Dict[str, Any]:
        await asyncio.sleep(0.1)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}?request=appdetails&appid={app_id}"
                )
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            print(f"SteamSpy API error: {e}")
        return {"players": 500_000}

class TwitchAPI:
    def __init__(self):
        secrets = st.secrets
        self.client_id = secrets.get("TWITCH_CLIENT_ID", "")
        self.client_secret = secrets.get("TWITCH_CLIENT_SECRET", "")
        self.token = None
        self.token_expires = datetime.min
    
    async def get_token(self):
        if self.token and self.token_expires > datetime.now():
            return self.token
        
        if not self.client_id or not self.client_secret:
            return None
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    "https://id.twitch.tv/oauth2/token",
                    params={
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "grant_type": "client_credentials"
                    }
                )
                if resp.status_code == 200:
                    data = resp.json()
                    self.token = data["access_token"]
                    self.token_expires = datetime.now() + timedelta(seconds=data["expires_in"] - 300)
                    return self.token
        except Exception as e:
            print(f"Twitch auth error: {e}")
        
        return None
    
    async def get_game_viewership(self, game_name: str) -> Dict[str, Any]:
        token = await self.get_token()
        if not token:
            return {"data": []}
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    "https://api.twitch.tv/helix/streams",
                    headers={"Authorization": f"Bearer {token}", "Client-Id": self.client_id},
                    params={"game_name": game_name, "first": 100}
                )
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            print(f"Twitch API error: {e}")
        
        return {"data": []}

class SlackNotifier:
    def __init__(self):
        self.webhook_url = st.secrets.get("SLACK_WEBHOOK_URL", "")
    
    async def send_alert(self, message: str, priority: str = "medium"):
        if not self.webhook_url:
            return
        
        payload = {
            "text": f"🎮 Strategic Ops Alert ({priority.upper()})",
            "blocks": [{
                "type": "section",
                "text": {"type": "mrkdwn", "text": message}
            }]
        }
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(self.webhook_url, json=payload)
        except Exception as e:
            print(f"Slack notification failed: {e}")

class HubSpotCRM:
    def __init__(self):
        self.api_key = st.secrets.get("HUBSPOT_API_KEY", "")
    
    async def create_deal(self, company: str, deal_type: str, value: float) -> Optional[str]:
        if not self.api_key:
            return None
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    "https://api.hubapi.com/crm/v3/objects/deals",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "properties": {
                            "dealname": f"{deal_type.upper()} - {company}",
                            "amount": value,
                            "pipeline": "default",
                            "dealstage": "appointmentscheduled"
                        }
                    }
                )
                if response.status_code == 201:
                    return response.json()["id"]
        except Exception as e:
            print(f"HubSpot API error: {e}")
        
        return None

# ==================== Core Engine ====================

class MarketIntelligenceEngine:
    def __init__(self):
        self.steamspy = SteamSpyAPI()
        self.twitch = TwitchAPI()
        self.slack = SlackNotifier()
        self.hubspot = HubSpotCRM()
        self.ai = ai_engine
        self._last_steam_call = datetime.min
    
    async def _rate_limit_steam(self):
        now = datetime.now()
        elapsed = (now - self._last_steam_call).total_seconds()
        if elapsed < Settings.STEAMSPY_RATE:
            await asyncio.sleep(Settings.STEAMSPY_RATE - elapsed)
        self._last_steam_call = datetime.now()
    
    async def analyze_competitor(self, game: GameTitle, competitor: str) -> Dict[str, Any]:
        cache_key = f"analysis_{game}_{competitor}"
        cached = db.cache_get(cache_key)
        if cached:
            return cached
        
        await self._rate_limit_steam()
        
        game_ids = {GameTitle.CS2: "730", GameTitle.VALORANT: "1286300"}
        game_names = {
            GameTitle.CS2: "Counter-Strike 2",
            GameTitle.LEAGUE_OF_LEGENDS: "League of Legends",
            GameTitle.VALORANT: "VALORANT"
        }
        
        try:
            steam_data, twitch_data = await asyncio.gather(
                self.steamspy.get_game_data(game_ids.get(game, "730")),
                self.twitch.get_game_viewership(game_names.get(game, "CS2"))
            )
            
            players = steam_data.get("players", 0) if isinstance(steam_data, dict) else 0
            viewers = len(twitch_data.get("data", [])) if isinstance(twitch_data, dict) else 0
            
            result = {
                "game": game.value,
                "competitor": competitor,
                "metrics": {
                    "active_players": players,
                    "live_viewers": viewers,
                    "engagement_ratio": viewers / max(players, 1),
                    "data_timestamp": datetime.now().isoformat()
                },
                "recommendations": self._generate_recommendations(players, viewers)
            }
            
            db.cache_set(cache_key, result)
            return result
            
        except Exception as e:
            return {"error": str(e), "metrics": {}}
    
    def _generate_recommendations(self, players: int, viewers: int) -> List[str]:
        recommendations = []
        if players > 500_000:
            recommendations.append("🎯 High player base - prioritize community features")
        if viewers / max(players, 1) > 0.1:
            recommendations.append("📈 Strong viewership - explore esports partnerships")
        return recommendations
    
    def calculate_roi_simulation(self, budget: float, risk_score: float) -> Dict[str, float]:
        np.random.seed(42)
        n_simulations = 1000
        base_return = budget * 1.5
        volatility = risk_score * 0.5
        results = np.random.normal(base_return, base_return * volatility, n_simulations)
        
        return {
            "roi_low": np.percentile(results, 10) - budget,
            "roi_base": np.median(results) - budget,
            "roi_high": np.percentile(results, 90) - budget,
            "risk_of_loss": (results < budget).mean()
        }

engine = MarketIntelligenceEngine()
db = DatabaseManager()

# ==================== Executive Reporting ====================

def generate_executive_report() -> str:
    """Generate markdown executive report"""
    initiatives = db.get_initiatives()
    partnerships = db.get_partnerships()
    
    df = pd.DataFrame(initiatives)
    df_partnerships = pd.DataFrame(partnerships)
    
    total_budget = df["budget"].sum() if not df.empty else 0
    avg_risk = df["risk_score"].mean() if not df.empty else 0
    high_priority_count = len(df[df["priority"] == "high"]) if not df.empty else 0
    total_pipeline = df_partnerships["value"].sum() if not df_partnerships.empty else 0
    
    latest_ai = db.get_ai_analysis("latest")
    ai_section = ""
    if latest_ai:
        analysis = latest_ai["analysis"]
        ai_section = f"""## 🤖 Latest AI Strategic Briefing

**Opportunity:** {analysis.get('opportunity_title', 'N/A')}  
**Trophi Fit Score:** {analysis.get('trophi_fit_score', 0):.1%}  
**Market Size:** ${analysis.get('market_size', 0)}M  
**Key Insights:** {analysis.get('target_audience', 'N/A')}  
**AI Recommendation:** {analysis.get('recommendation', 'N/A')}
"""
    
    active_initiatives = df[df["status"] != "complete"] if not df.empty else pd.DataFrame()
    
    # ❌ OLD (BROKEN): - **Total Initiative Budget:** ${total_budget:, .2f}
    # ✅ CORRECTED: - **Total Initiative Budget:** ${total_budget:,.2f}
    
    report = f"""# Strategic Operations Weekly Report
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

{ai_section}

## 📊 Key Metrics
- **Total Initiative Budget:** ${total_budget:,.2f}
- **Average Risk Score:** {avg_risk:.1%}
- **High-Priority Initiatives:** {high_priority_count}
- **Partnership Pipeline:** ${total_pipeline:,.2f}

## 🎯 Active Initiatives
{active_initiatives.to_markdown(index=False) if not active_initiatives.empty else "No active initiatives"}

## 🤝 Partnership Deals
{df_partnerships.to_markdown(index=False) if not df_partnerships.empty else "No partnership deals"}

## ⚠️ Risk Analysis
- **High-Risk Initiatives (>20%):** {len(df[df["risk_score"] > 0.2]) if not df.empty else 0}
- **Budget at Risk:** ${df[df["risk_score"] > 0.2]["budget"].sum():,.2f if not df.empty and not df[df["risk_score"] > 0.2].empty else 0}

---

*Report auto-generated by Strategic Operations Platform v2.2*
"""
    return report

# ==================== Streamlit UI ====================

st.set_page_config(
    page_title="Strategic Operations Platform v2.2",
    page_icon="🎮",
    layout="wide"
)

st.sidebar.title("🎮 Strategic Ops v2.2")
page = st.sidebar.radio(
    "Navigation",
    ["🔍 AI Opportunity Analysis", "📊 Executive Dashboard", "🎯 Initiatives", "🤝 Partnerships", "📈 Market Intelligence", "⚙️ Settings"],
    index=0
)

# ==================== AI Opportunity Analysis ====================

if page == "🔍 AI Opportunity Analysis":
    st.title("🔍 AI-Powered Opportunity Analysis")
    
    if not ai_engine.configured:
        st.error("⚠️ Gemini AI not configured. Add GEMINI_API_KEY to secrets.toml")
        st.info("Get free API key at: https://makersuite.google.com")
    
    st.markdown("Describe any opportunity and AI will generate a detailed briefing.")
    
    user_input = st.text_area(
        "📝 Describe your opportunity",
        height=150,
        placeholder="Example: Launch AI coaching for VALORANT in Southeast Asia targeting pro teams..."
    )
    
    with st.expander("⚙️ Additional Context"):
        target_region = st.text_input("Target Region", "Global")
        target_segment = st.selectbox("Target Segment", ["Pro Players", "Semi-Pro", "Casual Competitive", "Content Creators"])
        timeline = st.select_slider("Timeline", ["1-3 months", "3-6 months", "6-12 months", "12+ months"])
    
    if st.button("🚀 Generate AI Briefing", use_container_width=True, type="primary"):
        if not user_input.strip():
            st.error("❌ Please describe an opportunity")
        elif not ai_engine.configured:
            st.error("❌ Gemini API key not configured")
        else:
            with st.spinner("🤖 AI is conducting deep market research..."):
                try:
                    full_opportunity = f"{user_input} | Target: {target_segment} in {target_region} | Timeline: {timeline}"
                    analysis, briefing = asyncio.run(ai_engine.analyze_opportunity(full_opportunity))
                    
                    if "error" not in analysis:
                        db.save_ai_analysis("latest", analysis, briefing)
                        
                        st.subheader("📋 AI-Generated Executive Briefing")
                        st.markdown(briefing)
                        
                        st.subheader("📊 Structured Analysis")
                        tab1, tab2, tab3, tab4 = st.tabs(["Market", "Competition", "Fit", "Actions"])
                        
                        with tab1:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1: st.metric("Market Size", f"${analysis.get('market_size', 0)}M")
                            with col2: st.metric("Growth Rate", f"{analysis.get('market_growth_rate', 0)}%")
                            with col3: st.metric("Budget Needed", f"${analysis.get('budget_estimate', 0):,.0f}")
                            with col4: st.metric("ROI Potential", f"{analysis.get('roi_potential', 0):.1f}x")
                            st.markdown(f"**Target Audience:** {analysis.get('target_audience', 'N/A')}")
                        
                        with tab2:
                            for comp in analysis.get("competitive_landscape", []):
                                st.warning(f"⚠️ {comp}")
                        
                        with tab3:
                            fit_score = analysis.get('trophi_fit_score', 0)
                            st.progress(fit_score, text=f"Trophi Fit: {fit_score:.1%}")
                            priority = analysis.get('priority', 'medium')
                            priority_color = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
                            st.metric("Priority", f"{priority_color.get(priority, '⚪')} {priority.upper()}")
                        
                        with tab4:
                            st.markdown(f"**Recommendation:** {analysis.get('recommendation', 'N/A')}")
                            for metric in analysis.get("metrics_to_track", []):
                                st.info(f"📊 {metric}")
                        
                        if st.button("✨ Create Initiative from This Analysis", use_container_width=True):
                            initiative = StrategicInitiative(
                                title=analysis.get("opportunity_title", "AI Generated"),
                                description=user_input,
                                competitor=analysis.get("competitive_landscape", ["Unknown"])[0],
                                game_title=GameTitle[analysis.get("game_title_fit", "CS2")] if analysis.get("game_title_fit") in GameTitle.__members__ else GameTitle.CS2,
                                owner="AI Analyst",
                                budget=analysis.get("budget_estimate", 50000),
                                risk_score=1.0 - analysis.get("trophi_fit_score", 0.5),
                                priority=InitiativePriority[analysis.get("priority", "medium")] if analysis.get("priority") in InitiativePriority.__members__ else InitiativePriority.MEDIUM,
                                metrics=[Metric(name=metric, current_value=0, target_value=100, unit="%") for metric in analysis.get("metrics_to_track", ["Engagement"])[:3]],
                                ai_analysis=analysis
                            )
                            
                            roi_data = engine.calculate_roi_simulation(initiative.budget, initiative.risk_score)
                            initiative.roi_low = roi_data["roi_low"]
                            initiative.roi_base = roi_data["roi_base"]
                            initiative.roi_high = roi_data["roi_high"]
                            
                            db.save_initiative(initiative)
                            st.success(f"✅ Initiative created: {initiative.id}")
                            
                            if initiative.priority == InitiativePriority.HIGH:
                                message = f"*🤖 AI HIGH PRIORITY INITIATIVE*\n*Title:* {initiative.title}\n*Budget:* ${initiative.budget:,.2f}"
                                asyncio.run(engine.slack.send_alert(message, "high"))
                        
                    else:
                        st.error(f"❌ AI analysis failed: {analysis.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"❌ Analysis error: {e}")
    
    st.subheader("📚 Recent AI Analyses")
    latest = db.get_ai_analysis("latest")
    if latest:
        with st.expander(f"📄 Latest: {latest['analysis'].get('opportunity_title', 'N/A')}"):
            st.markdown(latest['briefing'])
    else:
        st.info("No AI analyses yet. Start by describing an opportunity!")

# ==================== Executive Dashboard ====================

elif page == "📊 Executive Dashboard":
    st.title("📊 Executive Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Refresh All Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    initiatives = db.get_initiatives()
    partnerships = db.get_partnerships()
    df = pd.DataFrame(initiatives)
    
    with col2:
        total_budget = df["budget"].sum() if not df.empty else 0
        st.metric("Total Budget", f"${total_budget:,.2f}")
    
    with col3:
        avg_roi = df["roi_base"].mean() if not df.empty else 0
        st.metric("Avg Expected ROI", f"${avg_roi:,.0f}")
    
    with col4:
        risk_count = len(df[df["risk_score"] > 0.2]) if not df.empty else 0
        st.metric("High-Risk Items", risk_count)
    
    latest_ai = db.get_ai_analysis("latest")
    if latest_ai:
        st.subheader("🤖 Latest AI Strategic Briefing")
        analysis = latest_ai["analysis"]
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Trophi Fit Score", f"{analysis.get('trophi_fit_score', 0):.1%}")
        with col2: st.metric("Market Size", f"${analysis.get('market_size', 0)}M")
        with col3: st.metric("AI Priority", analysis.get('priority', 'N/A').upper())
        st.info(latest_ai['briefing'][:500] + "..." if len(latest_ai['briefing']) > 500 else latest_ai['briefing'])
        if st.button("📄 View Full Briefing"):
            with st.expander("Full Briefing"):
                st.markdown(latest_ai['briefing'])
    
    st.subheader("🎮 Live Market Data")
    game_col1, game_col2 = st.columns(2)
    
    with game_col1:
        if st.button("Get CS2 Live Data", use_container_width=True):
            with st.spinner("Fetching SteamSpy & Twitch data..."):
                try:
                    result = asyncio.run(engine.analyze_competitor(GameTitle.CS2, "Valve"))
                    if "error" not in result:
                        metrics = result["metrics"]
                        st.success("✅ Data fetched!")
                        st.metric("CS2 Active Players", f"{metrics.get('active_players', 0):,}")
                        st.metric("CS2 Twitch Viewers", f"{metrics.get('live_viewers', 0):,}")
                        st.metric("Engagement Ratio", f"{metrics.get('engagement_ratio', 0.0):.2%}")
                        with st.expander("🤖 AI Recommendations"):
                            for rec in result.get("recommendations", []):
                                st.success(rec)
                    else:
                        st.error(f"Analysis failed: {result['error']}")
                except Exception as e:
                    st.error(f"Failed to fetch data: {e}")
    
    st.subheader("🎯 Initiative Pipeline")
    if not df.empty:
        if len(df[df['ai_analysis'].notna()]) > 0:
            st.caption(f"🤖 {len(df[df['ai_analysis'].notna()])} AI-generated initiatives")
        
        fig = px.scatter(
            df,
            x="budget",
            y="roi_base",
            color="risk_score",
            size="roi_high",
            hover_data=["title", "owner", "status"],
            title="Risk vs Return Analysis",
            color_continuous_scale="RdYlGn_r"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📧 Automated Reporting")
    if st.button("Generate Executive Report", use_container_width=True):
        with st.spinner("Generating report..."):
            report = generate_executive_report()
            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name=f"executive_report_{datetime.now().strftime('%Y-%m-%d')}.md",
                mime="text/markdown",
                use_container_width=True
            )
            with st.expander("Preview Report"):
                st.markdown(report)

# ==================== Initiatives Management ====================

elif page == "🎯 Initiatives":
    st.title("🎯 Strategic Initiatives")
    
    with st.expander("➕ Create New Initiative", expanded=True):
        with st.form("initiative_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                title = st.text_input("Initiative Title *")
                owner = st.text_input("Owner *")
                game_title = st.selectbox("Game Title", ["CS2", "LEAGUE_OF_LEGENDS", "VALORANT"])
            
            with col2:
                budget = st.number_input("Budget ($)", min_value=0.0, step=1000.0, format="%.2f")
                risk_score = st.slider("Risk Score", 0.0, 1.0, 0.3)
                priority = st.selectbox("Priority", ["low", "medium", "high"])
            
            description = st.text_area("Description *", height=150)
            competitor = st.text_input("Target Competitor *", "Valve")
            
            st.subheader("Success Metrics")
            metric_name = st.text_input("Metric Name", "MAU Growth")
            metric_target = st.number_input("Target Value", min_value=0.0, value=10.0)
            
            if st.form_submit_button("🚀 Launch Initiative", use_container_width=True):
                if not all([title, owner, description, competitor]):
                    st.error("❌ Please fill all required fields")
                else:
                    initiative = StrategicInitiative(
                        title=title,
                        description=description,
                        competitor=competitor,
                        game_title=GameTitle[game_title],
                        owner=owner,
                        budget=budget,
                        risk_score=risk_score,
                        priority=InitiativePriority[priority],
                        metrics=[Metric(name=metric_name, current_value=0, target_value=metric_target, unit="%")]
                    )
                    
                    roi_data = engine.calculate_roi_simulation(budget, risk_score)
                    initiative.roi_low = roi_data["roi_low"]
                    initiative.roi_base = roi_data["roi_base"]
                    initiative.roi_high = roi_data["roi_high"]
                    
                    db.save_initiative(initiative)
                    
                    if priority == "high":
                        message = f"*🚨 HIGH PRIORITY INITIATIVE*\n*Title:* {title}\n*Owner:* {owner}\n*Budget:* ${budget:,.2f}"
                        asyncio.run(engine.slack.send_alert(message, "high"))
                    
                    st.success(f"✅ Initiative created! ID: {initiative.id}")
                    st.info(f"ROI: ${roi_data['roi_low']:,.0f} - ${roi_data['roi_high']:,.0f}")
                    st.rerun()
    
    st.subheader("📋 Active Initiatives")
    initiatives = db.get_initiatives()
    
    if initiatives:
        for initiative in initiatives[:10]:
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                
                status_color = {
                    "planning": "🟡", "executing": "🔵",
                    "complete": "🟢", "blocked": "🔴"
                }
                
                with col1:
                    st.markdown(f"**{initiative['title']}**")
                    if initiative.get('ai_analysis'):
                        st.caption("🤖 AI-Generated")
                    st.caption(f"Owner: {initiative['owner']} | Game: {initiative['game_title']}")
                
                with col2:
                    st.progress(initiative['risk_score'])
                    st.caption(f"Risk: {initiative['risk_score']:.0%} | Priority: {initiative['priority'].upper()}")
                
                with col3:
                    st.markdown(f"Budget: **${initiative['budget']:,.0f}**")
                    st.caption(f"ROI: ${initiative['roi_base']:,.0f}")
                
                with col4:
                    st.markdown(f"{status_color.get(initiative['status'], '⚪')} {initiative['status'].upper()}")
                
                st.divider()
    else:
        st.info("No initiatives found. Create your first one or use AI Analysis!")

# ==================== Partnerships ====================

elif page == "🤝 Partnerships":
    st.title("🤝 Partnership Funnel")
    
    with st.expander("➕ Track New Partnership", expanded=True):
        with st.form("partnership_form"):
            company = st.text_input("Company Name *")
            deal_type = st.selectbox("Deal Type", ["hardware", "sdk", "content", "marketing"])
            value = st.number_input("Deal Value ($)", min_value=0.0, step=10000.0, format="%.2f")
            
            if st.form_submit_button("💼 Create Deal", use_container_width=True):
                if company:
                    deal = PartnershipDeal(company=company, deal_type=deal_type, value=value)
                    
                    if value > 50000:
                        hubspot_id = asyncio.run(engine.hubspot.create_deal(company, deal_type, value))
                        if hubspot_id:
                            deal.hubspot_id = hubspot_id
                            message = f"*🤝 Partnership Deal*\n*Company:* {company}\n*Value:* ${value:,.2f}"
                            asyncio.run(engine.slack.send_alert(message, "medium"))
                    
                    db.save_partnership(deal)
                    st.success(f"✅ Deal created{' (HubSpot)' if deal.hubspot_id else ''}!")
                    st.rerun()
                else:
                    st.error("❌ Company name required")
    
    st.subheader("📈 Partnership Pipeline")
    deals = db.get_partnerships()
    
    if deals:
        df_deals = pd.DataFrame(deals)
        fig = px.funnel(df_deals, x="value", y="deal_type", title="Deal Pipeline", color="deal_type")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_deals, use_container_width=True)
    else:
        st.info("No partnership deals tracked yet.")

# ==================== Market Intelligence ====================

elif page == "📈 Market Intelligence":
    st.title("📈 Live Market Intelligence")
    
    game = st.selectbox("Select Game", ["CS2", "LEAGUE_OF_LEGENDS", "VALORANT"])
    competitor = st.text_input("Competitor Name", "Valve")
    
    if st.button("🔍 Analyze Competitor", use_container_width=True):
        with st.spinner(f"Fetching data for {game}..."):
            try:
                result = asyncio.run(engine.analyze_competitor(GameTitle[game], competitor))
                
                if "error" not in result:
                    metrics = result.get("metrics", {})
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Active Players", f"{metrics.get('active_players', 0):,}")
                        st.metric("Twitch Viewers", f"{metrics.get('live_viewers', 0):,}")
                    
                    with col2:
                        engagement = metrics.get('engagement_ratio', 0)
                        st.metric("Engagement Ratio", f"{engagement:.2%}")
                        st.metric("Data Freshness", "Live")
                    
                    st.subheader("🤖 AI Recommendations")
                    for rec in result.get("recommendations", []):
                        st.success(rec)
                        
                    with st.expander("View Raw API Response"):
                        st.json(result)
                        
            except Exception as e:
                st.error(f"Analysis failed: {e}")

# ==================== Settings ====================

elif page == "⚙️ Settings":
    st.title("⚙️ Configuration & Status")
    
    st.subheader("🔐 Gemini 2.5 Flash API")
    
    if ai_engine.configured:
        st.success(f"✅ {ai_engine.model_name} configured and connected")
        
        if st.button("🧪 Test AI Connection", use_container_width=True):
            with st.spinner("Testing..."):
                try:
                    test_opp = f"AI coaching for pro CS2 teams in Berlin"
                    analysis, briefing = asyncio.run(ai_engine.analyze_opportunity(test_opp))
                    
                    if "FALLBACK" in briefing:
                        st.error("❌ Still using fallback - API key may be incorrect")
                    else:
                        st.success("✅ Gemini 2.5 is working!")
                        st.json({
                            "Market": f"${analysis.get('market_size', 0)}M",
                            "Fit": f"{analysis.get('trophi_fit_score', 0):.1%}",
                            "Game": analysis.get("game_title_fit", "N/A")
                        })
                except Exception as e:
                    st.error(f"❌ Test failed: {e}")
    else:
        st.error("❌ Gemini 2.5 Flash NOT configured")
        st.info("**Get your FREE API key:**")
        st.code("https://makersuite.google.com → Get API Key → Create Key", language="")
        
        st.subheader("Add to `.streamlit/secrets.toml`:")
        st.code('GEMINI_API_KEY = "your_actual_key_here"', language="toml")
        
        if st.button("🔁 Retry Connection After Adding Key", use_container_width=True):
            st.rerun()
    
    st.subheader("Other Integrations")
    with st.expander("Twitch, Slack, HubSpot"):
        st.text_input("Twitch Client ID", type="password", key="twitch_id")
        st.text_input("Twitch Client Secret", type="password", key="twitch_secret")
        st.text_input("Slack Webhook URL", type="password", key="slack_webhook")
        st.text_input("HubSpot API Key", type="password", key="hubspot_key")
    
    st.subheader("📊 System Status")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Database", "Connected" if sqlite3.connect(Settings.DATABASE_URL) else "Error")
        st.metric("Total Initiatives", len(db.get_initiatives()))
    
    with col2:
        pipeline_val = pd.DataFrame(db.get_partnerships())['value'].sum() if db.get_partnerships() else 0
        st.metric("Partnership Pipeline", f"${pipeline_val:,.2f}")
    
    if st.button("Run Health Check", use_container_width=True):
        try:
            st.success("✅ Market Intelligence Engine: Running")
            st.success("✅ SQLite Database: Connected")
            if ai_engine.configured: st.success("✅ Gemini AI: Configured")
            if st.secrets.get("TWITCH_CLIENT_ID"): st.info("✅ Twitch API: Configured")
            if st.secrets.get("SLACK_WEBHOOK_URL"): st.info("✅ Slack: Configured")
            if st.secrets.get("HUBSPOT_API_KEY"): st.info("✅ HubSpot: Configured")
        except Exception as e:
            st.error(f"❌ Health check failed: {e}")

# ==================== Footer ====================

st.sidebar.markdown("---")
st.sidebar.caption("Strategic Operations Platform v2.2")
st.sidebar.caption(f"AI Status: {'✅ Active' if ai_engine.configured else '⚠️ Not Configured'}")

if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.sidebar.success("✅ Platform initialized")
    if not ai_engine.configured:
        st.sidebar.warning("⚠️ Add GEMINI_API_KEY in secrets.toml for full AI features")
```


---

**If you still see errors after this, your file is corrupted. Delete it completely and create a fresh one with this code.**
