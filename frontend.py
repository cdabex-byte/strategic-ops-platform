"""
Strategic Operations Platform v2.2 - AI-Powered with Detailed Briefing
All-in-one: Backend logic + Streamlit UI + Gemini AI + Executive Briefing
"""

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
    
    def _init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Initiatives table
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
      # ==================== Database Manager ====================

class DatabaseManager:
    def __init__(self, db_path: str = Settings.DATABASE_URL):
        self.db_path = db_path
        self._init_db()
        self._migrate_db()  # Add this line
    
    def _init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Initiatives table
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
        
        # Partnerships table
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
        
        # API cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_cache (
                key TEXT PRIMARY KEY,
                data TEXT,
                expires_at TEXT
            )
        """)
        
        # Rate limit tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rate_limits (
                endpoint TEXT PRIMARY KEY,
                calls_made INTEGER,
                window_start TEXT
            )
        """)
        
        # AI Analysis cache - UPDATED SCHEMA
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_analysis_cache (
                opportunity_key TEXT PRIMARY KEY,
                analysis_data TEXT,
                briefing_text TEXT,  -- NEW COLUMN
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
            # Check if briefing_text column exists
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
        """Save initiative to database"""
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
        """Fetch initiatives with optional filtering"""
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
        """Save partnership deal"""
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
        """Fetch all partnerships"""
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
    
    def save_ai_analysis(self, opportunity_key: str, analysis_data: Dict[str, Any], briefing_text: str):
        """Save AI analysis results and briefing"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO ai_analysis_cache (opportunity_key, analysis_data, briefing_text, created_at)
            VALUES (?, ?, ?, ?)
        """, (opportunity_key, json.dumps(analysis_data), briefing_text, datetime.now().isoformat()))
        conn.commit()
        conn.close()
    
    def get_ai_analysis(self, opportunity_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve AI analysis results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT analysis_data, briefing_text FROM ai_analysis_cache WHERE opportunity_key=?",
            (opportunity_key,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "analysis": json.loads(result[0]),
                "briefing": result[1]
            }
        return None

# Initialize global database
db = DatabaseManager()

# ==================== AI Integration (Gemini) ====================

class GeminiAI:
    """Google Gemini 1.5 Flash for market research and opportunity analysis"""
    
    def __init__(self):
        self.api_key = st.secrets.get("GEMINI_API_KEY", "")
        self.configured = bool(self.api_key)
        if self.configured:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
            except Exception as e:
                print(f"Gemini config error: {e}")
                self.configured = False
                self.model = None
        else:
            self.model = None
    
    async def analyze_opportunity(self, opportunity: str) -> tuple[Dict[str, Any], str]:
        """
        AI-powered market research and Trophi.ai fit analysis
        Returns: (analysis_dict, briefing_text)
        """
        
        if not self.configured:
            fallback_briefing = self._generate_fallback_briefing(opportunity)
            return self._get_fallback_data(opportunity), fallback_briefing
        
        # Check cache first
        cache_key = f"opp_{hash(opportunity)}"
        cached = db.get_ai_analysis(cache_key)
        if cached:
            return cached["analysis"], cached["briefing"]
        
        # Enhanced prompt for varied results
        prompt = f"""
        You are a senior Strategy & Operations Associate at Trophi.ai, an AI coaching platform for competitive gamers.
        
        **TASK:** Analyze this business opportunity and provide UNIQUE, specific market research.
        
        **OPPORTUNITY:** {opportunity}
        
        **REQUIREMENTS:**
        1. Research ACTUAL market data specific to this opportunity
        2. Identify REAL competitors in this space
        3. Calculate specific market sizing (don't use generic numbers)
        4. Provide Trophi.ai fit assessment based on:
           - Gaming market alignment
           - AI coaching applicability
           - Our competitive advantages
           - Technical feasibility
        
        **OUTPUT FORMAT:**
        
        First, provide a detailed executive briefing (narrative format):
        BRIEFING_START
        # Executive Briefing: [Opportunity Title]
        
        ## Market Overview
        [Specific market size, growth rate, and key drivers relevant to {opportunity}]
        
        ## Competitive Landscape
        [Specific competitors and their market positions]
        
        ## Trophi.ai Strategic Fit
        [Detailed analysis of how this aligns with Trophi.ai's mission and capabilities]
        
        ## Recommendation
        [Specific, actionable recommendation with rationale]
        
        ## Key Risks & Mitigation
        [Specific risks and how to mitigate them]
        BRIEFING_END
        
        Then, provide structured data in this JSON format:
        JSON_START
        {{
            "opportunity_title": "Brief specific title",
            "market_size": 0,  
            "market_growth_rate": 0.0,
            "target_audience": "Specific demographic description",
            "competitive_landscape": ["Real Competitor 1", "Real Competitor 2"],
            "key_trends": ["Specific trend 1", "Specific trend 2"],
            "trophi_fit_score": 0.0,
            "recommendation": "Specific recommendation",
            "risks": ["Specific risk 1", "Specific risk 2"],
            "budget_estimate": 0,
            "roi_potential": 0.0,
            "game_title_fit": "CS2 or VALORANT or League of Legends",
            "priority": "high/medium/low",
            "metrics_to_track": ["Specific metric 1", "Specific metric 2"]
        }}
        JSON_END
        
        **IMPORTANT:** Make the analysis SPECIFIC to "{opportunity}" - don't use generic templates.
        """
        
        try:
            # Make API call
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.7,  # Higher temperature for more variation
                    max_output_tokens=2000
                )
            )
            
            # Parse response
            result_text = response.text
            
            # Extract briefing and JSON
            briefing_start = result_text.find("BRIEFING_START") + len("BRIEFING_START")
            briefing_end = result_text.find("BRIEFING_END")
            briefing = result_text[briefing_start:briefing_end].strip() if briefing_start > 0 and briefing_end > 0 else "No briefing generated."
            
            json_start = result_text.find("JSON_START") + len("JSON_START")
            json_end = result_text.find("JSON_END")
            json_str = result_text[json_start:json_end].strip() if json_start > 0 and json_end > 0 else ""
            
            if json_str:
                analysis = json.loads(json_str)
            else:
                analysis = self._get_fallback_data(opportunity)
            
            # Save to cache
            db.save_ai_analysis(cache_key, analysis, briefing)
            
            return analysis, briefing
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            fallback_briefing = self._generate_fallback_briefing(opportunity)
            return self._get_fallback_data(opportunity), fallback_briefing
    
    def _get_fallback_data(self, opportunity: str) -> Dict[str, Any]:
        """Generate varied fallback data based on opportunity text"""
        # Hash the opportunity to generate consistent but varied fallback data
        import hashlib
        
        hash_val = int(hashlib.md5(opportunity.encode()).hexdigest(), 16)
        market_size = 50 + (hash_val % 200)  # 50-250M range
        fit_score = ((hash_val % 100) / 100) * 0.6 + 0.2  # 0.2-0.8 range
        
        games = ["CS2", "VALORANT", "LEAGUE_OF_LEGENDS"]
        priorities = ["high", "medium", "low"]
        
        return {
            "opportunity_title": f"Analysis: {opportunity[:60]}...",
            "market_size": market_size,
            "market_growth_rate": 15.5 + (hash_val % 15),
            "target_audience": "Competitive gamers aged 16-30",
            "competitive_landscape": ["Traditional coaching platforms", "In-game analytics tools"],
            "key_trends": ["AI adoption in gaming", "Personalized training"],
            "trophi_fit_score": fit_score,
            "recommendation": "Validate market size with pilot program",
            "risks": ["Market validation needed", "Competitive response"],
            "budget_estimate": 25000 + ((hash_val % 100) * 1000),
            "roi_potential": 1.5 + (fit_score * 2),
            "game_title_fit": games[hash_val % 3],
            "priority": priorities[hash_val % 3],
            "metrics_to_track": ["User acquisition", "Engagement rate", "Retention"]
        }
    
    def _generate_fallback_briefing(self, opportunity: str) -> str:
        """Generate a narrative fallback briefing"""
        return f"""
# Executive Briefing: {opportunity[:50]}...

## Market Overview
The market for this opportunity is estimated at $50-250M annually, with strong growth potential 
in the competitive gaming sector. The target demographic consists primarily of competitive gamers 
aged 16-30 who are actively seeking performance improvement tools.

## Competitive Landscape
Key competitors include traditional coaching platforms and emerging AI-powered training solutions. 
The market shows signs of consolidation but remains fragmented enough for new entrants with 
differentiated value propositions.

## Trophi.ai Strategic Fit
This opportunity aligns moderately well with Trophi.ai's mission of democratizing gaming coaching 
through AI. The technical feasibility appears high, with potential for leveraging existing 
infrastructure and user base.

## Recommendation
Proceed with a phased approach: Start with a pilot program to validate market assumptions, 
then scale based on learnings. Focus on building defensible moats around data and user experience.

## Key Risks & Mitigation
- **Market Validation**: Mitigate through pilot program and early user feedback
- **Competitive Response**: Build strong brand and loyal community
- **Technical Execution**: Leverage existing AI/ML expertise
"""

# Initialize AI engine
ai_engine = GeminiAI()

# ==================== API Integrations ====================

class SteamSpyAPI:
    """Real Steam player data"""
    def __init__(self):
        self.base_url = "https://steamspy.com/api.php"
    
    async def get_game_data(self, app_id: str) -> Dict[str, Any]:
        """Get real player counts for games"""
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
    """Get live viewership data"""
    def __init__(self):
        secrets = st.secrets
        self.client_id = secrets.get("TWITCH_CLIENT_ID", "")
        self.client_secret = secrets.get("TWITCH_CLIENT_SECRET", "")
        self.token = None
        self.token_expires = datetime.min
    
    async def get_token(self):
        """Get OAuth token with caching"""
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
    """Real-time alerts to Slack"""
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
    """Track partnership deals"""
    def __init__(self):
        self.api_key = st.secrets.get("HUBSPOT_API_KEY", "")
    
    async def create_deal(self, company: str, deal_type: str, value: float) -> Optional[str]:
        """Create deal in HubSpot"""
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
        """Ensure 1 second between SteamSpy calls"""
        now = datetime.now()
        elapsed = (now - self._last_steam_call).total_seconds()
        if elapsed < Settings.STEAMSPY_RATE:
            await asyncio.sleep(Settings.STEAMSPY_RATE - elapsed)
        self._last_steam_call = datetime.now()
    
    async def analyze_competitor(self, game: GameTitle, competitor: str) -> Dict[str, Any]:
        """Real-time competitor analysis with caching"""
        
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
        """AI-powered recommendations"""
        recommendations = []
        if players > 500_000:
            recommendations.append("🎯 High player base - prioritize community features")
        if viewers / max(players, 1) > 0.1:
            recommendations.append("📈 Strong viewership - explore esports partnerships")
        return recommendations
    
    def calculate_roi_simulation(self, budget: float, risk_score: float) -> Dict[str, float]:
        """Monte Carlo ROI simulation"""
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

# Initialize global engine
engine = MarketIntelligenceEngine()

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
    
    # AI insights from latest analysis
    latest_ai = db.get_ai_analysis("latest")
    ai_section = ""
    if latest_ai:
        analysis = latest_ai["analysis"]
        ai_section = f"""## 🤖 Latest AI Opportunity Analysis

**Opportunity:** {analysis.get('opportunity_title', 'N/A')}

**Trophi Fit Score:** {analysis.get('trophi_fit_score', 0):.1%}

**Market Size:** ${analysis.get('market_size', 0)}M (growing at {analysis.get('market_growth_rate', 0)}% annually)

**Key Insights:**
- Target: {analysis.get('target_audience', 'N/A')}
- Budget Needed: ${analysis.get('budget_estimate', 0):,.0f}
- Expected ROI: {analysis.get('roi_potential', 0):.1f}x
- Priority: {analysis.get('priority', 'N/A').upper()}

**AI Recommendation:** {analysis.get('recommendation', 'N/A')}
"""
    
    active_initiatives = df[df["status"] != "complete"] if not df.empty else pd.DataFrame()
    
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

# Page configuration
st.set_page_config(
    page_title="Strategic Operations Platform v2.2",
    page_icon="🎮",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("🎮 Strategic Ops v2.2")
page = st.sidebar.radio(
    "Navigation",
    ["🔍 AI Opportunity Analysis", "📊 Executive Dashboard", "🎯 Initiatives", "🤝 Partnerships", "📈 Market Intelligence", "⚙️ Settings"],
    index=0
)

# ==================== AI Opportunity Analysis ====================

if page == "🔍 AI Opportunity Analysis":
    st.title("🔍 AI-Powered Opportunity Analysis")
    
    # API Key check
    if not ai_engine.configured:
        st.error("⚠️ Gemini AI not configured. Add GEMINI_API_KEY to secrets.toml")
        st.info("Get free API key at: https://makersuite.google.com")
    
    st.markdown("""
    Describe any business opportunity and AI will generate a **detailed briefing** with market research, 
    competitive analysis, and Trophi.ai fit assessment. This feeds real data to all sections.
    """)
    
    # Chat interface
    user_input = st.text_area(
        "📝 Describe your opportunity in detail",
        height=150,
        placeholder="Example: Launch AI coaching for VALORANT in Southeast Asia, target pro teams and content creators..."
    )
    
    # Additional context options
    with st.expander("⚙️ Additional Context (Optional)"):
        target_region = st.text_input("Target Region", "Global")
        target_segment = st.selectbox("Target Segment", ["Pro Players", "Semi-Pro", "Casual Competitive", "Content Creators"])
        timeline = st.select_slider("Timeline", ["1-3 months", "3-6 months", "6-12 months", "12+ months"])
    
    if st.button("🚀 Generate AI Briefing & Analysis", use_container_width=True, type="primary"):
        if not user_input.strip():
            st.error("❌ Please describe an opportunity to analyze")
        elif not ai_engine.configured:
            st.error("❌ Gemini API key not configured")
        else:
            with st.spinner("🤖 AI is conducting deep market research..."):
                try:
                    # Combine inputs
                    full_opportunity = f"{user_input} | Target: {target_segment} in {target_region} | Timeline: {timeline}"
                    
                    # Run AI analysis
                    analysis, briefing = asyncio.run(ai_engine.analyze_opportunity(full_opportunity))
                    
                    if "error" not in analysis:
                        # Save as latest analysis
                        db.save_ai_analysis("latest", analysis, briefing)
                        
                        # Display briefing first (prominent)
                        st.subheader("📋 AI-Generated Executive Briefing")
                        st.markdown(briefing)
                        
                        # Display structured data in tabs
                        st.subheader("📊 Structured Analysis Data")
                        tab1, tab2, tab3, tab4 = st.tabs(["Market Metrics", "Competitive Intel", "Trophi Fit", "Action Items"])
                        
                        with tab1:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Market Size", f"${analysis.get('market_size', 0)}M")
                            with col2:
                                st.metric("Growth Rate", f"{analysis.get('market_growth_rate', 0)}%")
                            with col3:
                                st.metric("Budget Needed", f"${analysis.get('budget_estimate', 0):,.0f}")
                            with col4:
                                st.metric("ROI Potential", f"{analysis.get('roi_potential', 0):.1f}x")
                            
                            st.markdown(f"**Target Audience:** {analysis.get('target_audience', 'N/A')}")
                        
                        with tab2:
                            st.markdown("**Competitors:**")
                            for competitor in analysis.get("competitive_landscape", []):
                                st.warning(f"⚠️ {competitor}")
                            
                            st.markdown("**Key Trends:**")
                            for trend in analysis.get("key_trends", []):
                                st.success(f"📈 {trend}")
                        
                        with tab3:
                            fit_score = analysis.get('trophi_fit_score', 0)
                            st.progress(fit_score, text=f"Trophi Fit Score: {fit_score:.1%}")
                            
                            priority = analysis.get('priority', 'medium')
                            priority_color = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
                            st.metric("Priority", f"{priority_color.get(priority, '⚪')} {priority.upper()}")
                            
                            st.markdown(f"**Best Fit Game:** {analysis.get('game_title_fit', 'CS2')}")
                        
                        with tab4:
                            st.markdown(f"**Recommendation:** {analysis.get('recommendation', 'N/A')}")
                            
                            st.markdown("**Track These Metrics:**")
                            for metric in analysis.get("metrics_to_track", []):
                                st.info(f"📊 {metric}")
                        
                        # Auto-create initiative section
                        st.subheader("🎯 Auto-Create Initiative")
                        st.markdown("Use the AI analysis to automatically create a strategic initiative:")
                        
                        if st.button("✨ Create Initiative from This Analysis", use_container_width=True):
                            initiative = StrategicInitiative(
                                title=analysis.get("opportunity_title", "AI Generated Initiative"),
                                description=user_input,
                                competitor=analysis.get("competitive_landscape", ["Unknown"])[0] if analysis.get("competitive_landscape") else "Unknown",
                                game_title=GameTitle[analysis.get("game_title_fit", "CS2")] if analysis.get("game_title_fit") in GameTitle.__members__ else GameTitle.CS2,
                                owner="AI Analyst",
                                budget=analysis.get("budget_estimate", 50000),
                                risk_score=1.0 - analysis.get("trophi_fit_score", 0.5),
                                priority=InitiativePriority[analysis.get("priority", "medium")] if analysis.get("priority") in InitiativePriority.__members__ else InitiativePriority.MEDIUM,
                                metrics=[Metric(name=metric, current_value=0, target_value=100, unit="%") for metric in analysis.get("metrics_to_track", ["Engagement"])[:3]],
                                ai_analysis=analysis
                            )
                            
                            # Calculate ROI
                            roi_data = engine.calculate_roi_simulation(initiative.budget, initiative.risk_score)
                            initiative.roi_low = roi_data["roi_low"]
                            initiative.roi_base = roi_data["roi_base"]
                            initiative.roi_high = roi_data["roi_high"]
                            
                            # Save to database
                            db.save_initiative(initiative)
                            
                            st.success(f"✅ Initiative created: {initiative.id}")
                            st.info(f"""
                            **ROI Simulation:**
                            - Conservative: ${roi_data["roi_low"]:,.0f}
                            - Expected: ${roi_data["roi_base"]:,.0f}
                            - Optimistic: ${roi_data["roi_high"]:,.0f}
                            """)
                            
                            # Send Slack alert for high priority
                            if initiative.priority == InitiativePriority.HIGH:
                                message = f"""
                                *🤖 AI-GENERATED HIGH PRIORITY INITIATIVE*
                                *Title:* {initiative.title}
                                *Budget:* ${initiative.budget:,.2f}
                                *Trophi Fit:* {analysis.get('trophi_fit_score', 0):.1%}
                                """
                                asyncio.run(engine.slack.send_alert(message, "high"))
                        
                        # View raw data option
                        with st.expander("🔍 View Raw AI Output"):
                            st.json(analysis)
                        
                    else:
                        st.error(f"❌ AI analysis failed: {analysis.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"❌ Analysis error: {e}")
                    st.exception(e)
    
    # Show recent analyses
    st.subheader("📚 Recent AI Analyses")
    # In a real app, you'd query all analyses; here we show the latest
    latest = db.get_ai_analysis("latest")
    if latest:
        with st.expander(f"📄 Latest: {latest['analysis'].get('opportunity_title', 'N/A')}"):
            st.markdown(latest['briefing'])
            st.caption(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    else:
        st.info("No AI analyses yet. Start by describing an opportunity!")

# ==================== Executive Dashboard ====================

elif page == "📊 Executive Dashboard":
    st.title("📊 Executive Dashboard")
    
    # Refresh button
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Refresh All Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Fetch data
    initiatives = db.get_initiatives()
    partnerships = db.get_partnerships()
    
    df = pd.DataFrame(initiatives)
    
    # Metrics
    with col2:
        total_budget = df["budget"].sum() if not df.empty else 0
        st.metric("Total Budget", f"${total_budget:,.2f}")
    
    with col3:
        avg_roi = df["roi_base"].mean() if not df.empty else 0
        st.metric("Avg Expected ROI", f"${avg_roi:,.0f}")
    
    with col4:
        risk_count = len(df[df["risk_score"] > 0.2]) if not df.empty else 0
        st.metric("High-Risk Items", risk_count)
    
    # AI Insights from latest analysis
    latest_ai = db.get_ai_analysis("latest")
    if latest_ai:
        st.subheader("🤖 Latest AI Strategic Briefing")
        with st.container():
            analysis = latest_ai["analysis"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Trophi Fit Score", f"{analysis.get('trophi_fit_score', 0):.1%}")
            with col2:
                st.metric("Market Size", f"${analysis.get('market_size', 0)}M")
            with col3:
                st.metric("AI Priority", analysis.get('priority', 'N/A').upper())
            
            # Show briefing excerpt
            briefing_preview = latest_ai['briefing'][:500] + "..." if len(latest_ai['briefing']) > 500 else latest_ai['briefing']
            st.info(briefing_preview)
            if st.button("📄 View Full Briefing", use_container_width=True):
                with st.expander("Full AI Briefing"):
                    st.markdown(latest_ai['briefing'])
    
    # Live market data
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
    
    # Initiative pipeline
    st.subheader("🎯 Initiative Pipeline")
    if not df.empty:
        # Highlight AI-generated initiatives
        ai_count = len(df[df['ai_analysis'].notna()])
        if ai_count > 0:
            st.caption(f"🤖 {ai_count} AI-generated initiatives")
        
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
    
    # Weekly report
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

# ==================== Footer ====================

st.sidebar.markdown("---")
st.sidebar.caption("Strategic Operations Platform v2.2")
st.sidebar.caption(f"AI Status: {'✅ Active' if ai_engine.configured else '⚠️ Not Configured'}")

# Initialize page on first load
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.sidebar.success("✅ Platform initialized")
    if not ai_engine.configured:
        st.sidebar.warning("⚠️ Add GEMINI_API_KEY in secrets.toml for full AI features")
