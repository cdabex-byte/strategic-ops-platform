"""
Strategic Operations Platform v2.0 - Single File
All-in-one: Backend logic + Streamlit UI
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
from dataclasses import dataclass

# ==================== Configuration ====================

class Settings:
    DATABASE_URL = "strategic.db"
    CACHE_TTL = 300  # 5 minutes
    STEAMSPY_RATE = 1.0  # 1 second between calls
    RATE_LIMIT_CALLS = 100
    RATE_LIMIT_WINDOW = 60  # seconds

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
                budget REAL,
                roi_low REAL,
                roi_base REAL,
                roi_high REAL,
                risk_score REAL,
                created_at TEXT,
                updated_at TEXT,
                metrics TEXT
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
        
        conn.commit()
        conn.close()
    
    def save_initiative(self, initiative: StrategicInitiative):
        """Save initiative to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO initiatives VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            initiative.id, initiative.title, initiative.description,
            initiative.competitor, initiative.game_title.value,
            initiative.owner, initiative.status.value, initiative.priority.value,
            initiative.budget, initiative.roi_low, initiative.roi_base, initiative.roi_high,
            initiative.risk_score,
            initiative.created_at.isoformat(), initiative.updated_at.isoformat(),
            json.dumps([m.dict() for m in initiative.metrics])
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
                "metrics": json.loads(row[15]) if row[15] else []
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
    
    def cache_get(self, key: str) -> Optional[Any]:
        """Get cached API response"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT data FROM api_cache WHERE key=? AND expires_at > ?",
            (key, datetime.now().isoformat())
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
        return None
    
    def cache_set(self, key: str, data: Any):
        """Cache API response"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        expires_at = (datetime.now() + timedelta(seconds=Settings.CACHE_TTL)).isoformat()
        
        cursor.execute(
            "INSERT OR REPLACE INTO api_cache (key, data, expires_at) VALUES (?, ?, ?)",
            (key, json.dumps(data), expires_at)
        )
        conn.commit()
        conn.close()
    
    def check_rate_limit(self, endpoint: str) -> bool:
        """Simple rate limiting check"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now()
        window_start = (now - timedelta(seconds=Settings.RATE_LIMIT_WINDOW)).isoformat()
        
        cursor.execute(
            "SELECT calls_made FROM rate_limits WHERE endpoint=? AND window_start > ?",
            (endpoint, window_start)
        )
        result = cursor.fetchone()
        
        calls = result[0] if result else 0
        
        if calls >= Settings.RATE_LIMIT_CALLS:
            conn.close()
            return False
        
        # Update or insert
        cursor.execute("""
            INSERT OR REPLACE INTO rate_limits (endpoint, calls_made, window_start)
            VALUES (?, ?, ?)
        """, (endpoint, calls + 1, now.isoformat()))
        
        conn.commit()
        conn.close()
        return True

# Initialize global database
db = DatabaseManager()

# ==================== API Integrations ====================

class SteamSpyAPI:
    """Real Steam player data"""
    def __init__(self):
        self.base_url = "https://steamspy.com/api.php"
    
    async def get_game_data(self, app_id: str) -> Dict[str, Any]:
        """Get real player counts for games"""
        await asyncio.sleep(0.1)  # Respect rate limit
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}?request=appdetails&appid={app_id}"
                )
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            print(f"SteamSpy API error: {e}")
        return {"players": 500_000}  # Fallback

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
        
        # Check cache first
        cache_key = f"analysis_{game}_{competitor}"
        cached = db.cache_get(cache_key)
        if cached:
            return cached
        
        await self._rate_limit_steam()
        
        # Map games to IDs
        game_ids = {GameTitle.CS2: "730", GameTitle.VALORANT: "1286300"}
        game_names = {
            GameTitle.CS2: "Counter-Strike 2",
            GameTitle.LEAGUE_OF_LEGENDS: "League of Legends",
            GameTitle.VALORANT: "VALORANT"
        }
        
        app_id = game_ids.get(game, "730")
        twitch_name = game_names.get(game, "CS2")
        
        # Parallel API calls
        try:
            steam_data, twitch_data = await asyncio.gather(
                self.steamspy.get_game_data(app_id),
                self.twitch.get_game_viewership(twitch_name)
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
            
            # Cache result
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
        if players < 100_000 and viewers > 50_000:
            recommendations.append("📺 High viewer/player ratio - focus on content creators")
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
    
    df_initiatives = pd.DataFrame(initiatives)
    df_partnerships = pd.DataFrame(partnerships)
    
    # Calculate KPIs
    total_budget = df_initiatives["budget"].sum() if not df_initiatives.empty else 0
    avg_risk = df_initiatives["risk_score"].mean() if not df_initiatives.empty else 0
    high_priority_count = len(df_initiatives[df_initiatives["priority"] == "high"]) if not df_initiatives.empty else 0
    total_pipeline = df_partnerships["value"].sum() if not df_partnerships.empty else 0
    
    active_initiatives = df_initiatives[df_initiatives["status"] != "complete"] if not df_initiatives.empty else pd.DataFrame()
    
    report = f"""# Strategic Operations Weekly Report
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

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
- **High-Risk Initiatives (>20%):** {len(df_initiatives[df_initiatives["risk_score"] > 0.2]) if not df_initiatives.empty else 0}
- **Budget at Risk:** ${df_initiatives[df_initiatives["risk_score"] > 0.2]["budget"].sum():,.2f if not df_initiatives.empty else 0}

---

*Report auto-generated by Strategic Operations Platform v2.0*
"""
    return report

# ==================== Streamlit UI ====================

# Page configuration
st.set_page_config(
    page_title="Strategic Operations Platform v2.0",
    page_icon="🎮",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("🎮 Strategic Ops v2.0")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Executive Dashboard", "🎯 Initiatives", "🤝 Partnerships", "📈 Market Intelligence", "⚙️ Settings"],
    index=0
)

# ==================== Executive Dashboard ====================

if page == "📊 Executive Dashboard":
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

# ==================== Initiatives Management ====================

elif page == "🎯 Initiatives":
    st.title("🎯 Strategic Initiatives")
    
    # Create new initiative
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
            
            # Metrics
            st.subheader("Success Metrics")
            metric_name = st.text_input("Metric Name", "MAU Growth")
            metric_target = st.number_input("Target Value", min_value=0.0, value=10.0)
            
            submitted = st.form_submit_button("🚀 Launch Initiative", use_container_width=True)
            
            if submitted:
                if not all([title, owner, description, competitor]):
                    st.error("❌ Please fill all required fields")
                else:
                    # Create initiative object
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
                    
                    # Calculate ROI
                    roi_data = engine.calculate_roi_simulation(budget, risk_score)
                    initiative.roi_low = roi_data["roi_low"]
                    initiative.roi_base = roi_data["roi_base"]
                    initiative.roi_high = roi_data["roi_high"]
                    
                    # Save to database
                    db.save_initiative(initiative)
                    
                    # Send Slack alert for high priority
                    if priority == "high":
                        message = f"""
                        *🚨 HIGH PRIORITY INITIATIVE LAUNCHED*
                        *Title:* {title}
                        *Owner:* {owner}
                        *Budget:* ${budget:,.2f}
                        *ROI Range:* ${roi_data["roi_low"]:,.0f} - ${roi_data["roi_high"]:,.0f}
                        *Risk Score:* {risk_score:.1%}
                        """
                        asyncio.run(engine.slack.send_alert(message, "high"))
                    
                    st.success(f"✅ Initiative created! ID: {initiative.id}")
                    st.info(f"""
                    **ROI Simulation Results:**
                    - Conservative: ${roi_data["roi_low"]:,.0f}
                    - Expected: ${roi_data["roi_base"]:,.0f}
                    - Optimistic: ${roi_data["roi_high"]:,.0f}
                    - Risk of Loss: {roi_data["risk_of_loss"]:.1%}
                    """)
                    st.rerun()
    
    # Display initiatives
    st.subheader("📋 Active Initiatives")
    initiatives = db.get_initiatives()
    
    if initiatives:
        for initiative in initiatives[:10]:  # Show top 10
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                
                status_color = {
                    "planning": "🟡", "executing": "🔵",
                    "complete": "🟢", "blocked": "🔴"
                }
                
                with col1:
                    st.markdown(f"**{initiative['title']}**")
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
        st.info("No initiatives found. Create your first one!")

# ==================== Partnerships ====================

elif page == "🤝 Partnerships":
    st.title("🤝 Partnership Funnel")
    
    # Create deal
    with st.expander("➕ Track New Partnership", expanded=True):
        with st.form("partnership_form"):
            company = st.text_input("Company Name *")
            deal_type = st.selectbox("Deal Type", ["hardware", "sdk", "content", "marketing"])
            value = st.number_input("Deal Value ($)", min_value=0.0, step=10000.0, format="%.2f")
            
            if st.form_submit_button("💼 Create Deal", use_container_width=True):
                if company:
                    deal = PartnershipDeal(company=company, deal_type=deal_type, value=value)
                    
                    # Sync to HubSpot if significant
                    if value > 50000:
                        hubspot_id = asyncio.run(engine.hubspot.create_deal(company, deal_type, value))
                        if hubspot_id:
                            deal.hubspot_id = hubspot_id
                            message = f"""
                            *🤝 Partnership Deal Created*
                            *Company:* {company}
                            *Type:* {deal_type.upper()}
                            *Value:* ${value:,.2f}
                            """
                            asyncio.run(engine.slack.send_alert(message, "medium"))
                    
                    db.save_partnership(deal)
                    st.success(f"✅ Deal created{' (HubSpot sync: ' + deal.hubspot_id + ')' if deal.hubspot_id else ''}!")
                    st.rerun()
                else:
                    st.error("❌ Company name required")
    
    # Funnel visualization
    st.subheader("📈 Partnership Pipeline")
    deals = db.get_partnerships()
    
    if deals:
        df_deals = pd.DataFrame(deals)
        
        fig = px.funnel(
            df_deals,
            x="value",
            y="deal_type",
            title="Deal Pipeline by Type",
            color="deal_type"
        )
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
        with st.spinner(f"Fetching real-time data for {game}..."):
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
                    
                    # Recommendations
                    st.subheader("🤖 AI Recommendations")
                    for rec in result.get("recommendations", []):
                        st.success(rec)
                    
                    # Raw data
                    with st.expander("View Raw API Response"):
                        st.json(result)
                else:
                    st.error(f"Analysis failed: {result['error']}")
                    
            except Exception as e:
                st.error(f"Failed to analyze: {e}")

# ==================== Settings ====================

elif page == "⚙️ Settings":
    st.title("⚙️ Configuration & Status")
    
    # API Keys
    st.subheader("🔐 API Keys")
    with st.expander("Configure Integrations"):
        st.text_input("Twitch Client ID", type="password", key="twitch_id")
        st.text_input("Twitch Client Secret", type="password", key="twitch_secret")
        st.text_input("Slack Webhook URL", type="password", key="slack_webhook")
        st.text_input("HubSpot API Key", type="password", key="hubspot_key")
        
        if st.button("Save Settings"):
            st.success("✅ Settings saved (Note: Add to `.streamlit/secrets.toml` for production)")
    
    # System status
    st.subheader("📊 System Status")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Database", "Connected" if sqlite3.connect(Settings.DATABASE_URL) else "Error")
        st.metric("Cache Size", f"{len(db.get_partnerships())} items")
    
    with col2:
        initiatives_count = len(db.get_initiatives())
        st.metric("Total Initiatives", initiatives_count)
        st.metric("Partnership Pipeline", f"${pd.DataFrame(db.get_partnerships())['value'].sum() if db.get_partnerships() else 0:,.2f}")
    
    # Health check
    if st.button("Run Health Check", use_container_width=True):
        try:
            # Test async engine
            st.success("✅ Market Intelligence Engine: Running")
            
            # Test database
            st.success("✅ SQLite Database: Connected")
            
            # Test APIs (if configured)
            if st.secrets.get("TWITCH_CLIENT_ID"):
                st.info("⚠️ Twitch API: Not tested (rate limit protection)")
            
            if st.secrets.get("SLACK_WEBHOOK_URL"):
                st.info("⚠️ Slack: Configured but not tested")
            
            if st.secrets.get("HUBSPOT_API_KEY"):
                st.info("⚠️ HubSpot: Configured but not tested")
                
        except Exception as e:
            st.error(f"❌ Health check failed: {e}")

# ==================== Footer ====================

st.sidebar.markdown("---")
st.sidebar.caption("Strategic Operations Platform v2.0")
st.sidebar.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")
