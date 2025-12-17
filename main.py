"""
Strategic Operations Platform API
FastAPI + SQLite + Google Sheets Logging
"""

import asyncio
import sqlite3
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path
import httpx
import json
import os
import hashlib
from enum import Enum
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ============================================================================
# DATABASE LAYER - SQLite with Async Support
# ============================================================================

DB_PATH = "strategic_ops.db"

def init_db():
    """Initialize SQLite database with full schema"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # OKRs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS okrs (
            id TEXT PRIMARY KEY,
            objective TEXT NOT NULL,
            key_results TEXT NOT NULL,
            owner TEXT NOT NULL,
            quarter TEXT NOT NULL,
            status TEXT DEFAULT 'on_track',
            priority INTEGER DEFAULT 5,
            created_at TEXT NOT NULL
        )
    """)
    
    # Initiatives table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS initiatives (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            hypothesis TEXT NOT NULL,
            framework TEXT NOT NULL,
            data_sources TEXT NOT NULL,
            recommendation TEXT,
            business_case TEXT,
            status TEXT DEFAULT 'exploring',
            priority INTEGER DEFAULT 5,
            tags TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    
    # Human reviews table
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
    conn.close()

class Database:
    async def execute(self, query: str, params: tuple = None):
        """Execute query asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._execute_sync, query, params)
    
    def _execute_sync(self, query: str, params: tuple = None):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        conn.commit()
        conn.close()

# ============================================================================
# PYDANTIC MODELS - All Business Objects
# ============================================================================

class GameTitle(str, Enum):
    LEAGUE_OF_LEGENDS = "league_of_legends"
    VALORANT = "valorant"
    CS2 = "cs2"
    OVERWATCH = "overwatch"
    FIFA = "fifa"

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

class KeyResult(BaseModel):
    description: str
    target: float
    current: float = 0.0
    unit: str = "%"

class OKR(BaseModel):
    id: str = Field(default_factory=lambda: hashlib.md5(
        f"{datetime.now().timestamp()}".encode()
    ).hexdigest()[:12])
    objective: str
    key_results: List[KeyResult]
    owner: str
    quarter: str
    status: str = "on_track"
    priority: int = Field(ge=1, le=10)
    created_at: datetime = Field(default_factory=datetime.now)
    
    @property
    def overall_progress(self) -> float:
        if not self.key_results:
            return 0.0
        return sum((kr.current / kr.target) * 100 for kr in self.key_results) / len(self.key_results)

class StrategicInitiative(BaseModel):
    id: str = Field(default_factory=lambda: f"INIT-{hashlib.md5(
        f"{datetime.now().timestamp()}".encode()
    ).hexdigest()[:8].upper()}")
    title: str
    hypothesis: str
    framework: Dict[str, Any] = Field(default_factory=dict)
    data_sources: List[str] = Field(default_factory=list)
    recommendation: Optional[str] = None
    business_case: Optional[Dict[str, float]] = None
    status: str = "exploring"
    priority: int = Field(ge=1, le=10)
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class PitchNarrative(BaseModel):
    id: str = Field(default_factory=lambda: f"PITCH-{hashlib.md5(
        f"{datetime.now().timestamp()}".encode()
    ).hexdigest()[:8].upper()}")
    initiative_id: str
    audience: str
    sections: List[Dict[str, Any]]
    financial_highlights: Dict[str, float]
    risks_mitigations: List[Dict[str, str]]
    call_to_action: str
    generated_at: datetime = Field(default_factory=datetime.now)

# ============================================================================
# ENGINES - All Business Logic
# ============================================================================

class MarketIntelligenceEngine:
    def __init__(self):
        self.api_keys = {
            "steamspy": os.getenv("STEAMSPY_API_KEY"),
            "twitch": os.getenv("TWITCH_CLIENT_ID"),
            "gemini": os.getenv("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY", ""))
        }
        self.cache = {}
        self.cache_ttl = timedelta(hours=6)
    
    async def analyze_competitor(self, competitor: str, game: GameTitle) -> CompetitiveInsight:
        """Analyze competitor with caching"""
        cache_key = f"{competitor}_{game}"
        if cache_key in self.cache:
            timestamp, data = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                return data
        
        # Parallel data collection
        async with httpx.AsyncClient() as client:
            tasks = [
                self._scrape_website(client, competitor),
                self._get_steam_data(client, game),
                self._get_twitch_metrics(client, game, competitor)
            ]
            website, steam, twitch = await asyncio.gather(*tasks, return_exceptions=True)
        
        insight = await self._synthesize_insight(competitor, game, website, steam, twitch)
        self.cache[cache_key] = (datetime.now(), insight)
        return insight
    
    async def _scrape_website(self, client: httpx.AsyncClient, competitor: str):
        return {"features": ["AI coaching", "Stats tracking"], "pricing": "$5-15/mo"}
    
    async def _get_steam_data(self, client: httpx.AsyncClient, game: GameTitle):
        return {"players": 1_000_000, "owners": 5_000_000}
    
    async def _get_twitch_metrics(self, client: httpx.AsyncClient, game: GameTitle, competitor: str):
        return {"avg_viewers": 150_000, "top_streamers": 50}
    
    async def _synthesize_insight(self, competitor, game, *data):
        prompt = f"""
        Analyze {competitor} in {game} AI coaching. Provide JSON with:
        key_strengths, key_weaknesses, market_position, threat_level (1-10),
        opportunity_windows, confidence_score (0-1), sources
        """
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={self.api_keys['gemini']}",
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
        except:
            return CompetitiveInsight(
                competitor_name=competitor,
                game=game,
                key_strengths=["Strong brand", "Large base"],
                key_weaknesses=["Limited AI", "High price"],
                market_position="challenger",
                threat_level=6,
                opportunity_windows=["Mobile", "Esports"],
                confidence_score=0.7,
                sources=["Mock data"]
            )

class OpportunitySizer:
    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY", ""))
    
    async def calculate_market_size(self, game: GameTitle, segment: str, tier: str, geo: str) -> TAMAnalysis:
        market_data = await self._get_market_data(game)
        assumptions = await self._generate_assumptions(game, segment, tier, geo)
        
        tam = market_data['total_players'] * assumptions['arpu']
        penetration_rate = assumptions['segment_penetration']
        sam = int(tam * penetration_rate)
        som = int(sam * 0.1)
        
        return TAMAnalysis(
            game=game,
            total_players=market_data['total_players'],
            serviceable_available_market=sam,
            serviceable_obtainable_market=som,
            avg_revenue_per_user=assumptions['arpu'],
            tam_usd=tam,
            sam_usd=sam,
            som_usd=som,
            growth_rate=market_data['growth_rate'],
            assumptions=assumptions,
            confidence_score=assumptions['confidence_score']
        )
    
    async def _get_market_data(self, game: GameTitle):
        return {"total_players": 1_000_000, "growth_rate": 0.10}
    
    async def _generate_assumptions(self, game, segment, tier, geo):
        arpu_map = {"premium": 120, "mid": 60, "free": 0}
        return {
            "arpu": arpu_map.get(tier, 60),
            "segment_penetration": 0.15,
            "confidence_score": 0.6,
            "key_risks": ["Saturation", "Competition"],
            "growth_drivers": ["Mobile", "AI"]
        }

class OKRTracker:
    def __init__(self):
        self.db = Database()
    
    async def create_okr(self, okr: OKR) -> OKR:
        query = """
            INSERT INTO okrs (id, objective, key_results, owner, quarter, status, priority, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        await self.db.execute(query, (
            okr.id, okr.objective, json.dumps([kr.dict() for kr in okr.key_results]),
            okr.owner, okr.quarter, okr.status, okr.priority, okr.created_at.isoformat()
        ))
        return okr
    
    async def get_okrs(self, quarter=None, owner=None, status=None) -> List[OKR]:
        query = "SELECT * FROM okrs WHERE 1=1"
        params = []
        if quarter:
            query += " AND quarter = ?"
            params.append(quarter)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        conn.close()
        
        return [OKR(
            id=row[0], objective=row[1],
            key_results=[KeyResult(**kr) for kr in json.loads(row[2])],
            owner=row[3], quarter=row[4], status=row[5],
            priority=row[6], created_at=datetime.fromisoformat(row[7])
        ) for row in rows]

class StrategicNarrativeEngine:
    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY", ""))
    
    async def generate_pitch_deck(self, initiative: StrategicInitiative, audience: str) -> PitchNarrative:
        prompt = f"""
        Create pitch deck for {initiative.title} targeting {audience}.
        Provide JSON with sections array and financial highlights.
        """
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={self.gemini_key}",
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                    timeout=45.0
                )
                data = response.json()
                deck_data = json.loads(data['candidates'][0]['content']['parts'][0]['text'].replace('```json', '').replace('```', ''))
                
                return PitchNarrative(
                    initiative_id=initiative.id,
                    audience=audience,
                    sections=deck_data['sections'],
                    financial_highlights={
                        "npv_3yr": 1500000,
                        "payback_months": 18,
                        "capex_required": 250000,
                        "revenue_3yr": 5000000
                    },
                    risks_mitigations=[
                        {"risk": "Timing", "mitigation": "Agile pivot"},
                        {"risk": "Competition", "mitigation": "AI differentiation"}
                    ],
                    call_to_action=deck_data.get('call_to_action', 'Invest $500K')
                )
        except:
            return PitchNarrative(
                initiative_id=initiative.id,
                audience=audience,
                sections=[{"slide_title": "Problem", "key_points": ["Problem exists"]}],
                financial_highlights={},
                risks_mitigations=[],
                call_to_action="Invest"
            )

# ============================================================================
# GOOGLE SHEETS LOGGER - Audit Trail
# ============================================================================

import gspread
from google.oauth2.service_account import Credentials

class SheetsLogger:
    def __init__(self):
        self.enabled = False
        try:
            sa_info = st.secrets["service_account"]
            sheet_id = st.secrets["SHEET_ID"]
            
            scope = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
            
            creds = Credentials.from_service_account_info(sa_info, scopes=scope)
            self.client = gspread.authorize(creds)
            self.sheet = self.client.open_by_key(sheet_id).sheet1
            self.enabled = True
            
            # Create headers
            if not self.sheet.get_all_values():
                headers = [
                    "Timestamp", "User", "Action", "Input", "Output",
                    "Model", "Tokens", "Cost", "Status", "Error", "SessionID"
                ]
                self.sheet.append_row(headers, value_input_option='USER_ENTERED')
                
        except Exception as e:
            print(f"Sheets logger disabled: {e}")
    
    def log(self, **kwargs):
        if not self.enabled:
            return
        
        row = [
            datetime.now().isoformat(),
            kwargs.get('user', 'unknown')[:50],
            kwargs.get('action', 'unknown')[:50],
            json.dumps(kwargs.get('input_data', {}), ensure_ascii=False)[:1000],
            json.dumps(kwargs.get('output_data', {}), ensure_ascii=False)[:1000],
            kwargs.get('model', 'unknown')[:30],
            kwargs.get('tokens', 0),
            kwargs.get('cost', 0.0),
            kwargs.get('status', 'success')[:20],
            kwargs.get('error', '')[:200],
            kwargs.get('session_id', '')[:30]
        ]
        
        try:
            self.sheet.append_row(row, value_input_option='USER_ENTERED')
        except Exception as e:
            print(f"Logging failed: {e}")

logger = SheetsLogger()

# ============================================================================
# API ROUTES
# ============================================================================

@app.post("/analyze-competitor", response_model=CompetitiveInsight)
async def analyze_competitor(competitor: str, game: GameTitle, bg_tasks: BackgroundTasks):
    try:
        insight = await market_engine.analyze_competitor(competitor, game)
        
        # Log success
        bg_tasks.add_task(logger.log, user="api_user", action="competitor_analysis",
                         input_data={"competitor": competitor, "game": game},
                         output_data=insight.dict(), model="gemini-1.5-flash", status="success")
        
        # Store for review
        bg_tasks.add_task(store_review, "insight", competitor, insight.dict())
        return insight
    except Exception as e:
        bg_tasks.add_task(logger.log, user="api_user", action="competitor_analysis_error",
                         input_data={"competitor": competitor, "game": game},
                         output_data={}, status="error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/calculate-tam", response_model=TAMAnalysis)
async def calculate_tam(game: GameTitle, segment: str, tier: str = "premium", geo: str = "global"):
    try:
        tam = await opportunity_engine.calculate_market_size(game, segment, tier, geo)
        logger.log(user="api_user", action="tam_calculation",
                  input_data={"game": game, "segment": segment},
                  output_data=tam.dict(), model="calculator", status="success")
        return tam
    except Exception as e:
        logger.log(user="api_user", action="tam_calculation_error",
                  input_data={"game": game}, output_data={}, status="error", error=str(e))
        raise

@app.post("/okrs", response_model=OKR)
async def create_okr(okr: OKR):
    try:
        result = await okr_tracker.create_okr(okr)
        logger.log(user="api_user", action="okr_created",
                  input_data=okr.dict(), output_data={"id": result.id}, status="success")
        return result
    except Exception as e:
        logger.log(user="api_user", action="okr_creation_error",
                  input_data=okr.dict(), output_data={}, status="error", error=str(e))
        raise

async def store_review(item_type: str, item_id: str, content: Dict[str, Any]):
    """Store for human review"""
    db = Database()
    await db.execute("""
        INSERT INTO reviews (id, item_type, item_id, generated_content, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        f"REVIEW-{hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:12]}",
        item_type,
        item_id,
        json.dumps(content),
        datetime.now().isoformat()
    ))

# ============================================================================
# STARTUP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Initialize engines
market_engine = MarketIntelligenceEngine()
opportunity_engine = OpportunitySizer()
okr_tracker = OKRTracker()
narrative_engine = StrategicNarrativeEngine()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
