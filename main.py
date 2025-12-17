"""
Strategic Operations Platform API v1.0
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
# DATABASE & MODELS
# ============================================================================

DB_PATH = "strategic_ops.db"

def init_db():
    """Initialize SQLite with schema"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
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
# ENGINES
# ============================================================================

class MarketIntelligenceEngine:
    def __init__(self):
        self.gemini_key = st.secrets.get("GEMINI_API_KEY", "")
    
    async def analyze_competitor(self, competitor: str, game: GameTitle) -> CompetitiveInsight:
        prompt = f"""
        Analyze {competitor} in {game} AI coaching. Provide JSON with:
        key_strengths, key_weaknesses, market_position, threat_level (1-10),
        opportunity_windows, confidence_score (0-1), sources
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
        except:
            # Fallback
            return CompetitiveInsight(
                competitor_name=competitor,
                game=game,
                key_strengths=["Strong brand", "Large base"],
                key_weaknesses=["Limited AI", "High price"],
                market_position="challenger",
                threat_level=6,
                opportunity_windows=["Mobile", "Esports"],
                confidence_score=0.7,
                sources=["Mock"]
            )

class OpportunitySizer:
    async def calculate_market_size(self, game: GameTitle, segment: str, tier: str, geo: str) -> TAMAnalysis:
        arpu_map = {"premium": 120, "mid": 60, "free": 0}
        total_players = 1_000_000
        arpu = arpu_map.get(tier, 60)
        
        tam = total_players * arpu
        penetration_rate = 0.15
        sam = int(tam * penetration_rate)
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
            assumptions={"arpu": arpu, "penetration": penetration_rate},
            confidence_score=0.6
        )

# ============================================================================
# GOOGLE SHEETS LOGGER
# ============================================================================

import gspread
from google.oauth2.service_account import Credentials

class SheetsLogger:
    def __init__(self):
        self.enabled = False
        self.sheet = None
        
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
            
            # Create headers if empty
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
            print("Mock logging:", kwargs.get('action'))
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.post("/analyze-competitor", response_model=CompetitiveInsight)
async def analyze_competitor(competitor: str, game: GameTitle, bg_tasks: BackgroundTasks):
    try:
        engine = MarketIntelligenceEngine()
        insight = await engine.analyze_competitor(competitor, game)
        
        # Log to Sheets
        bg_tasks.add_task(logger.log, user="api_user", action="competitor_analysis",
                         input_data={"competitor": competitor, "game": game},
                         output_data=insight.dict(), model="gemini-1.5-flash", status="success")
        
        # Store for human review
        bg_tasks.add_task(store_review, "insight", competitor, insight.dict())
        return insight
    except Exception as e:
        bg_tasks.add_task(logger.log, user="api_user", action="competitor_analysis_error",
                         input_data={"competitor": competitor}, output_data={}, status="error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/calculate-tam", response_model=TAMAnalysis)
async def calculate_tam(game: GameTitle, segment: str, tier: str = "premium", geo: str = "global"):
    try:
        engine = OpportunitySizer()
        tam = await engine.calculate_market_size(game, segment, tier, geo)
        
        logger.log(user="api_user", action="tam_calculation",
                  input_data={"game": game, "segment": segment},
                  output_data=tam.dict(), model="calculator", status="success")
        return tam
    except Exception as e:
        logger.log(user="api_user", action="tam_calculation_error",
                  input_data={"game": game}, output_data={}, status="error", error=str(e))
        raise

async def store_review(item_type: str, item_id: str, content: Dict[str, Any]):
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
# RUN
# ============================================================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
