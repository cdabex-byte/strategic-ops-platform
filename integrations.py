"""
Real API Integrations: SteamSpy, Twitch, Slack, HubSpot
"""

import httpx
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import streamlit as st

class SteamSpyAPI:
    """Real Steam player data"""
    def __init__(self):
        self.base_url = "https://steamspy.com/api.php"
    
    async def get_game_data(self, app_id: str) -> Dict[str, Any]:
        """Get real player counts for games"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}?request=appdetails&appid={app_id}",
                timeout=10.0
            )
            return response.json()

class TwitchAPI:
    """Get live viewership data"""
    def __init__(self):
        self.client_id = st.secrets.get("TWITCH_CLIENT_ID", "")
        self.client_secret = st.secrets.get("TWITCH_CLIENT_SECRET", "")
        self.token = None
    
    async def get_token(self):
        """Get OAuth token"""
        if not self.token or self.token["expires_at"] < datetime.now():
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://id.twitch.tv/oauth2/token",
                    params={
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "grant_type": "client_credentials"
                    }
                )
                data = resp.json()
                self.token = {
                    "access_token": data["access_token"],
                    "expires_at": datetime.now() + timedelta(seconds=data["expires_in"] - 300)
                }
        return self.token["access_token"]
    
    async def get_game_viewership(self, game_name: str) -> Dict[str, Any]:
        token = await self.get_token()
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.twitch.tv/helix/streams",
                headers={"Authorization": f"Bearer {token}", "Client-Id": self.client_id},
                params={"game_name": game_name, "first": 100}
            )
            return response.json()

class SlackNotifier:
    """Real-time alerts to #strategy channel"""
    def __init__(self):
        self.webhook_url = st.secrets.get("SLACK_WEBHOOK_URL", "")
    
    async def send_alert(self, message: str, priority: str = "medium"):
        if not self.webhook_url:
            return
        
        payload = {
            "text": f"🎮 Strategic Ops Alert ({priority})",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": message
                    }
                }
            ]
        }
        
        async with httpx.AsyncClient() as client:
            await client.post(self.webhook_url, json=payload)

class HubSpotCRM:
    """Track partnership deals"""
    def __init__(self):
        self.api_key = st.secrets.get("HUBSPOT_API_KEY", "")
    
    async def create_deal(self, company: str, deal_type: str, value: float) -> str:
        """Create a partnership deal in HubSpot"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.hubapi.com/crm/v3/objects/deals",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "properties": {
                        "dealname": f"{deal_type} - {company}",
                        "amount": value,
                        "pipeline": "default",
                        "dealstage": "appointmentscheduled"
                    }
                }
            )
            return response.json()["id"]
