"""
Google Sheets Logger - Enhanced Debug Version
"""

import json
from datetime import datetime
import streamlit as st

class SheetsLogger:
    def __init__(self):
        self.enabled = False
        self.error = None
        
        try:
            # Debug: Show what we're trying to load
            print("🔍 Trying to load service account...")
            
            sa_info = st.secrets["service_account"]
            sheet_id = st.secrets["SHEET_ID"]
            
            st.sidebar.success("✅ Service Account found in secrets")
            st.sidebar.success(f"✅ Sheet ID: {sheet_id[:10]}...")
            
            from google.oauth2.service_account import Credentials
            import gspread
            
            scope = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
            
            # Debug: Show keys we're extracting
            required_keys = ['type', 'project_id', 'private_key', 'client_email']
            missing_keys = [k for k in required_keys if k not in sa_info]
            if missing_keys:
                raise ValueError(f"Missing keys in service account: {missing_keys}")
            
            creds = Credentials.from_service_account_info(sa_info, scopes=scope)
            self.client = gspread.authorize(creds)
            self.sheet = self.client.open_by_key(sheet_id).sheet1
            self.enabled = True
            
            print("✅ Sheets logger fully initialized")
            st.sidebar.success("✅ Sheets logger connected")
            
            # Create headers if empty
            if not self.sheet.get_all_values():
                headers = [
                    "Timestamp", "User", "Action", "Input", "Output",
                    "Model", "Tokens", "Cost", "Status", "Error", "SessionID"
                ]
                self.sheet.append_row(headers, value_input_option='USER_ENTERED')
                print("✅ Headers created")
                
        except Exception as e:
            self.error = str(e)
            print(f"❌ Sheets logger failed: {e}")
            st.sidebar.error(f"Logger init failed: {e}")
            st.sidebar.info("Running in mock mode")
    
    def log(self, **kwargs):
        """Log to Google Sheets with error reporting"""
        if not self.enabled:
            st.toast(f"⚠️ Mock log: {kwargs.get('action')}", icon="📝")
            return False
        
        try:
            row = [
                datetime.now().isoformat(),
                kwargs.get('user', 'anonymous')[:50],
                kwargs.get('action', 'unknown')[:50],
                json.dumps(kwargs.get('input_data', {}), default=str)[:1000],
                json.dumps(kwargs.get('output_data', {}), default=str)[:1000],
                kwargs.get('model', 'unknown')[:30],
                kwargs.get('tokens', 0),
                kwargs.get('cost', 0.0),
                kwargs.get('status', 'success')[:20],
                kwargs.get('error', '')[:200],
                kwargs.get('session_id', '')[:30]
            ]
            
            self.sheet.append_row(row, value_input_option='USER_ENTERED')
            print(f"✅ LOGGED: {kwargs.get('action')}")
            return True
            
        except Exception as e:
            print(f"❌ Logging error: {e}")
            st.toast(f"Logging failed: {e}", icon="❌")
            return False

_sheets_logger_instance = None

def get_logger() -> SheetsLogger:
    """Singleton pattern"""
    global _sheets_logger_instance
    if _sheets_logger_instance is None:
        _sheets_logger_instance = SheetsLogger()
    return _sheets_logger_instance
