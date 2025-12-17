"""
Google Sheets Logger for Streamlit Cloud
"""

import json
from datetime import datetime
import streamlit as st

class SheetsLogger:
    def __init__(self):
        self.enabled = False
        
        try:
            sa_info = st.secrets["service_account"]
            sheet_id = st.secrets["SHEET_ID"]
            
            from google.oauth2.service_account import Credentials
            import gspread
            
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
                
            print("✅ Sheets logger connected")
            
        except Exception as e:
            print(f"⚠️ Sheets logger disabled: {e}")
    
    def log(self, **kwargs):
        if not self.enabled:
            print(f"MOCK LOG: {kwargs.get('action')}")
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

_sheets_logger_instance = None

def get_logger():
    global _sheets_logger_instance
    if _sheets_logger_instance is None:
        _sheets_logger_instance = SheetsLogger()
    return _sheets_logger_instance
