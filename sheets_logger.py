# sheets_logger.py
import json
import streamlit as st
from typing import Dict, Any

class SheetsLogger:
    def __init__(self, service_account_info: Dict[str, Any] = None, sheet_id: str = None):
        """Initialize with optional Google Sheets - mocks if unavailable"""
        self.enabled = False
        
        # Skip if no credentials provided
        if not service_account_info or not sheet_id:
            st.toast("📊 Sheets logging disabled (demo mode)", icon="ℹ️")
            return
        
        try:
            from google.oauth2.service_account import Credentials
            import gspread
            
            scope = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
            
            creds = Credentials.from_service_account_info(
                service_account_info, 
                scopes=scope
            )
            self.client = gspread.authorize(creds)
            self.sheet = self.client.open_by_key(sheet_id).sheet1
            self.enabled = True
            
            # Initialize headers if empty
            if not self.sheet.get_all_values():
                self.sheet.append_row([
                    "Timestamp", "User", "Action", "Input", "Output", 
                    "Model", "Tokens", "Cost", "Status"
                ])
            st.toast("✅ Sheets logger initialized", icon="✔")
            
        except Exception as e:
            st.toast(f"Sheets logger failed: {e}. Using mock mode.", icon="⚠️")
            self.enabled = False
    
    def log_interaction(self, **kwargs):
        """Log interaction or mock if disabled"""
        if not self.enabled:
            # Mock logging - just show a toast
            st.toast(f"📊 Logged: {kwargs.get('action', 'action')}", icon="📝")
            return
        
        # Real logging
        try:
            row = [
                st.session_state.get('session_start', 'N/A'),
                kwargs.get('user', 'unknown'),
                kwargs.get('action', 'unknown'),
                json.dumps(kwargs.get('input_data', {}), ensure_ascii=False)[:500],
                json.dumps(kwargs.get('output_data', {}), ensure_ascii=False)[:500],
                kwargs.get('model', 'unknown'),
                kwargs.get('tokens', 0),
                kwargs.get('cost_estimate', 0.0),
                kwargs.get('status', 'success')
            ]
            self.sheet.append_row(row, value_input_option='USER_ENTERED')
            st.toast("✅ Logged to Google Sheets", icon="📊")
        except Exception as e:
            st.toast(f"Logging error: {e}", icon="❌")

# Convenience function
def get_logger():
    """Get logger instance with fallback"""
    try:
        # Try to get from secrets
        sa_info = st.secrets.get("service_account", None)
        sheet_id = st.secrets.get("SHEET_ID", None)
        return SheetsLogger(sa_info, sheet_id)
    except:
        return SheetsLogger()  # Mock mode
