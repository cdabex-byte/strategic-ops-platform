import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import json

class SheetsLogger:
    def __init__(self, service_account_file: str, sheet_id: str):
        """Initialize Google Sheets connection"""
        scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        
        creds = Credentials.from_service_account_file(
            service_account_file, scopes=scope
        )
        self.client = gspread.authorize(creds)
        self.sheet = self.client.open_by_key(sheet_id).sheet1
        
        # Initialize headers if empty
        if not self.sheet.get_all_values():
            self.sheet.append_row([
                "Timestamp", "User", "Action", "Input", "Output", 
                "Model Used", "Tokens", "Cost Estimate", "Status"
            ])
    
    def log_interaction(
        self, 
        user: str,
        action: str,
        input_data: dict,
        output_data: dict,
        model: str,
        tokens: int = 0,
        cost_estimate: float = 0.0,
        status: str = "success"
    ):
        """Log API call to Google Sheets"""
        row = [
            datetime.now().isoformat(),
            user,
            action,
            json.dumps(input_data, ensure_ascii=False)[:500],  # Truncate long inputs
            json.dumps(output_data, ensure_ascii=False)[:500],
            model,
            tokens,
            cost_estimate,
            status
        ]
        
        try:
            self.sheet.append_row(row, value_input_option='USER_ENTERED')
            print(f"✅ Logged to Google Sheets: {action}")
        except Exception as e:
            print(f"❌ Sheets logging failed: {e}")

# Global logger instance
logger = None

def init_logger(service_account_file: str, sheet_id: str):
    global logger
    logger = SheetsLogger(service_account_file, sheet_id)
