#!/usr/bin/env python3
"""
Cleans hidden unicode characters from frontend.py
"""

import re

# Read the problematic file
with open('frontend.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Pattern to find the broken f-string (with any unicode space)
pattern = r'total_budget:\s*\.\s*2f'
replacement = 'total_budget:,.2f'

# Clean it
cleaned_content = re.sub(pattern, replacement, content)

# Write back
with open('frontend.py', 'w', encoding='utf-8') as f:
    f.write(cleaned_content)

print("✅ Cleaned frontend.py")
print("Now redeploy your app")
