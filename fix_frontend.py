#!/usr/bin/env python3
"""
Nuclear fix for hidden Unicode in frontend.py
"""

import re
import sys

# Read the file
with open('frontend.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Pattern to catch ANY Unicode whitespace between , and .
# This matches: comma + any whitespace (including \xa0) + dot + 2f
broken_pattern = r'(:,\s*)(\.\s*2f)'

# Replace with: comma + dot + 2f (no space)
fixed_content = re.sub(broken_pattern, r':,\2', content)

# Also fix any other similar patterns in the file
# This catches all instances of {var:,.2f} with hidden spaces
fixed_content = re.sub(r'(\w+):\s*\.\s*2f', r'\1:,.2f', fixed_content)

# Write back
with open('frontend.py', 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print("✅ FIXED: Removed hidden Unicode characters")
print("Now redeploy your app")
