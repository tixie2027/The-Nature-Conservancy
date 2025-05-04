import json
import ftfy
import re

# Helper function to clean text using ftfy
def clean_text_with_ftfy(text):
    # Fix Unicode issues
    fixed_text = ftfy.fix_text(text)

    # Ligature and glyph name replacements
    ligature_replacements = {
        ' /uniFB01 ': 'fi',   # Replace fi ligature glyph
        '/uniFB02': 'fl',   # Replace fl ligature glyph (if encountered)
        ' /C0 ': '-',
        ' /C6 ': '+-',
    }

    # Additional replacements (newlines, dashes, control characters)
    replacements = {
        '\n': ' ',           # Replace newlines with spaces
        '\u2013': '-',       # Replace en dash with hyphen
        '\u0002': '|',       # Replace control characters with pipe
        '\u0003': ''         # Example: Remove any control chars
    }

    # Apply ligature replacements
    for key, value in ligature_replacements.items():
        fixed_text = fixed_text.replace(key, value)

    # Apply general replacements
    for key, value in replacements.items():
        fixed_text = fixed_text.replace(key, value)

    # Clean extra spaces
    return re.sub(r'\s+', ' ', fixed_text).strip()

# Recursively clean all text in JSON structure
def clean_json(data):
    if isinstance(data, dict):
        return {key: clean_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_json(item) for item in data]
    elif isinstance(data, str):
        return clean_text_with_ftfy(data)
    else:
        return data

