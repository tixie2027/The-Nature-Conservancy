import json
from pathlib import Path

# --- load files -------------------------------------------------------------
with open("benchmarking_mapped.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open(Path("/Users/sharm51155/Desktop/TNC/The-Nature-Conservancy/pdf_processing/matches.json"),
          "r", encoding="utf-8") as f:
    matches = json.load(f)

# --- build a fast title → key lookup ---------------------------------------
title_to_match_key = {
    v["title"].strip().lower(): k          # value = original key in matches.json
    for k, v in matches.items()
    if "title" in v and isinstance(v["title"], str)
}

# --- iterate through questions ---------------------------------------------
for question, details in data.items():
    actual_title = details[3].strip()      # the title stored in benchmarking_mapped.json
    match_key    = title_to_match_key.get(actual_title.lower())

    print(f"Actual title : {actual_title}")
    print(f"JSON title   : {match_key}\n")
