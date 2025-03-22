import os
import json
from pathlib import Path

def convert_md_to_json(md_file, json_file):
    with open(md_file, "r", encoding="utf-8") as f:
        md_content = f.readlines()

    # Structure to hold parsed content
    md_structure = {"sections": []}
    current_section = {"heading": None, "content": []}

    # Parse Markdown line by line
    for line in md_content:
        line = line.strip()

        if line.startswith("## "):  # Detect section heading (second-level)
            # Save the previous section if it exists
            if current_section["heading"] or current_section["content"]:
                md_structure["sections"].append(current_section)
                current_section = {"heading": None, "content": []}

            # Start a new section
            current_section["heading"] = line.lstrip("# ").strip()

        elif line:  # Non-empty lines (add content to the section)
            current_section["content"].append(line)

    # Add the last section if it exists
    if current_section["heading"] or current_section["content"]:
        md_structure["sections"].append(current_section)

    # Write the parsed content to a JSON file
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(md_structure, f, ensure_ascii=False, indent=4)

    print(f"Markdown content successfully converted to {json_file}")
