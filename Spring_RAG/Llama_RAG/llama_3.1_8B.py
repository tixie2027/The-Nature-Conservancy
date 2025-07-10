import sqlite3
import json
import ollama
import re

# Connect to database
conn = sqlite3.connect('papers.db')
cursor = conn.cursor()

# Create table if not exists
cursor.execute('''
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    confidence_agroforestry INTEGER,
    confidence_biomass_carbon_growth INTEGER,
    confidence_review_vs_research TEXT,
    confidence_score_type INTEGER,
    geographic_location TEXT
)
''')
conn.commit()


# Load and read raw JSON text
with open("cleaned_Verma_2014_biomass.json", "r", encoding="utf-8") as file:
    raw_json = file.read()

# Ask the model to analyze it
questionToAsk = (
    f"Please analyze the following JSON and explain what it's about:\n\n{raw_json}\n\n"
    "1. Rate your confidence (1–10) that the paper mentions agroforestry.\n"
    "2. Rate your confidence (1–10) that the paper mentions biomass, carbon, or plant growth.\n"
    "3. Is this a literature review or a novel research paper? Output one of: 'review' or 'research'. Also, rate your confidence (1–10).\n"
    "4. Does the paper mention any geographical location(s)? Output one of: 'yes' or 'no'. Also, rate your confidence (1-10).\n"
    "Return your answer in JSON format with the keys: confidence_agroforestry, confidence_biomass_carbon_growth, confidence_review_vs_research, confidence_score_type, geographic_location."
)

desiredModel = 'llama3.1:8b'

response = ollama.chat(
    model=desiredModel,
    messages=[
        {
            'role': 'user',
            'content': questionToAsk,
        },
    ]
)

OllamaResponse = response['message']['content']
print(OllamaResponse)


json_part = re.search(r'\{.*\}', OllamaResponse, re.DOTALL).group()
parsed_json = json.loads(json_part)

# Insert into database
cursor.execute('''
INSERT INTO papers (filename, confidence_agroforestry, confidence_biomass_carbon_growth, confidence_review_vs_research, confidence_score_type, geographic_location)
VALUES (?, ?, ?, ?, ?, ?)
''', (
    "cleaned_Verma_2014_biomass.json",
    parsed_json['confidence_agroforestry'],
    parsed_json['confidence_biomass_carbon_growth'],
    parsed_json['confidence_review_vs_research'],
    parsed_json['confidence_score_type'],
    parsed_json['geographic_location']
))

conn.commit()
conn.close()


