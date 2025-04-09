import json
import ollama

# Load JSON
with open("cleaned_Verma_2014_biomass.json", "r", encoding="utf-8") as file:
    raw_json = file.read()

questionToAsk = f"Based on the introduction or the abstract, is this paper talking about agroforestry carbon, biomass, or soil composition?:\n\n{raw_json}"

desiredModel = 'llama3.1:8b'

# Ask the question
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

# Write the response to a text file
with open("OutputOllama.txt", "w", encoding="utf-8") as text_file:
    text_file.write(OllamaResponse)

