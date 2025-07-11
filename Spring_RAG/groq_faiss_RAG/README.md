
# Agroforestry RAG System


A lightweight Retrieval-Augmented Generation (RAG) system designed to answer user questions related to carbon sequestration, biomass modeling, and agroforestry using peer-reviewed scientific literature.

  

## Overview

  

This project loads a set of structured scientific JSON files related to agroforestry, embeds their content using Sentence Transformers, and indexes them using FAISS. It then allows users to input natural language questions, retrieves the most relevant excerpts, and generates grounded responses using the Groq LLM API (e.g., LLaMA3).

  

## Components

  

-  `index_script.py`: Builds a FAISS index from pre-parsed JSON documents containing agroforestry research.

-  `main.py`: Command-line chatbot that performs similarity search + LLM reasoning with Groq.

-  `requirements.txt`: All Python dependencies.

-  `faiss_index/`: Contains the FAISS index (`index.faiss`, `index.pkl`).

-  `cleaned_*.json`: Structured and cleaned papers (sections with titles + paragraphs).

- `benchmarking.json`: Agroforestry LLM benchmarking dataset created from human labelled data by TNC and agroforestry scientists

- `benchmarking.py`: Evaluates the answer quality of an LLM by sampling questions from  `benchmarking.json`

- `benchmarking_results.json`: Recorded results from most recent benchmark 

- `TNC.code-workspace`: Shortcut to access complete article JSON files in local computer Downloads folder

  

## Example Papers Included

  

- Verma 2014: Predictive models for biomass in Grewia optiva.

- Upson & Burgess 2013: Soil organic carbon and root distribution in silvoarable systems.

- N'Gbala 2017: Carbon stock comparisons in cocoa, teak, and forests.

- Kaonga 2012: CO2FIX model simulation in Zambia woodlots.

  

## Setup

  

1.  **Clone the repository**

  

2.  **Install dependencies**

```bash

pip install -r requirements.txt

```

  

3.  **Set environment variable**

Create a `.env` file with:

```

GROQ_API_KEY=your_api_key_here

```

  

4.  **Build the FAISS index (if not already built)**

```bash

python index_script.py

```

  

5.  **Run the chatbot interface**

```bash

python main.py

```

  

## Sample Usage

  

```

Question: What are common models for estimating biomass in Grewia optiva?

  

Answer:

"Out of the six non-linear models, allometric model (Y = a × DBH^b) fulfills the validation criterion to the best possible extent..." (Predictive models for biomass and carbon stocks estimation in Grewia optiva)

```

  

## Dependencies

  

See `requirements.txt` for full list, including:

-  `sentence-transformers`

-  `faiss-cpu`

-  `langchain`

-  `groq`

-  `transformers`

  

## Notes

  

- Each JSON paper must follow the structure:

```json

{

"sections": [

{ "heading": "Introduction", "content": ["text", "text", ...] },

...

]

}

```

- All content is embedded using `sentence-transformers/all-MiniLM-L6-v2`.