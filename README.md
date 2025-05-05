# Nature Conservancy PDF-to-JSON Pipeline

Developed by Harvey Mudd College students in collaboration with The Nature Conservancy (TNC), this project transforms ecological research articles in PDF format into clean, machine-readable JSON. It enables downstream applications such as search, summarization, and retrieval-augmented generation (RAG).

---

## Repository Contents

- `docling_test.py`: Uses IBM’s Docling to convert PDFs into Markdown (`.md`).
- `mdtojson.py`: Converts Markdown to structured JSON format with section headings.
- `json_editor.py`: Cleans Unicode artifacts and normalizes text in the JSON.
- `json_heading_fix.ipynb`: (Notebook) Standardizes JSON structure into sections like Introduction, Methods, Results, etc.
- `main.py`: Automates the pipeline — takes a PDF, outputs cleaned JSON in one step.
- `example_data.zip`: Contains example inputs and outputs.
- `Llama3.1_8B.py`: (In development) Uses LLaMA 3.1 with Ollama for local RAG on processed JSON documents.

---

## Requirements

See `requirements.txt` for all dependencies, including:
- `docling`
- `sentence-transformers`
- `faiss-cpu`
- `langchain_huggingface`
- `transformers`
- `groq`, `langchain_community`
- `python-dotenv`, `scikit-learn`

Install with:

```bash
pip install -r requirements.txt
```

---

## Usage

1. Place PDFs inside the `training_data/` directory.
2. Run the pipeline:
   ```bash
   python main.py
   ```
   This will:
   - Extract Markdown from PDFs using Docling.
   - Convert `.md` to `.json`.
   - Clean text and save final output to `finished_data/`.

3. (Optional) Standardize structure using `json_heading_fix.ipynb`.

4. (Optional) Run `Llama3.1_8B.py` for retrieval-based question answering on parsed documents.

---

## Example Application

- Identify whether a paper discusses agroforestry, carbon sequestration, or soil composition using LLaMA 3.1 + Ollama (see `Llama3.1_8B.py`).

---

## General Workflow

> ![Pipeline Workflow](https://github.com/user-attachments/assets/99370082-66ec-4c12-ade6-42b52101e4f0)

---

## Acknowledgments

This project is made possible by Harvey Mudd College students and faculty, in partnership with The Nature Conservancy.
