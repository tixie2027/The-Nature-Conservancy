
# Nature Conservancy RAG-LLM Pipeline (Spring Version)

  

  

Note: For the most up-to-date pipeline, please use the code form the `Summer_RAG` folder.

  

The code in this folder is an initial version of our pipeline, which we built off of the code provided to us by the Spring CTC team. One of the main differences between the Spring and Summer versions is that the Spring version must be ran within a code editor (like Visual Studio Code), and the Summer version is a Jupyter Notebook can be ran in Google Colab (or Visual Studio Code). 


---

  

  

## Folder Contents
  

-  `groq_faiss_rag`: Uses FAISS as vector database and retrieves LLM models from Groq API. 
  

-  `Llama_RAG`:  Uses LLaMA 3.1 with Ollama for local RAG on processed JSON documents.

  


---

  


## Requirements

  

  

See `requirements.txt` for all dependencies, including:

  

-  `sentence-transformers`

  

-  `faiss-cpu`

  

-  `langchain_huggingface`

  

-  `transformers`

  

-  `groq`, `langchain_community`

  

-  `python-dotenv`, `scikit-learn`

  

  

Install with:

  

  

```bash

  

pip  install  -r  requirements.txt

  

```

 

---

  

  

## Usage

  
Run the pipeline:

  

```bash

python  main.py

```
  

This will:

  
- Use the converted files from `finished_data/` as the context from which to retrieve additional information
- Intake a user query related to agroforestry
- Output answer and cite articles on which the answer is based upon


---

 
## Acknowledgments

  

  

This project is made possible by Harvey Mudd College and Pomona College students and faculty, in partnership with The Nature Conservancy.