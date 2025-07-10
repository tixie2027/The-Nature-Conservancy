import os
import json
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Set path to your folder
parsed_dir = os.path.join(os.path.dirname(__file__), "parsed_json_example")

# Load SentenceTransformer embedding model
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

docs = []

for file in tqdm(os.listdir(parsed_dir)):
    if file.endswith(".json"):
        with open(os.path.join(parsed_dir, file), "r") as f:
            data = json.load(f)

            # Flatten all content from the sections
            content_blocks = []
            for section in data.get("sections", []):
                heading = section.get("heading")
                section_text = "\n".join(section.get("content", []))
                if heading:
                    content_blocks.append(f"{heading}\n{section_text}")
                else:
                    content_blocks.append(section_text)

            full_text = "\n\n".join(content_blocks)

            # Use filename as fallback title
            title = file.replace(".json", "").replace("_", " ")

            docs.append(Document(page_content=full_text, metadata={"title": title}))

# Build FAISS index and save
faiss_index = FAISS.from_documents(docs, embedding=embedding_function)
faiss_index.save_local("faiss_index")