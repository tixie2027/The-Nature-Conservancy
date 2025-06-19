from dotenv import load_dotenv
load_dotenv()
from groq import Groq
import json
from pathlib import Path
from termcolor import colored
import spacy
import nltk

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer
nltk.download("punkt")

# OLD ARTICLE PICKER:

# def get_relevant_excerpts(user_question, docsearch):
#     """
#     Perform similarity search and return top 3 excerpts with titles.
#     """
#     relevant_docs = docsearch.similarity_search(user_question, k=3)
#     excerpts = []

#     for doc in relevant_docs:
#         title = doc.metadata.get("title", "Unknown Source")
#         excerpt = f"📄 **{title}**\n{doc.page_content.strip()[:2000]}..."  # Truncate for safety
#         excerpts.append(excerpt)

#     return "\n\n---\n\n".join(excerpts)


def cosine(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))


def get_relevant_excerpts(query,q_vec,embedding_function,metadata,folder_of_jsons):
   
    ## ELIMINATE ARTICLES WITHOUT MAIN DETAILS IN Q
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(query)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON", "GPE", "LOC", "PRODUCT", "NORP", "FAC"]]
    print(colored("Retrieved extracts:"))
    print(colored(entities, "blue"))
    
    # Only filter if we found entities in the query
    if entities:
        articles_to_remove = []
        
        # Loop through folder of jsons
        for filename, meta in metadata.items():
            article_path = Path(folder_of_jsons, filename)
            try:
                with open(article_path, 'r', encoding='utf-8') as f:
                    article_json = f.read()  # Read as string
                    
                    # Check if ALL entities are present in the article string
                    entity_missing = False
                    for entity in entities:
                        if entity.lower() not in article_json.lower():
                            entity_missing = True
                            break
                    
                    # If even one entity is missing, mark for removal
                    if entity_missing:
                        articles_to_remove.append(filename)
                        
            except (FileNotFoundError, UnicodeDecodeError) as e:
                print(colored(f"Error processing {filename}: {e}", "red"))
                articles_to_remove.append(filename)
        
        if(len(articles_to_remove)!=len(metadata)):
        # Remove articles that don't contain ALL entities
            for filename in articles_to_remove:
                metadata.pop(filename, None)
        
        print(colored(f"Filtered from {len(metadata) + len(articles_to_remove)} to {len(metadata)} articles", "green"))

        

    
    ## KEY WORDS
    for filename, meta in metadata.items():  
        keywords = meta["key_words"]
        for i in range(len(keywords)):
            if(keywords[i] in query):
                meta["score"] += 0.5
    ## SUMMARY COMPARISONS
    for meta in metadata.values():
        if "summary_vec" not in meta:              # cache so can reuse later
            meta["summary_vec"] = embedding_function.embed_query(meta["summary"])

        sim = cosine(q_vec, meta["summary_vec"])   # −1 … 1
        meta["score"] += sim      

    
    ranked = sorted(metadata.items(), key=lambda x: x[1]["score"], reverse=True)
    relevant_articles = []

    for rank, (fname, meta) in enumerate(ranked[:5], start=1):  # top-5
        file_path = Path(folder_of_jsons, fname)
        print(f"{fname:55s}  score={meta['score']:.3f}")
        print(colored(meta["key_words"], "red"))
 
        with open(file_path, "r", encoding="utf-8") as f:
            article_dict = json.load(f)

        article_str = json.dumps(article_dict, indent=2, ensure_ascii=False)

        relevant_articles.append({
            "label": f"ARTICLE {rank}",          # or use meta.get("title", …)
            "text":  article_str
        })

    return relevant_articles
    

def rerank_context_per_article(articles, query, sentences_per_article, embed_model):
    """
    For each article:
      • split into sentences
      • rerank sentences vs. the query
      • keep the top k
    Return a single string that preserves ARTICLE 1 / ARTICLE 2 … blocks.
    """
    blocks = []

    for art in articles:
        sentences   = chunk_document(art["text"])
        reranked    = rerank_chunks(sentences, query, embed_model)
        top_k_sents = [s for s, _ in reranked[:sentences_per_article]]

        block = f'{art["label"]}:\n' + "\n".join(top_k_sents)
        blocks.append(block)
    print("\n\n".join(blocks))
    return "\n\n".join(blocks)    

def chunk_document(text, max_tokens=510):
    max_chars = max_tokens * 3
    chunks, current_chunk = [], []
    sentences = sent_tokenize(text)
    i = 0
    
    while i < len(sentences):
        sentence = sentences[i]
        
        # check if this sentence starts a table
        if '|' in sentence and sentence.strip().startswith('|'):
            table_rows = []
            j = i
            
            # finding all table rows
            while j < len(sentences) and '|' in sentences[j] and sentences[j].strip().startswith('|'):
                table_rows.append(sentences[j])
                j += 1
            
            complete_table = '\n'.join(table_rows)
            test_chunk = " ".join(current_chunk + [complete_table])

            # checking if goes over char limit
            if len(test_chunk) > max_chars:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                
                # checking if table goes over limit
                if len(complete_table) > max_chars:
                    chunks.append(complete_table)
                else:
                    current_chunk = [complete_table]
            # under limit
            else:
                current_chunk.append(complete_table)

            i = j
            continue
        #Below is for when the sentence isn't a table
        test_chunk = " ".join(current_chunk + [sentence])
        if len(test_chunk) > max_chars:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
        else:
            current_chunk.append(sentence)
        i += 1

    # adding back anything left in the current chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def embed_texts(texts, model):
    """
    Embedding text, given model
    """
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings

def rerank_chunks(chunks, query, embed_model):
    """
    Ranking relevance of each chunk to query, returns a list of most to least relevant chunk along
    with the relevance score
    """
    query_embedding = embed_model.encode(query)

    chunk_embeddings = embed_model.encode(chunks)

    similarities = cosine_similarity(
        [query_embedding],
        chunk_embeddings
    )[0]

    scored_chunks = list(zip(chunks, similarities))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    return scored_chunks

def rerank_context(excerpts, user_question, relevant_count, embed_model):
    """
    Performs reranking given excerpts, and returns a cleaned string with
    `relevant_count` sentences pulled from those excerpts.
    """

    if isinstance(excerpts, list):
        excerpts = "\n".join(excerpts)   # collapse list → one big string

    chunks = chunk_document(excerpts)    # now safe: string → sent_tokenize
    reranked = rerank_chunks(chunks, user_question, embed_model)
    context  = "\n".join(chunk for chunk, _ in reranked[:relevant_count])

    return context

def generate_response(client, model, user_question, relevant_excerpts, benchmarking):
    """
    Generate a grounded response using Groq + relevant text.
    """

    system_prompt = """
You are an expert assistant for interpreting scientific research on agroforestry and carbon sequestration.

Use the provided excerpts from research papers. If the answer is not present, say: "Answer not found in provided excerpts."

If possible, quote values directly and cite the source in parentheses using the paper title. Interpret tables if needed.

Assume the excerpts are from the same location mentioned in the question.

Be concise. Avoid extra context, interpretation, or commentary.


Here are examples of ideal answers:

Q: "What is the carbon impact of planting Leucaena + Napier in Central Kenya, Meru south?"  
A: "18.6 g/kg"

Q: "What is the carbon impact of planting cropland in Badessa?"  
A: "3.19 g.kg-1"

Q: "What is the carbon sequestration of Acacia Senegal in Mbeere?"  
A: "Answer not found in provided excerpts."

"""

    if benchmarking:
        system_prompt = "Always respond with **only** the **numerical value and unit**. No explanations or no extra text." + system_prompt

    print(colored(relevant_excerpts,"blue"))

    messages = []
    messages.append({"role": "system", "content": system_prompt})
    # open the questions folder and list all the files
    questions_dir = os.path.join(os.path.dirname(__file__), "questions")
    question_files = os.listdir(questions_dir)
    answer_dir = os.path.join(os.path.dirname(__file__), "answers")
    answer_files = os.listdir(answer_dir)
    question_files = [os.path.join(questions_dir, f) for f in question_files if f.endswith('.txt')]
    answer_files = [os.path.join(answer_dir, f) for f in answer_files if f.endswith('.txt')]
    for i in range(len(question_files)):
        # open the file
        with open(question_files[i]) as f:
            messages.append({"role": "user", "content": f.read()})
        with open(answer_files[i]) as f:
            messages.append({"role": "assistant", "content": f.read()})


    messages.append({"role": "user", "content": f"User Question: {user_question}\n\nRelevant Excerpts:\n\n{relevant_excerpts}"})

    print(colored("\n\nMessages:\n", "green"))
    print(messages)
    chat_completion = client.chat.completions.create(
        model=model,
        messages=messages
    )

    return chat_completion.choices[0].message.content


def main():
    model = "llama3-8b-8192"
    groq_api_key = os.getenv("GROQ_API_KEY")
    hugging_face_key = os.getenv("HUGGINGFACE_HUB_TOKEN")

    if not groq_api_key:
        raise EnvironmentError("Missing GROQ_API_KEY in environment.")

    # Initialize Groq + embedding model
    client = Groq(api_key=groq_api_key)
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load FAISS index
    faiss_index = FAISS.load_local(
        "faiss_index",
        embedding_function,
        allow_dangerous_deserialization=True
    )

    print("\n🌿 Agroforestry RAG System (Powered by FAISS + Groq)")
    print("Ask a question related to carbon stocks, sequestration, soil data, or agroforestry studies.\n")
    
    embed_model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1", token=hugging_face_key)

    while True:
        with open("metadata.json", "r") as f:
            metadata = json.load(f)
        for filename, meta in metadata.items():  
            meta["score"] = 0

        user_question = input(colored("🔎 Your question: ", "yellow")).strip()

        if not user_question:
            continue
        q_vec = embedding_function.embed_query(user_question) 
#path_to_articles = ""
        path_to_articles =  os.getenv("path_to_articles")
        excerpts = get_relevant_excerpts(user_question, q_vec, embedding_function, metadata, path_to_articles)
        context = rerank_context_per_article(excerpts, user_question,
                                     sentences_per_article=6,
                                     embed_model=embed_model)

        response = generate_response(client, model, user_question, context, benchmarking = False)
        print(colored("\n Answer:\n" + response + "\n", "magenta"))



if __name__ == "__main__":
    main()
