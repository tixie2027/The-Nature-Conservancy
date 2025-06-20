from dotenv import load_dotenv
load_dotenv()
import re
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
nltk.download("punkt")

hugging_face_key = os.getenv("HUGGINGFACE_HUB_TOKEN")

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

def chunk_document(text, max_tokens=400):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",token=hugging_face_key)
    chunks, current_chunk = [], []
    
    for sentence in sent_tokenize(text):
        tentative_chunk = " ".join(current_chunk + [sentence])
        token_ids = tokenizer.encode(tentative_chunk, truncation=False, add_special_tokens=True, max_length = max_tokens)
        
        if len(token_ids) > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
        else:
            current_chunk.append(sentence)

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

def flatten_article_text(article_json):
    """
    Extract only relevant textual content from nested JSON structure.
    """
    text_blocks = []

    def recursive_extract(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    recursive_extract(v)
                elif isinstance(v, str):
                    text_blocks.append(v)
        elif isinstance(obj, list):
            for item in obj:
                recursive_extract(item)

    recursive_extract(article_json)
    return "\n".join(text_blocks)

import re
from nltk.tokenize import sent_tokenize



def get_numerical_chunks_from_articles(excerpts):
    """
    Extract and return simple contextual numerical chunks from each article excerpt.
    """
    numeric_chunks = []
    number_pattern = re.compile(r'\d+(\.\d+)?')

    for art in excerpts:
        try:
            text = art["text"]  # Use as raw string, no JSON parsing
            
            for match in number_pattern.finditer(text):
                start = max(0, match.start() - 200)
                end = min(len(text), match.end() + 200)
                chunk = text[start:end].strip()
                numeric_chunks.append(f'{art["label"]}:\n{chunk}')
        
        except Exception as e:
            print(f"Failed to process article {art['label']}: {e}")
            continue

    print(colored(f"\n\nExtracted {len(numeric_chunks)} numeric chunks.\n", "green"))
    with open("numeric_chunks.txt", "w") as f:
        for chunk in numeric_chunks:
            f.write(chunk + "\n\n")
    return "\n\n".join(numeric_chunks)

def get_text_chunks_from_articles(client, model, excerpts, user_question, sentences_per_article, embedding_function):
    """
    Generate text chunks from articles based on the user question.
    """
    embedded_question = embedding_function.embed_query(user_question)
    context = []

    for i in range(len(excerpts)):
        article = excerpts[i]["text"] 
        prompt = f"""You are an expert assistant for interpreting scientific research on agroforestry and carbon sequestration. Check whether the following article contains information relevant to the question: {user_question}.
        If it does, return the first {sentences_per_article} sentences that are relevant to the question. If it does not, return "Answer not found in provided excerpts". Assume that the articles are talking about the same location as the question. ONLY return the sentences, no extra text, no explanation, no interpretation, no commentary."
        {article}
        """ 
        # cut references from prompt
        if "References" in prompt:
            prompt = prompt.split("References")[0]
        
        # cut to first 6000 characters
        if len(prompt) > 6000:
            print(colored(f"Prompt too long ({len(prompt)} characters), truncating to 6000 characters.", "red"))
            prompt = prompt[:6000]
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        print(colored("\n\nResponse:\n", "green"))
        print(response.choices[0].message.content)
        context.append(response.choices[0].message.content)

    return context

def get_chunks_from_articles(client, model,excerpts, user_question,sentences_per_article,embedding_function,type_question):
    if type_question == "num":
        print(colored("Numerical question detected, using numerical response generation.", "green"))
        return get_numerical_chunks_from_articles(excerpts)
    elif type_question == "text":
        print(colored("Text question detected, using text response generation.", "green"))
        return get_text_chunks_from_articles(client, model, excerpts, user_question, sentences_per_article, embedding_function)
    else:
        print(colored("Invalid type_question, defaulting to text response generation.", "red"))
        type_question = "text"
        get_text_chunks_from_articles(client, model, excerpts, user_question, sentences_per_article, embedding_function)
   


def main():
    model = "llama-3.1-8b-instant"
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise EnvironmentError("Missing GROQ_API_KEY in environment.")

    # Initialize Groq + embedding model
    client = Groq(api_key=groq_api_key)
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load FAISS index
    faiss_index = FAISS.load_local(
        "faiss_index",
        embedding_function,
        allow_dangerous_deserialization=True
    )

    print("\n🌿 Agroforestry RAG System (Powered by FAISS + Groq)")
    print("Ask a question related to carbon stocks, sequestration, soil data, or agroforestry studies.\n")
    
    embed_model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")

    while True:
        with open("metadata.json", "r") as f:
            metadata = json.load(f)
        for filename, meta in metadata.items():  
            meta["score"] = 0
        type_question = input(colored("Type 'num' if your question needs a numerical answer or 'text' if it needs a text answer: ", "yellow")).strip().lower()
        user_question = input(colored("🔎 Your question: ", "yellow")).strip()

        if not user_question:
            continue
        q_vec = embedding_function.embed_query(user_question) 
        path_to_articles =  os.getenv("path_to_articles")
        # get the correct articles
        excerpts = get_relevant_excerpts(user_question, q_vec, embedding_function, metadata, path_to_articles)
        # get the right chunks from the articles
        context = get_chunks_from_articles(client, model,excerpts, user_question, sentences_per_article=6, embedding_function=embedding_function,type_question=type_question)
        # context = rerank_context_per_article(excerpts, user_question,
        #                              sentences_per_article=6,
        #                              embed_model=embed_model)

        # response = generate_response(client, model, user_question, context, benchmarking = False)
        # print(colored("\n Answer:\n" + response + "\n", "magenta"))



if __name__ == "__main__":
    main()
