from dotenv import load_dotenv
load_dotenv()
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

def get_relevant_excerpts(user_question, docsearch):
    """
    Perform similarity search and return top 3 excerpts with titles.
    """
    relevant_docs = docsearch.similarity_search(user_question, k=3)
    excerpts = []

    for doc in relevant_docs:
        title = doc.metadata.get("title", "Unknown Source")
        excerpt = f"📄 **{title}**\n{doc.page_content.strip()[:2000]}..."  # Truncate for safety
        excerpts.append(excerpt)

    return "\n\n---\n\n".join(excerpts)

def chunk_document(text):
    """
    Chunk documents by using basic tokenizer
    """
    sentences = sent_tokenize(text)
    return sentences

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
    Performs reranking given excerpts, and returns a cleaned string with relevant_count
    number of sentences pulled from excerpts
    """
    chunks = chunk_document(excerpts)

    reranked = rerank_chunks(chunks, user_question, embed_model)

    context = "\n".join([chunk for chunk, _ in reranked[:relevant_count]])

    return context

def generate_response(client, model, user_question, relevant_excerpts):
    """
    Generate a grounded response using Groq + relevant text.
    """
    system_prompt = """
You are an expert in agroforestry and carbon sequestration. Use the excerpts from peer-reviewed scientific papers to answer the user's question. 
Quote directly when possible, and always cite the paper title in parentheses after the quote.
"""

    chat_completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User Question: {user_question}\n\nRelevant Excerpts:\n\n{relevant_excerpts}"}
        ]
    )

    return chat_completion.choices[0].message.content


def main():
    model = "llama3-8b-8192"
    groq_api_key = os.getenv("GROQ_API_KEY")

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

    while True:
        user_question = input("🔎 Your question: ").strip()
        if not user_question:
            continue

        excerpts = get_relevant_excerpts(user_question, faiss_index)

        embed_model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
        context = rerank_context(excerpts, user_question, 20, embed_model) # pulls the 20 most relevant sentences

        response = generate_response(client, model, user_question, context)
        print("\n🧠 Answer:\n" + response + "\n")



if __name__ == "__main__":
    main()
