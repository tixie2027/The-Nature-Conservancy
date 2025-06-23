import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from termcolor import colored
from main import get_relevant_excerpts, rerank_context_per_article, generate_response
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
from groq import Groq
import random

def similarity_score(model_response, answer, model, client):
    if model_response == "Answer not found in provided excerpts.":
        return 0
    
    similarity = client.chat.completions.create(
        model = model,
        messages = [
            {
                "role": "system", "content" : f"""Given this value: {model_response} and this value: {answer}. Give
                a score to how similar they are on a range of 0.00 - 1.00. If these two values are in different units, convert to the same unit first
                then compare. ONLY output the similarity score and no other text"""
            }
        ]
    )
    similarity = similarity.choices[0].message.content
    print(similarity, type(similarity))
    try:
        similarity = float(similarity)
        return similarity
    except:
        print("not successfukl")
        return 0



load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

# Models
embed_model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# FAISS index
faiss_index = FAISS.load_local(
    "faiss_index",
    embedding_function,
    allow_dangerous_deserialization=True
)


with open("benchmarking.json", "r") as f:
    qa_data = json.load(f)

all_questions = list(qa_data .items())

# sample 1/200 of the questions
sampled_questions = random.sample(all_questions, max(1, len(all_questions) // 200))

path_to_articles =  os.getenv("path_to_articles")

results = []

for question, answer_parts in sampled_questions:
    numeric_value, unit = answer_parts[1], answer_parts[2]
    true_answer = f"{numeric_value} {unit}"

    # embed question
    q_vec = embedding_function.embed_query(question)
    with open("metadata.json", "r") as f:
        metadata = json.load(f)
    for meta in metadata.values():
        meta["score"] = 0

    # getting context + generation
    excerpts = get_relevant_excerpts(question, q_vec, embedding_function, metadata, path_to_articles)
    context = rerank_context_per_article(excerpts, question,
                                         sentences_per_article=6,
                                         embed_model=embed_model)
    
    if len(context) > 6000: 
        context = context[:6000]

    model_answer = generate_response(client, "llama3-8b-8192", question, context, benchmarking = True)

    # response/answer similarity comparison
    similarity = similarity_score(model_answer, true_answer, "llama3-8b-8192", client)
    

    results.append({
        "question": question,
        "true_answer": true_answer,
        "model_answer": str(model_answer),
        "similarity": float(similarity)
    })

    print(colored(f"\nQ: {question}", "yellow"))
    print(colored(f" Answer: {true_answer}", "green"))
    print(colored(f"Model: {model_answer}", "cyan"))
    print(colored(f"Similarity: {similarity:.3f}", "magenta"))


with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

avg_score = np.mean([r["similarity"] for r in results])
print(colored(f"\nAverage Embedding Similarity: {avg_score:.3f}", "green"))
