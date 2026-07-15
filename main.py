import os
from sentence_transformers import SentenceTransformer
from groq import Groq

def cosine_similarity(a, b):
    """Computes the cosine similarity between two vectors."""
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
LANGUAGE_MODEL = 'llama3-8b-8192'
DATASET_PATH = 'physics-facts.txt'

VECTOR_DB = []
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Setup Groq Client
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("Error: Missing GROQ_API_KEY environment variable. Please set it before running.")
    exit(1)

client = Groq(api_key=api_key)

def load_and_embed():
    """Loads the physics facts dataset and computes embeddings for each chunk."""
    try:
        with open(DATASET_PATH, 'r', encoding="utf-8") as file:
            dataset = file.readlines()
            
        print(f"Loading {len(dataset)} facts from {DATASET_PATH}...")
        for i, chunk in enumerate(dataset):
            chunk = chunk.strip()
            if not chunk:
                continue
            
            print(f"Generating embedding for chunk {i+1}/{len(dataset)}...")
            embedding = embedder.encode(chunk).tolist()
            
            VECTOR_DB.append((chunk, embedding))
        print("Vector Database initialization complete!")
    except FileNotFoundError:
        print(f"Error: Could not find {DATASET_PATH}. Please make sure it exists.")
        exit(1)

def retrieve(query, top_k=3):
    """Finds the top_k most relevant chunks for a given query."""
    query_embedding = embedder.encode(query).tolist()
    
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
        
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def main():
    load_and_embed()
    
    print("\n" + "="*50)
    print("Welcome to the Particle Physics RAG Assistant!")
    print("Type your question below. Type 'exit' or 'quit' to close.")
    print("="*50 + "\n")
    
    while True:
        prompt = input("Ask a question: ")
        if prompt.lower() in ['exit', 'quit']:
            break
            
        results = retrieve(prompt)
        print("\n--- Retrieved Facts ---")
        for chunk, score in results:
            print(f"[{score:.3f}] {chunk}")
            
        context = '\n'.join([chunk for chunk, score in results])
        
        system_prompt = f"""You are a helpful assistant that answers questions about particle physics using ONLY the context below.
If the context does not contain the answer, say "I don't have enough information to answer that."

Context:
{context}"""
        
        print("\n--- Generating Answer ---")
        try:
            response = client.chat.completions.create(
                model=LANGUAGE_MODEL,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt},
                ]
            )
            print(f"AI: {response.choices[0].message.content}\n")
        except Exception as e:
            print(f"Error generating response: {e}\n")

if __name__ == "__main__":
    main()
