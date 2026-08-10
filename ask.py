import chromadb
import ollama
from sentence_transformers import SentenceTransformer

# =========================
# INIT
# =========================
DB_PATH = r"C:\Users\Bouchra\Desktop\Self_Guided_Microsoft\RAG_Project\chroma_db"
COLLECTION_NAME = "company_policy_documents"

OLLAMA_MODEL = "llama3"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(COLLECTION_NAME)

embedder = SentenceTransformer(EMBEDDING_MODEL)

# =========================
# RETRIEVAL
# =========================
def retrieve(query, k=15, threshold=0.30):
    query_vec = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    if not docs:
        return []

    retrieved = []

    for doc, meta, distance in zip(docs, metas, distances):
        similarity = 1 / (1 + distance)

        if similarity >= threshold:
            retrieved.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", "unknown"),
                "similarity": round(similarity, 3)
            })

    return retrieved


# =========================
# BASELINE (OLLAMA ONLY)
# =========================
def baseline_llm(question):
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response["message"]["content"].strip()


# =========================
# RAG (OLLAMA + COMPANY CORPUS)
# =========================
def rag_llm(question):
    retrieved_docs = retrieve(question)

    if not retrieved_docs:
        return "Not found in the company documents."

    context_parts = []

    for item in retrieved_docs:
        context_parts.append(
            f"Source: {item['source']} | Page: {item['page']}\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an enterprise policy question-answering assistant.

Answer the question using ONLY the company documents provided in the context.

Rules:
- Do not use external knowledge.
- Do not guess.
- Do not mention chunks.
- If the answer is not clearly found in the company documents, answer exactly:
  Not found in the company documents.
- Give a clear and concise answer.
- When possible, mention the document source and page number.

Company document context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"].strip()


# =========================
# MAIN LOOP
# =========================
while True:
    q = input("\nAsk a company-policy question (or 'exit'): ").strip()

    if q.lower() == "exit":
        break

    if not q:
        continue

    print("\n====================")
    print("BASELINE (OLLAMA)")
    print("====================")
    print(baseline_llm(q))

    print("\n====================")
    print("RAG (OLLAMA + COMPANY DOCUMENTS)")
    print("====================")
    print(rag_llm(q))