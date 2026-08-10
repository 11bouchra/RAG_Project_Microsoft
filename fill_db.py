import os
import shutil
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================
# PATHS
# =========================
PROJECT_PATH = r"C:\Users\Bouchra\Desktop\Self_Guided_Microsoft\RAG_Project"
DATA_PATH = os.path.join(PROJECT_PATH, "Corpus")
CHROMA_PATH = os.path.join(PROJECT_PATH, "chroma_db")

COLLECTION_NAME = "company_policy_documents"

# =========================
# CLEAN OLD DATABASE
# =========================
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

# =========================
# LOAD PDF FILES
# =========================
print("Loading PDF corpus...")

loader = PyPDFDirectoryLoader(DATA_PATH)
documents = loader.load()

print("Pages loaded:", len(documents))

# =========================
# CHUNKING
# =========================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)

print("Chunks created:", len(chunks))

# =========================
# EMBEDDING MODEL
# =========================
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

texts = [chunk.page_content for chunk in chunks]

metas = []
for i, chunk in enumerate(chunks):
    source = os.path.basename(chunk.metadata.get("source", "unknown"))
    page = chunk.metadata.get("page", "unknown")

    metas.append({
        "source": source,
        "page": page,
        "chunk_id": i
    })

print("Creating embeddings...")
embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

ids = [f"chunk_{i}" for i in range(len(texts))]

# =========================
# STORE IN CHROMA
# =========================
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

collection.upsert(
    documents=texts,
    ids=ids,
    embeddings=embeddings,
    metadatas=metas
)

print("\nRAG DATABASE READY ✔")
print("Collection:", COLLECTION_NAME)
print("PDF pages:", len(documents))
print("Chunks:", len(chunks))