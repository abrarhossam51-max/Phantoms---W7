from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. CONNECT TO CHROMADB
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

client = chromadb.PersistentClient(
    path=str(PROJECT_DIR / "chroma_db")
)

collection = client.get_collection(
    name="w7d2_large"
)

print("ChromaDB connected successfully!")
print("Collection loaded successfully!")
print("Total chunks:", collection.count())


# ==========================================
# 2. LOAD DIFFERENT EMBEDDING MODEL
# ==========================================

print("\nLoading a different embedding model...")

wrong_model = SentenceTransformer(
    "paraphrase-MiniLM-L3-v2"
)

print("Different embedding model loaded!")


# ==========================================
# 3. TEST QUESTION
# ==========================================

question = "What does a cell in a database table contain?"

print("\n========== ORIGINAL MODEL ==========")

original_results = collection.query(
    query_texts=[question],
    n_results=3
)

for i, document in enumerate(original_results["documents"][0]):
    print(f"\nResult {i + 1}:")
    print("Document:", document)
    print("Distance:", original_results["distances"][0][i])


# ==========================================
# 4. TEST DIFFERENT EMBEDDING MODEL
# ==========================================

print("\n========== DIFFERENT EMBEDDING MODEL ==========")

wrong_embedding = wrong_model.encode(question).tolist()

wrong_results = collection.query(
    query_embeddings=[wrong_embedding],
    n_results=3
)

for i, document in enumerate(wrong_results["documents"][0]):
    print(f"\nResult {i + 1}:")
    print("Document:", document)
    print("Distance:", wrong_results["distances"][0][i])


# ==========================================
# 5. EXPERIMENT CONCLUSION
# ==========================================

print("\n========== EXPERIMENT 1 SUMMARY ==========")

print("Original model: ChromaDB default embedding model")
print("Different model: paraphrase-MiniLM-L3-v2")

print("\nCompare the retrieved documents and distances.")
print("Observe how changing the embedding model affects retrieval.")

print("\nExperiment 1 completed!")