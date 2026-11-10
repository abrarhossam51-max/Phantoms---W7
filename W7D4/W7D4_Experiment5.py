from pathlib import Path
import chromadb

# =====================================
# 1. CONNECT TO CHROMADB
# =====================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

client = chromadb.PersistentClient(
    path=str(PROJECT_DIR / "chroma_db")
)

collection = client.get_collection("w7d2_large")

print("ChromaDB connected successfully!")
print("Collection loaded successfully!")
print("Total chunks:", collection.count())


# =====================================
# 2. QUESTION
# =====================================

question = "What does a cell in a database table contain?"


# =====================================
# 3. EXPERIMENT A
# WITHOUT METADATA FILTERING
# =====================================

print("\n================================")
print("A - WITHOUT METADATA FILTERING")
print("================================")

results_a = collection.query(
    query_texts=[question],
    n_results=3
)

for i, document in enumerate(results_a["documents"][0]):

    print(f"\nResult {i+1}:")
    print("Document:", document[:400])
    print("Metadata:", results_a["metadatas"][0][i])
    print("Distance:", results_a["distances"][0][i])


# =====================================
# 4. EXPERIMENT B
# WITH METADATA FILTERING
# =====================================

print("\n================================")
print("B - WITH METADATA FILTERING")
print("================================")

results_b = collection.query(
    query_texts=[question],
    n_results=3,
    where={"page": 1}
)

for i, document in enumerate(results_b["documents"][0]):

    print(f"\nResult {i+1}:")
    print("Document:", document[:400])
    print("Metadata:", results_b["metadatas"][0][i])
    print("Distance:", results_b["distances"][0][i])


# =====================================
# 5. SUMMARY
# =====================================

print("\n================================")
print("EXPERIMENT 5 SUMMARY")
print("================================")

print("Without filtering:", len(results_a["ids"][0]), "results")
print("With filtering:", len(results_b["ids"][0]), "results")

print("\nExperiment 5 completed!")