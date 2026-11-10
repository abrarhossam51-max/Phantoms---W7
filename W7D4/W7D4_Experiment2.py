from pathlib import Path
import chromadb


# ==========================================
# 1. CONNECT TO CHROMADB
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

client = chromadb.PersistentClient(
    path=str(PROJECT_DIR / "chroma_db")
)
print("ChromaDB connected successfully!")


# ==========================================
# 2. COLLECTIONS
# ==========================================

collections = {
    "SMALL": "w7d2_small",
    "MEDIUM": "w7d2_medium",
    "LARGE": "w7d2_large"
}


# ==========================================
# 3. TEST QUESTION
# ==========================================

question = "What does a cell in a database table contain?"

print("\nQuestion:", question)


# ==========================================
# 4. TEST DIFFERENT CHUNK SIZES
# ==========================================

for chunk_type, collection_name in collections.items():

    print("\n================================")
    print(f"CHUNK SIZE: {chunk_type}")
    print("================================")

    collection = client.get_collection(
        name=collection_name
    )

    print("Total chunks:", collection.count())

    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    for i, document in enumerate(results["documents"][0]):

        print(f"\nResult {i + 1}:")
        print("Document:", document)
        print("Distance:", results["distances"][0][i])


# ==========================================
# 5. EXPERIMENT SUMMARY
# ==========================================

print("\n================================")
print("EXPERIMENT 2 SUMMARY")
print("================================")

print("Compared Small, Medium, and Large chunks.")

print("Observe:")
print("1. Which chunk retrieves the correct information?")
print("2. Which chunk provides more relevant context?")
print("3. How does chunk size affect retrieval quality?")

print("\nExperiment 2 completed!")