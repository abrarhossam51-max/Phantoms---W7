from pathlib import Path
import chromadb

# 1. CONNECT TO CHROMADB

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

client = chromadb.PersistentClient(
    path=str(PROJECT_DIR / "chroma_db")
)

print("ChromaDB connected successfully!")

# 2. LOAD COLLECTION

collection = client.get_collection(
    name="w7d2_large"
)

print("Collection loaded successfully!")
print("Total chunks:", collection.count())

# 3. TEST QUESTION

question = "What does a cell in a database table contain?"

print("\nQuestion:", question)


# 4. TEST DIFFERENT TOP-K VALUES

top_k_values = [1, 3, 5]

for k in top_k_values:

    print("\n================================")
    print(f"TOP-K = {k}")
    print("================================")

    results = collection.query(
        query_texts=[question],
        n_results=k
    )

    for i, document in enumerate(results["documents"][0]):

        print(f"\nResult {i + 1}:")
        print("Document:", document)
        print("Distance:", results["distances"][0][i])

# 5. EXPERIMENT SUMMARY

print("\n================================")
print("EXPERIMENT 3 SUMMARY")
print("================================")

print("Compared Top-1, Top-3, and Top-5.")

print("\nObserve:")
print("1. Which Top-K retrieves relevant information?")
print("2. Does increasing Top-K improve context?")
print("3. Does retrieving more chunks add irrelevant information?")

print("\nExperiment 3 completed!")