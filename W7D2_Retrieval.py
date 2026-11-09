from pathlib import Path
import chromadb

# 1. CONNECT TO CHROMADB

BASE_DIR = Path(__file__).parent

client = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db")
)

print("ChromaDB connected successfully!")

# 2. QUESTIONS

questions = [
    "What does a cell in a database table contain?",
    "Which SQL data type stores exact decimal values such as money?",
    "Which tier reads and writes data from databases?",
    "What is the purpose of a Primary Key?",
    "Which normal form eliminates transitive dependencies?",
    "Why can't StudentID alone be a Primary Key when a student enrolls in multiple subjects?",
    "How do 2NF and 3NF differ in handling dependencies?",
    "How do the three tiers of a 3-tier architecture work together?",
    "What is the difference between SQL and NoSQL databases in scalability?",
    "How does database replication work across multiple servers?"
]

# 3. COLLECTIONS

collection_names = [
    "w7d2_small",
    "w7d2_medium",
    "w7d2_large"
]

# 4. SAVE FULL RESULTS

output_file = BASE_DIR / "retrieval_results.txt"

with open(output_file, "w", encoding="utf-8") as file:

    for collection_name in collection_names:

        collection = client.get_collection(name=collection_name)

        file.write("\n" + "=" * 60 + "\n")
        file.write(f"COLLECTION: {collection_name}\n")
        file.write("=" * 60 + "\n")

        print(f"\nProcessing {collection_name}...")

        for i, question in enumerate(questions, start=1):

            results = collection.query(
                query_texts=[question],
                n_results=3
            )

            file.write(f"\nQ{i}: {question}\n")

            for rank in range(3):

                document = results["documents"][0][rank]
                distance = results["distances"][0][rank]
                metadata = results["metadatas"][0][rank]

                file.write(f"\nResult {rank + 1}:\n")
                file.write(f"Document: {document}\n")
                file.write(f"Distance: {distance}\n")
                file.write(f"Source: {metadata.get('source', 'Unknown')}\n")
                file.write(f"Page: {metadata.get('page', 'Unknown')}\n")

            # Compact terminal output
            top_document = results["documents"][0][0]
            top_distance = results["distances"][0][0]
            top_metadata = results["metadatas"][0][0]

            print(
                f"Q{i}: Top result | "
                f"Distance: {top_distance:.4f} | "
                f"Page: {top_metadata.get('page', 'Unknown')}"
            )

print("\n" + "=" * 40)
print("Retrieval Evaluation Completed!")
print(f"Full results saved to: {output_file}")
print("=" * 40)