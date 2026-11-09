import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="phantoms_week7"
)

print("ChromaDB connected successfully!")
print("Collection created successfully!")

documents = [
    "Python lists are mutable sequences that can store items of different types.",
    "Git branches let developers work on features without affecting the main code.",
    "A REST API exposes resources through HTTP methods like GET and POST.",
    "SQL JOINs combine rows from two tables based on a related column.",
    "Regular walking for thirty minutes a day improves heart health.",
    "Drinking enough water helps maintain energy and concentration.",
    "Sleep of seven to eight hours supports muscle recovery and memory.",
    "Stretching before exercise can reduce the risk of muscle injuries.",
    "To make rice fluffy, rinse it first and let it steam after cooking.",
    "Letting dough rest allows the gluten to relax and makes bread softer.",
    "Searing meat at high heat creates a flavorful brown crust.",
    "Adding salt to pasta water seasons the noodles from the inside."
]

ids = [f"doc_{i}" for i in range(1, 13)]

metadatas = [
    {"topic": "programming"},
    {"topic": "programming"},
    {"topic": "programming"},
    {"topic": "programming"},
    {"topic": "fitness"},
    {"topic": "fitness"},
    {"topic": "fitness"},
    {"topic": "fitness"},
    {"topic": "cooking"},
    {"topic": "cooking"},
    {"topic": "cooking"},
    {"topic": "cooking"}
]

collection.upsert(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)

print("\n12 Documents added successfully!")
print("Total documents:", collection.count())

queries = [
    "How do I store different kinds of values together in Python?",
    "How can I avoid hurting myself when working out?",
    "Why is my bread dense and hard?",
    "What is the best way to merge data from two database tables?",
    "How do I deploy a rocket to Mars?"
]

print("\n========== SEARCH RESULTS ==========")

for i, query in enumerate(queries, 1):

    print(f"\nQ{i}: {query}")

    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    for j in range(3):
        print(f"\nResult {j + 1}:")
        print("Document:", results["documents"][0][j])
        print("Distance:", results["distances"][0][j])

print("\n========== Q2 WITHOUT FILTER ==========")

q2 = queries[1]

results = collection.query(
    query_texts=[q2],
    n_results=3
)

for i in range(3):
    print("\nDocument:", results["documents"][0][i])
    print("Distance:", results["distances"][0][i])

print("\n========== Q2 WITH FITNESS FILTER ==========")

filtered_results = collection.query(
    query_texts=[q2],
    n_results=3,
    where={"topic": "fitness"}
)

for i in range(len(filtered_results["documents"][0])):
    print("\nDocument:", filtered_results["documents"][0][i])
    print("Distance:", filtered_results["distances"][0][i])

print("\n========== FINAL COUNT ==========")
print("Total documents:", collection.count())
print("Program completed successfully!")