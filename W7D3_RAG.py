
from pathlib import Path
import chromadb
from groq import Groq

# ==========================================
# 1. PROJECT SETUP
# ==========================================

BASE_DIR = Path(__file__).parent
RESULTS_FILE = BASE_DIR / "w7d3_rag_results.txt"

print("\n========== PROJECT SETUP ==========")

# Connect to ChromaDB
client = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db")
)

print("ChromaDB connected successfully!")

# ==========================================
# 2. LOAD VECTOR DATABASE
# ==========================================

collection = client.get_collection(
    name="w7d2_large"
)

print("Collection loaded successfully!")
print("Total chunks:", collection.count())

# ==========================================
# 3. CONNECT TO GROQ
# ==========================================

groq_client = Groq()

print("Groq connected successfully!")

# ==========================================
# 4. RETRIEVAL FUNCTION
# ==========================================

def retrieve_chunks(question):

    results = collection.query(
        query_texts=[question],
        n_results=min(3, collection.count())
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print("\n========== RETRIEVAL RESULTS ==========")

    for i, document in enumerate(documents):

        print(f"\nResult {i + 1}:")
        print("Document:", document)
        print("Distance:", distances[i])
        print("Source:", metadatas[i]["source"])
        print("Page:", metadatas[i]["page"])

    return documents, metadatas, distances

# ==========================================
# 5. RAG ANSWER FUNCTION
# ==========================================

def answer_with_rag(question):

    # Step 1: Retrieve relevant chunks

    documents, metadatas, distances = retrieve_chunks(question)

    # Step 2: Build context

    context = ""

    for i, document in enumerate(documents):

        context += f"""
Source: {metadatas[i]["source"]}
Page: {metadatas[i]["page"]}
Content: {document}

"""

    # Step 3: System prompt

    system_prompt = """
You are a document-based question answering assistant.

Answer the user's question using ONLY the provided context.

Rules:

1. Use only information found in the context.
2. Do not use outside knowledge.
3. Do not invent or assume information.
4. If the answer is not found in the context,
   reply exactly: مش موجود في المستند
5. If the context contains conflicting information,
   clearly mention the conflict.
6. Keep your answer clear and concise.
"""

    # Step 4: Send context and question to Groq

    response = groq_client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{question}
"""
            }
        ],

        temperature=0
    )

    # Step 5: Extract answer

    answer = response.choices[0].message.content

    # Step 6: Display answer

    print("\n========== RAG ANSWER ==========")

    print("Question:", question)
    print("\nAnswer:", answer)

    # Step 7: Display sources

    print("\n========== SOURCES ==========")

    for metadata in metadatas:

        print(
            "Source:", metadata["source"],
            "| Page:", metadata["page"]
        )

    return answer, metadatas

# ==========================================
# 6. DAY 2 TEST QUESTIONS
# ==========================================

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

# ==========================================
# 7. RUN ALL 10 RAG TESTS
# ==========================================

print("\n========== STARTING RAG TESTS ==========")

with open(RESULTS_FILE, "w", encoding="utf-8") as file:

    file.write("PHANTOMS AI - WEEK 7\n")
    file.write("DAY 3 - RAG TEST RESULTS\n")
    file.write("=" * 60 + "\n")

    for index, question in enumerate(questions, start=1):

        print("\n\n")
        print("=" * 60)
        print(f"QUESTION {index}/10")
        print("=" * 60)

        try:

            answer, metadatas = answer_with_rag(question)

            # Save question

            file.write(f"\nQUESTION {index}: {question}\n")

            # Save answer

            file.write(f"ANSWER: {answer}\n")

            # Save sources

            file.write("SOURCES:\n")

            for metadata in metadatas:

                file.write(
                    f"- Source: {metadata['source']} "
                    f"| Page: {metadata['page']}\n"
                )

            file.write("-" * 60 + "\n")

            file.flush()

        except Exception as error:

            print("\nERROR:", error)

            file.write(f"\nQUESTION {index}: {question}\n")
            file.write(f"ERROR: {error}\n")
            file.write("-" * 60 + "\n")

            file.flush()

# ==========================================
# 8. FINISH
# ==========================================

print("\n========== RAG TESTING FINISHED ==========")

print("All questions have been processed.")

print("Total questions:", len(questions))

print("Results saved to:")
print(RESULTS_FILE)