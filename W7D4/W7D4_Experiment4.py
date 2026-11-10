from pathlib import Path
import os
import chromadb
from groq import Groq

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


# =====================================
# 2. CONNECT TO GROQ
# =====================================

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing!")

groq_client = Groq(api_key=api_key)

MODEL = "openai/gpt-oss-120b"

print("Groq connected successfully!")


# =====================================
# 3. RETRIEVE CONTEXT
# =====================================

question = "What does a cell in a database table contain?"

results = collection.query(
    query_texts=[question],
    n_results=3
)

documents = results["documents"][0]

context = "\n\n".join(documents)

print("\n========== RETRIEVED CONTEXT ==========")
print(context)


# =====================================
# 4. DEFINE THREE PROMPTS
# =====================================

prompts = {

    "A - Without Restrictions": """
You are a helpful assistant.
Answer the user's question.
""",

    "B - Context Only": """
You are a database assistant.
Answer the question using ONLY the provided context.
Do not use outside information.
""",

    "C - Context Only + Abstention": """
You are a database assistant.
Answer the question using ONLY the provided context.
Do not use outside information.
If the answer is not available in the context,
say "I don't know".
Never guess or invent information.
"""
}


# =====================================
# 5. RUN EXPERIMENTS
# =====================================

for experiment_name, system_prompt in prompts.items():

    print("\n" + "=" * 50)
    print(experiment_name)
    print("=" * 50)

    response = groq_client.chat.completions.create(
        model=MODEL,

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

    answer = response.choices[0].message.content

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(answer)


print("\nExperiment 4 completed!")