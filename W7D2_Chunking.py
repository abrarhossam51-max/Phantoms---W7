from pathlib import Path
from pypdf import PdfReader
import chromadb

# =====================================
# 1. FIND PROJECT FOLDER AND PDF
# =====================================

BASE_DIR = Path(__file__).parent

pdf_files = list(BASE_DIR.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError("No PDF files found!")

pdf_path = pdf_files[0]

print("Selected PDF:", pdf_path.name)

# =====================================
# 2. READ PDF
# =====================================

reader = PdfReader(str(pdf_path))

print("\nPDF loaded successfully!")
print("Total pages:", len(reader.pages))

all_pages = []

for page_number, page in enumerate(reader.pages, start=1):

    text = page.extract_text() or ""

    if text.strip():

        all_pages.append({
            "page": page_number,
            "text": text,
            "source": pdf_path.name
        })

    print(f"Page {page_number}: {len(text)} characters")

print("\nTotal extracted pages:", len(all_pages))

# First page preview
print("\n========== FIRST PAGE PREVIEW ==========")

print(all_pages[0]["text"][:1000])

# =====================================
# 3. CHUNKING FUNCTION
# =====================================

def create_chunks(pages, chunk_size, overlap):

    chunks = []

    for page in pages:

        text = page["text"]
        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": page["source"],
                    "page": page["page"]
                }
            })

            start += chunk_size - overlap

    return chunks

# =====================================
# 4. CHUNKING EXPERIMENTS
# =====================================

chunk_settings = {

    "Small": {
        "chunk_size": 150,
        "overlap": 30
    },

    "Medium": {
        "chunk_size": 400,
        "overlap": 80
    },

    "Large": {
        "chunk_size": 900,
        "overlap": 180
    }

}

# =====================================
# 5. DISPLAY CHUNKING RESULTS
# =====================================

for name, settings in chunk_settings.items():

    chunks = create_chunks(
        all_pages,
        settings["chunk_size"],
        settings["overlap"]
    )

    print(f"\n========== {name} CHUNKS ==========")

    print("Chunk Size:", settings["chunk_size"])
    print("Overlap:", settings["overlap"])
    print("Total Chunks:", len(chunks))

    print("\nFirst Chunk Preview:")
    print(chunks[0]["text"])

    print("\nFirst Chunk Metadata:")
    print(chunks[0]["metadata"])

# =====================================
# 6. CONNECT TO CHROMADB
# =====================================

client = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db")
)

print("\nChromaDB connected successfully!")

# =====================================
# 7. CREATE THREE COLLECTIONS
# =====================================

for name, settings in chunk_settings.items():

    # Create chunks
    chunks = create_chunks(
        all_pages,
        settings["chunk_size"],
        settings["overlap"]
    )

    # Collection name
    collection_name = f"w7d2_{name.lower()}"

    # Check existing collections
    existing_collections = [
        collection.name
        for collection in client.list_collections()
    ]

    # Delete old collection if it exists
    if collection_name in existing_collections:

        client.delete_collection(
            name=collection_name
        )

        print(f"\nOld collection deleted: {collection_name}")

    # Create collection
    collection = client.create_collection(
        name=collection_name
    )

    # Store chunks
    collection.add(

        documents=[
            chunk["text"]
            for chunk in chunks
        ],

        metadatas=[
            chunk["metadata"]
            for chunk in chunks
        ],

        ids=[
            f"{name.lower()}_{i}"
            for i in range(len(chunks))
        ]

    )

    print(f"\n{name} Collection Created Successfully!")

    print("Collection Name:", collection_name)
    print("Stored Chunks:", collection.count())

# =====================================
# 8. FINAL VERIFICATION
# =====================================

print("\n========== FINAL COLLECTIONS ==========")

for collection in client.list_collections():

    if collection.name.startswith("w7d2_"):

        print(
            collection.name,
            "=>",
            collection.count(),
            "chunks"
        )

print("\nAll Collections Stored Successfully!")