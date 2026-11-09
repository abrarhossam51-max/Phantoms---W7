import chromadb
from pathlib import Path

BASE_DIR = Path(__file__).parent

client = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db")
)

for collection in client.list_collections():
    print(collection.name)