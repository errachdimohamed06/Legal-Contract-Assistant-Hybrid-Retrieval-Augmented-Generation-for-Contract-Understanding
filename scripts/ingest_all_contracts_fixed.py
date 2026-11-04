import sys
import yaml
import shutil
from pathlib import Path
from tqdm import tqdm

sys.path.append(".")
from src.core.chunker import chunk_contract

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

with open("configs/app_config.yaml", "r") as f:
    app_config = yaml.safe_load(f)

with open("configs/model_config.yaml", "r") as f:
    model_config = yaml.safe_load(f)

contracts_folder = Path(app_config["paths"]["data_dir"]).resolve()
chroma_folder = Path(app_config["paths"]["chroma_persist_dir"]).resolve()

if chroma_folder.exists():
    print(f"Suppression de l'ancienne base ChromaDB...")
    shutil.rmtree(chroma_folder)

chroma_folder.mkdir(parents=True, exist_ok=True)

client = chromadb.PersistentClient(path=str(chroma_folder))

embedding_model = HuggingFaceEmbeddings(
    model_name=model_config["embedding"]["model_name"],
    model_kwargs={"device": model_config["embedding"]["device"]},
    encode_kwargs={"normalize_embeddings": model_config["embedding"]["normalize_embeddings"]}
)

vector_store = Chroma(
    client=client,
    collection_name=model_config["vector_db"]["collection_name"],
    embedding_function=embedding_model
)

txt_files = list(contracts_folder.glob("*.txt"))
print(f"Fichiers trouvés : {len(txt_files)}")

all_docs = []
error_files = []

for file_path in tqdm(txt_files, desc="Chunking des contrats"):
    try:
        with open(str(file_path), "r", encoding="utf-8") as f:
            raw_text = f.read()
        chunks = chunk_contract(raw_text)
        for header, body in chunks:
            doc = Document(
                page_content=body,
                metadata={
                    "source": file_path.name,
                    "header": header
                }
            )
            all_docs.append(doc)
    except Exception as e:
        error_files.append((file_path.name, str(e)))
        continue

print(f"Nombre total de chunks créés : {len(all_docs)}")

if error_files:
    print(f"\nFichiers ignorés : {len(error_files)}")
    for name, err in error_files[:10]:
        print(f"  - {name} : {err[:50]}...")

print("Ajout à ChromaDB par lots de 1000...")
batch_size = 1000
total_batches = (len(all_docs) + batch_size - 1) // batch_size

for i in tqdm(range(0, len(all_docs), batch_size), desc="Envoi des lots", total=total_batches):
    batch = all_docs[i:i + batch_size]
    vector_store.add_documents(batch)

print("Ingestion terminée avec succès !")
print(f"Toutes les données sont sauvegardées dans : {chroma_folder}")