import os
import pickle
import faiss
import numpy as np
from nova.vector.embed import embed_text


def split_into_chunks(text, max_chars=500, overlap=100):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += max_chars - overlap
    return chunks


def ingest_documents(folder_path: str, save_path: str = "data/vector_dbs/tennis_index"):
    texts = []
    metadata = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                full_text = f.read()
                chunks = split_into_chunks(full_text)
                texts.extend(chunks)
                metadata.extend([{"filename": filename}] * len(chunks))

    if not texts:
        print("[WARN] No text files found to ingest.")
        return

    print(f"[INFO] Ingesting {len(texts)} chunks from {len(metadata)} documents...")
    print("[DEBUG] First chunk preview:\n", texts[0][:300])

    # Create embeddings
    embeddings = embed_text(texts)  # Should return list/array of vectors
    embeddings = np.array(embeddings).astype("float32")

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save vector index and associated text chunks
    faiss.write_index(index, save_path + ".index")
    with open(save_path + ".pkl", "wb") as f:
        pickle.dump(texts, f)

    print(f"[INFO] Vector DB saved to {save_path}.index and {save_path}.pkl")