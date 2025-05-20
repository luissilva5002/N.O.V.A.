import faiss
import pickle
import os
import numpy as np


def _paths(db_path: str):
    return db_path + ".index", db_path + ".pkl"


def search_db(query_vector: np.ndarray, db_path: str, top_k: int = 4) -> list[str]:
    index_path, texts_path = _paths(db_path)

    if not os.path.exists(index_path):
        print("[ERROR] FAISS index not found.")
        return []

    index = faiss.read_index(index_path)
    with open(texts_path, "rb") as f:
        texts = pickle.load(f)

    query_vector = np.array([query_vector]).astype("float32")
    distances, indices = index.search(query_vector, top_k)

    results = []
    for i, dist in zip(indices[0], distances[0]):
        if i < len(texts):
            print(f"[DEBUG] Match index {i}, score: {dist:.4f}")
            results.append(texts[i])
        else:
            print(f"[WARN] Ignored index {i} (out of bounds)")

    return results