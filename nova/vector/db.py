import faiss
import pickle
import os
import numpy as np


def _paths(db_path: str):
    return db_path + ".index", db_path + ".pkl"


def search_db(query_vector, db_path, return_scores=False, top_k=4):

    index = faiss.read_index(f"{db_path}/vector_store.index")
    with open(f"{db_path}/vector_store.pkl", "rb") as f:
        texts = pickle.load(f)

    distances, indices = index.search(np.array([query_vector]).astype("float32"), top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(texts):
            text = texts[idx]
            score = distances[0][i]
            if return_scores:
                results.append((text, score))
            else:
                results.append(text)

    return results