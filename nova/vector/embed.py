# nova/vector/embed.py

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(texts):
    return model.encode(texts, show_progress_bar=True)
