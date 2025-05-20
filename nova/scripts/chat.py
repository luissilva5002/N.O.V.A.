from llama_cpp import Llama
from nova.vector.embed import embed_text
from nova.vector.db import search_db

# Global cache for model instance
_llm = None

def get_llm(model_path: str = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"):
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=6,
            verbose=False
        )
    return _llm

def format_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(chunk.strip() for chunk in context_chunks if chunk.strip())
    return f"""You are N.O.V.A., an intelligent assistant.

Answer the following question based only on the context provided.

Context:
{context}

Question: {question}
Answer:"""

def chat_with_context(question: str, db_path: str, model_path: str) -> str:
    llm = get_llm(model_path)

    query_vec = embed_text(question)
    context_chunks = search_db(query_vec, db_path)

    if not context_chunks:
        return "I couldn't find any relevant information to answer that."

    print("\n[DEBUG] Retrieved context:")
    for i, chunk in enumerate(context_chunks):
        print(f"\n--- Chunk {i+1} ---\n{chunk.strip()}\n")

    prompt = format_prompt(question, context_chunks)
    print("\n[DEBUG] Prompt being sent to LLM:\n", prompt[:500], "..." if len(prompt) > 500 else "")

    output = llm(prompt, max_tokens=256, stop=["User:", "\n\n"])
    return output["choices"][0]["text"].strip()