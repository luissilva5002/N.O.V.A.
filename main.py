import json
from nova.vector.ingest import ingest_documents
from nova.scripts.chat import chat_with_context

SETTINGS_FILE = "settings.json"

def load_settings(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    settings = load_settings(SETTINGS_FILE)

    documents_path = settings["scan_root"]
    db_path = settings["vector_db_path"]
    model_path = settings["model_path"]

    print(">> Ingesting documents...")
    ingest_documents(documents_path, db_path)

    print(">> Ready. Type your question (type 'exit' to quit).")
    while True:
        user_input = input("\nYou: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        answer = chat_with_context(user_input, db_path, model_path)
        print(f"\nN.O.V.A.: {answer}")

if __name__ == "__main__":
    main()
