import os
import sys
import subprocess
import json

REQUIREMENTS_FILE = "requirements.txt"
SETTINGS_FILE = "settings.json"

# Step 1: Ensure all dependencies are installed
def install_requirements():
    try:
        import llama_cpp  # test an important package
    except ImportError:
        print(">> Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE])

# Step 2: Load assistant modules AFTER installation
def run_nova():
    from nova.vector.ingest import ingest_documents
    from nova.scripts.chat import chat_with_context

    def load_settings(path):
        with open(path, "r") as f:
            return json.load(f)

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
    install_requirements()
    run_nova()
