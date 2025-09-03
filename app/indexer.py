import os
import shutil
import git
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configuration ---
# Define a list of file extensions to process for a wider range of projects
ALLOWED_EXTENSIONS = {
    '.py', '.js', '.jsx', '.ts', '.tsx', '.html', '.css', '.scss', '.md',
    '.java', '.cpp', '.h', '.c', '.cs', '.go', '.rs', '.php', '.rb', '.swift',
    '.kt', '.kts', '.sh', '.yml', '.yaml', '.json', '.toml', '.ini', '.cfg'
}

# --- KEY CHANGE: Define directories and files to ignore ---
# This prevents the index from being cluttered with irrelevant data.
EXCLUDED_DIRS = {'.git', 'node_modules', 'venv', '.venv', '__pycache__', 'dist', 'build'}
EXCLUDED_FILES = {'package-lock.json', 'yarn.lock'}


def safe_load_file(path: str) -> list[Document]:
    """Safely reads a file with fallback encoding and returns it as a Document."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return [Document(page_content=text, metadata={"source": path})]
    except Exception as e:
        print(f"⚠️  Skipped {path} due to error: {e}")
        return []


def process_repository(repo_url: str) -> tuple[bool, str]:
    """
    Clones a repository, processes its files, and builds a FAISS vector store.
    """
    repo_dir = "repo"
    vector_store_path = "vector_store"

    # --- 1. Cleanup: Remove old repo and vector store if they exist ---
    print("🧹 Cleaning up old directories...")
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)
    if os.path.exists(vector_store_path):
        shutil.rmtree(vector_store_path)

    # --- 2. Clone Repository ---
    try:
        print(f"Cloning repository: {repo_url}...")
        git.Repo.clone_from(repo_url, repo_dir)
        print("✅ Repository cloned.")
    except Exception as e:
        return False, f"Failed to clone repository: {e}"

    # --- 3. Load and Process Files ---
    docs = []
    print("Processing files...")
    for root, dirs, files in os.walk(repo_dir):
        # --- KEY CHANGE: Exclude specified directories ---
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

        for file in files:
            # --- KEY CHANGE: Exclude specified files ---
            if file in EXCLUDED_FILES:
                continue

            if any(file.endswith(ext) for ext in ALLOWED_EXTENSIONS):
                path = os.path.join(root, file)
                docs.extend(safe_load_file(path))

    if not docs:
        return False, "No valid documents found to index in the repository."

    print(f"Found {len(docs)} documents. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    # --- 4. Create Vector Store ---
    try:
        print("Generating embeddings and building vector store...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(vector_store_path)
        print("✅ Vector store created and saved.")
    except Exception as e:
        return False, f"Failed to create vector store: {e}"

    # --- 5. Final Cleanup ---
    finally:
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
            print("✅ Cleaned up temporary repository directory.")

    return True, "Repository indexed successfully! You can now ask questions."

