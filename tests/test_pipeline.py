from app.indexer import index_codebase
from app.rag_pipeline import load_rag

def test_pipeline():
    # Index a small repo (replace with test repo URL)
    index_codebase("https://github.com/psf/requests.git")

    # Load agent
    agent = load_rag()

    # Ask a simple question
    response = agent.run("What does the requests.get function do?")
    assert "HTTP" in response or "request" in response
