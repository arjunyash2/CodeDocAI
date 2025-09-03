from flask import Flask, request, jsonify, render_template
from app.agent import create_agent
from app.indexer import process_repository
import os

app = Flask(__name__, template_folder="../templates", static_folder="../static")

# --- Global State ---
# We use a simple global variable to hold our agent.
# It starts as None and is created after a repo is indexed.
agent = None


def is_vector_store_ready():
    """Check if the vector store has been created."""
    return os.path.exists("vector_store")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/setup", methods=["POST"])
def setup():
    """
    Endpoint to clone a repo and build the vector store.
    """
    global agent
    data = request.get_json()
    repo_url = data.get("repo_url", "")

    if not repo_url:
        return jsonify({"success": False, "message": "❌ No repository URL provided"}), 400

    try:
        success, message = process_repository(repo_url)
        if success:
            # If successful, create the agent for the new vector store
            agent = create_agent()
            return jsonify({"success": True, "message": f"✅ {message}"})
        else:
            # If failed, ensure agent is None
            agent = None
            return jsonify({"success": False, "message": f"❌ {message}"}), 500

    except Exception as e:
        agent = None  # Reset agent on failure
        print(f"⚠️  Error in /setup: {str(e)}")
        return jsonify({"success": False, "message": f"⚠️  An unexpected error occurred: {str(e)}"}), 500


@app.route("/ask", methods=["POST"])
def ask():
    """
    Endpoint to ask questions to the agent.
    """
    global agent
    if not is_vector_store_ready() or agent is None:
        return jsonify(
            {"response": "The knowledge base is not ready. Please provide a GitHub repository to get started."}), 400

    data = request.get_json()
    question = data.get("query", "")

    if not question:
        return jsonify({"response": "❌ No query provided"}), 400

    try:
        print(f"Invoking agent with query: '{question}'")
        result = agent.invoke({"query": question})
        print(f"Full agent result: {result}")

        response = result.get("result", "Agent did not return a valid output.")
        return jsonify({"response": response})

    except Exception as e:
        error_message = f"⚠️  An error occurred while processing the query: {str(e)}"
        print(error_message)
        return jsonify({"response": error_message}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)

