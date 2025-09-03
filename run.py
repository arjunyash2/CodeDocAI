import os
import subprocess

print("🔹 Step 1: Building FAISS index...")
subprocess.run(["python", "app/indexer.py"])

print("🔹 Step 2: Starting Flask app...")
os.system("python app/main.py")