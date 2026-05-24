import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# 1. Initialize a highly efficient, lightweight embedding model (Runs on CPU)
print("Loading semantic embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2') 

# 2. Simulate chunks extracted from a corporate policy PDF
documents = [
    "The standard probation period for new lateral hires is six months, subject to review by the reporting manager.",
    "Employees are eligible for a gym reimbursement up to INR 15,000 annually under our wellness program.",
    "Production release deployments must be scheduled on Thursdays between 11:00 PM and 2:00 AM IST.",
    "The corporate office in Pune remains closed on the third Saturday of every month for deep cleaning."
]

# 3. Convert text chunks into mathematical vectors (Embeddings)
print("Generating vector embeddings...")
embeddings = model.encode(documents)
dimension = embeddings.shape[1]  # This model outputs 384-dimensional vectors

# 4. Initialize FAISS (An open-source vector index engine optimized for speed)
# Think of this like setting up a specialized index on a database column
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))
print(f"Successfully indexed {index.ntotal} document chunks into the vector store.")

# 5. Simulate a natural language query
user_query = "When can we push code changes to production?"
print(f"\nUser Query: '{user_query}'")

# Convert the user query into the exact same vector space
query_embedding = model.encode([user_query])

# Search the index for the Top 1 closest match (k=1)
distances, indices = index.search(np.array(query_embedding).astype('float32'), k=1)

# 6. Output the retrieved context
best_match_idx = indices[0][0]
retrieved_context = documents[best_match_idx]

print("\n--- [System Retrieval Complete] ---")
print(f"Retrieved Context (Distance score: {distances[0][0]:.4f}):")
print(f"👉 {retrieved_context}")
print("\nNext Architectural Step: This text block would now be injected into an LLM prompt as context.")