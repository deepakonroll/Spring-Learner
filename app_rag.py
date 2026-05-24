import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# NEW: Import the specialized text splitter
from llama_index.core.node_parser import SentenceSplitter

def run_fine_tuned_retrieval():
    print("Step 1: Setting up local models...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = None 

    # NEW: Tighten the chunk size dramatically. 
    # We tell the system to break text into blocks of roughly 40 tokens 
    # instead of the default 1024.
    Settings.node_parser = SentenceSplitter(chunk_size=40, chunk_overlap=5)

    print("\nStep 2: Parsing files from your local './data' folder...")
    documents = SimpleDirectoryReader("D:\data").load_data()

    print("\nStep 3: Calculating vectors for the small, precise chunks...")
    index = VectorStoreIndex.from_documents(documents)
    
    # Let's see how many separate chunks the system created this time!
    # Under the hood, LlamaIndex handles storage in a docstore
    nodes = index.docstore.docs.values()
    print(f"👉 Architecture Optimization: Sliced the text into {len(nodes)} distinct vector chunks.")

    retriever = index.as_retriever(similarity_top_k=1)

    print("\n🚀 Fine-Tuned Local Search Active! Type 'exit' to quit.")
    while True:
        user_query = input("\nSearch your PDF for: ")
        if user_query.strip().lower() == 'exit':
            break
            
        if not user_query.strip():
            continue
            
        nodes = retriever.retrieve(user_query)
        
        if nodes:
            print("\n=== [Precise Text Match Found] ===")
            print(f"Content: {nodes[0].node.get_content()}")
            print(f"Mathematical Distance Score: {nodes[0].score:.4f}")
            print("==================================")

if __name__ == "__main__":
    run_fine_tuned_retrieval()