import os
from agent_logic import EmbeddingManager, VectorStore, RAGPipeline

def main():
    print("--- Code Culture AI Agent (CC) ---")
    
    # Paths adjusted relative to script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "vector_store")
    
    try:
        embed_manager = EmbeddingManager()
        vector_store = VectorStore(persist_directory=data_dir)
        pipeline = RAGPipeline(vector_store, embed_manager)

        if vector_store.collection.count() == 0:
            print("Vector store is empty. Please run data ingestion first.")
        
        print("\nCC: Greetings, builder. I am online and ready to assist you.")
        print("    Type 'exit' to conclude our session.")
        
        while True:
            query = input("\nYou: ")
            if query.lower() in ['exit', 'quit', 'bye']:
                print("CC: Farewell. Keep building.")
                break
            
            if not query.strip():
                continue
                
            print("CC: Processing insights...")
            answer = pipeline.ask(query)
            print(f"\nCC: {answer}")

    except Exception as e:
        print(f"Error initializing CC: {e}")

if __name__ == "__main__":
    main()
