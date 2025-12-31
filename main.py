import os
import uuid
import numpy as np
import chromadb
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

class EmbeddingManager:
    """Handles document embedding generation using SentenceTransformer"""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded. Dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded")
        return self.model.encode(texts, show_progress_bar=False)

class VectorStore:
    """Manages document embeddings in a ChromaDB vector store"""
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "./data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Code Culture document embeddings", "hnsw:space": "cosine"}
            )
            print(f"Vector store initialized. Collection '{self.collection_name}' has {self.collection.count()} documents.")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings length mismatch")
        
        ids = [f"doc_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(documents))]
        metadatas = [dict(doc.metadata) for doc in documents]
        texts = [doc.page_content for doc in documents]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=texts
        )
        print(f"Added {len(documents)} documents to vector store.")

class RAGPipeline:
    """Handles the full RAG pipeline: retrieval + generation"""
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.llm = self._setup_llm()

    def _setup_llm(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("WARNING: GOOGLE_API_KEY not found in environment.")
            return None
        return ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=api_key, temperature=0.2)

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.3) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        retrieved_docs = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                score = 1 - results['distances'][0][i]
                if score >= min_score:
                    retrieved_docs.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'score': score
                    })
        return retrieved_docs

    def ask(self, query: str) -> str:
        if not self.llm:
            return "Error: LLM not configured. Please set GOOGLE_API_KEY."
        
        context_docs = self.retrieve(query)
        if not context_docs:
            return "I couldn't find any relevant information in my knowledge base to answer that."

        context_text = "\n\n".join([doc['content'] for doc in context_docs])
        prompt = f"""You are CC, the AI agent for Code Culture. 
Use the following context to answer the user's question accurately and concisely.
If the answer isn't in the context, say you don't know based on the available records.

Context:
{context_text}

Question: {query}

Answer:"""
        
        return self._invoke_llm(prompt)

    def _invoke_llm(self, prompt: str) -> str:
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(prompt)
                content = response.content
                if isinstance(content, list):
                    # Handle case where content is a list of parts (common in some Gemini versions)
                    text_content = ""
                    for part in content:
                        if isinstance(part, dict) and 'text' in part:
                            text_content += part['text']
                        elif isinstance(part, str):
                            text_content += part
                    return text_content.strip()
                return content
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    print(f"Rate limit hit (429). Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                print(f"LLM Error: {e}")
                return f"I'm sorry, I encountered an error with my reasoning engine: {e}"
        return "I'm sorry, I failed to get a response after several attempts."

def main():
    print("--- Code Culture AI Agent (CC) ---")
    
    # Initialize managers
    # Paths adjusted relative to script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "vector_store")
    
    embed_manager = EmbeddingManager()
    vector_store = VectorStore(persist_directory=data_dir)
    pipeline = RAGPipeline(vector_store, embed_manager)

    if vector_store.collection.count() == 0:
        print("Vector store is empty. Please run data ingestion first (or check data path).")
    
    print("\nCC is ready! Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() in ['exit', 'quit', 'bye']:
            print("CC: Goodbye!")
            break
        
        if not query.strip():
            continue
            
        print("CC: Thinking...")
        answer = pipeline.ask(query)
        print(f"\nCC: {answer}")

if __name__ == "__main__":
    main()
