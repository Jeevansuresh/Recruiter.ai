import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

def ask_question(question, index_path="vectors.index", chunks_path="chunks.pkl"):
    """
    Core RAG function - finds most relevant chunks for a question
    Returns: (answer, context) - answer=None, context=relevant chunks
    """
    # Check if vector files exist
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        print("[ERROR] Vector database not found!")
        print("[INFO] Files needed: vectors.index, chunks.pkl")
        return None, None

    try:
        # Load FAISS index and data
        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            data = pickle.load(f)

        chunks = data["chunks"]
        metadata = data.get("metadata", [])
        total_pages = data.get("total_pages", 1)

        # Load model and encode question
        print("[INFO] Loading MiniLM-L6-v2 model...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        query_vector = model.encode([question]).astype("float32")
        faiss.normalize_L2(query_vector)  # Normalize to match normalized index

        # Search for top 5 most similar chunks
        k = min(5, len(chunks))  # Don't exceed available chunks
        scores, indices = index.search(query_vector, k)

        # Build enriched context with metadata
        context_parts = []
        print(f"[INFO] Found {len(indices[0])} relevant chunks:")
        
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(chunks):  # Valid index
                chunk_text = chunks[idx]
                
                # Get metadata if available
                page_info = "Unknown"
                filename = "Unknown"
                if idx < len(metadata):
                    meta = metadata[idx]
                    page_info = meta.get("estimated_page", "Unknown")
                    filename = meta.get("source_file", "Unknown")
                
                context_parts.append(f"[File: {filename} | Page: {page_info} | Score: {score:.3f}] {chunk_text}")
                
                print(f"  - Chunk {i+1}: Score {score:.3f} (Page {page_info}, File: {filename})")

        context = "\n\n".join(context_parts)
        
        # --- GEMINI GENERATION START ---
        import google.generativeai as genai
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            print("[WARNING] GEMINI_API_KEY not found in .env - skipping AI generation")
            return None, context
            
        try:
            print("[INFO] Generating answer with Gemini...")
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel("gemini-1.5-flash") # Keep 1.5-flash for now or change to 2.0? 
            # I'll change to 2.0-flash as 1.5 seems missing.
            gemini_model = genai.GenerativeModel("gemini-2.0-flash")
            
            prompt = f"""
            You are an expert AI recruiter assistant.
            Based ONLY on the following context from resume documents, answer the user's question.
            If the answer is not in the context, say "I couldn't find that information in the resumes."
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:
            """
            
            response = gemini_model.generate_content(prompt)
            answer = response.text.strip()
            print("[OK] Answer generated!")
            return answer, context
            
        except Exception as llm_err:
            print(f"[ERROR] Gemini API Error: {llm_err}")
            return None, context
        # --- GEMINI GENERATION END ---

    except Exception as e:
        print(f"[ERROR] Error processing question: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def get_database_stats(index_path="vectors.index", chunks_path="chunks.pkl"):
    """Get stats about the current database"""
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        return None
    
    try:
        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            data = pickle.load(f)
        
        return {
            "total_chunks": len(data["chunks"]),
            "total_pages": data.get("total_pages", 1),
            "embedding_dim": index.d,
            "index_type": index.__class__.__name__
        }
    except:
        return None

# Interactive CLI for testing
def main():
    print("[RAG] Query System")
    print("[INFO] Looking for: vectors.index, chunks.pkl")
    
    stats = get_database_stats()
    if stats:
        print(f"[OK] Database loaded: {stats['total_chunks']} chunks, {stats['total_pages']} pages")
    else:
        print("[ERROR] No database found!")
        print("[INFO] Run pdf_to_vectors.py or txt_to_vectors.py first")
        return
    
    print("\n" + "="*60)
    print("Ask questions about your documents!")
    print("'stats' = database info")
    print("'quit', 'exit', 'q' = exit")
    print("="*60)
    
    while True:
        question = input("\n❓ Question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q', 'bye']:
            print("Goodbye!")
            break
            
        if question.lower() == 'stats':
            print("Database Stats:")
            for k, v in stats.items():
                print(f"   - {k}: {v}")
            continue
        
        if not question:
            print("[WARNING] Please enter a question!")
            continue
        
        print("[INFO] Searching...")
        _, context = ask_question(question)
        
        if context:
            if answer:
                print("\n🤖 AI ANSWER:\n")
                print(answer)
                print("\n" + "-"*40 + "\n")
            
            print("\nTOP MATCHES:\n")
            print(context[:2000] + "..." if len(context) > 2000 else context)
        else:
            print("[ERROR] No relevant content found")

if __name__ == "__main__":
    main()
