import faiss
import numpy as np
import pickle
import google.generativeai as genai
import os
from pathlib import Path

def txt_to_vectors(txt_path, model=None):
    """Process TXT file into chunks and create FAISS vector database"""
    print(f"[INFO] Reading TXT: {txt_path}")
    
    # Read TXT with UTF-8 encoding
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(txt_path, "r", encoding="latin-1") as f:
            text = f.read()
    
    if not text.strip():
        print("[ERROR] No text found in file")
        return [], []
    
    # Single file = 1 logical page
    total_pages = 1
    print(f"[INFO] Total text length: {len(text):,} characters")
    
    # Create chunks with metadata
    chunks, metadata = create_txt_chunks(text, txt_path)
    print(f"[INFO] Created {len(chunks)} chunks")
    
    # Generate embeddings in batches to avoid Google API 504 timeout on large inputs
    print("[INFO] Loading Google Embeddings API generating vectors (batched)...")
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    
    available_models = [m.name for m in genai.list_models() if 'embedContent' in m.supported_generation_methods]
    chosen_model = "models/text-embedding-004" if "models/text-embedding-004" in available_models else (available_models[-1] if available_models else "models/gemini-embedding-2-preview")
    print(f"[INFO] Using model: {chosen_model}")
    
    BATCH_SIZE = 20
    all_embeddings = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        result = genai.embed_content(
            model=chosen_model,
            content=batch,
            task_type="retrieval_document"
        )
        all_embeddings.extend(result['embedding'])
        print(f"[INFO] Embedded batch {i // BATCH_SIZE + 1}/{(len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE}")
    
    embeddings = np.array(all_embeddings)
    
    # Create FAISS index — detect dimension dynamically from actual Google API output
    dim = embeddings.shape[1]
    print(f"[INFO] Creating FAISS index (dim={dim})...")
    embeddings_f32 = embeddings.astype("float32")
    faiss.normalize_L2(embeddings_f32)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_f32)
    
    # Save immediately for standalone use
    save_vectors(embeddings, chunks, metadata, total_pages, "vectors.index", "chunks.pkl")
    
    print("[OK] TXT vector database created!")
    print(f"[INFO] Shape: {embeddings.shape}")
    return embeddings, chunks, metadata

def create_txt_chunks(text, txt_path):
    """Create overlapping chunks from text with metadata"""
    chunks = []
    metadata = []
    
    for i in range(0, len(text), 400):
        chunk_text = text[i:i + 500].strip()
        if len(chunk_text) > 50:  # Minimum chunk size
            chunks.append(chunk_text)
            metadata.append({
                "start_pos": i,
                "estimated_page": 1,
                "source_file": Path(txt_path).name
            })
    
    return chunks, metadata

def save_vectors(embeddings, chunks, metadata, total_pages, index_path="vectors.index", chunks_path="chunks.pkl"):
    """Save vectors and metadata"""
    import faiss
    embeddings_f32 = embeddings.astype("float32")
    faiss.normalize_L2(embeddings_f32)
    dim = embeddings_f32.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_f32)
    
    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump({
            "chunks": chunks,
            "metadata": metadata,
            "total_pages": total_pages,
        }, f)
    
    print(f"[OK] Saved: {index_path}, {chunks_path}")

if __name__ == "__main__":
    txt_file = "faq/hotel_info.txt"
    if os.path.exists(txt_file):
        embeddings, chunks = txt_to_vectors(txt_file)
    else:
        print("[INFO] Create 'faq/hotel_info.txt' and run again!")
