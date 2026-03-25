import faiss
import numpy as np
import pickle
import os
import google.generativeai as genai
import PyPDF2
from pathlib import Path

def pdf_to_vectors(pdf_path, model=None):
    """Process PDF into chunks and create FAISS vector database"""
    print(f"[INFO] Processing PDF: {pdf_path}")
    
    # Read PDF
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print("[ERROR] No text extracted from PDF")
        return [], [], [], 0
    
    total_pages = extract_pdf_pages(pdf_path)
    print(f"[INFO] Extracted {len(text):,} chars from {total_pages} pages")
    
    # Create chunks with page tracking
    chunks, chunk_metadata = create_pdf_chunks(text, total_pages, pdf_path)
    print(f"[INFO] Created {len(chunks)} chunks")
    
    # Generate embeddings in batches to avoid Google API 504 timeout on large inputs
    print("[INFO] Generating embeddings via Google API (batched)...")
    import google.generativeai as genai
    import os
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
    
    print("[OK] PDF processing complete!")
    return embeddings, chunks, chunk_metadata, total_pages

def extract_text_from_pdf(pdf_path):
    """Extract all text from PDF"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num} ---\n{page_text}\n"
    except Exception as e:
        print(f"[WARNING] PDF extraction warning: {e}")
    return text.strip()

def extract_pdf_pages(pdf_path):
    """Get total page count"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return len(pdf_reader.pages)
    except:
        return 0

def create_pdf_chunks(text, total_pages, pdf_path):
    """Split text into overlapping chunks with page metadata"""
    chunks = []
    metadata = []
    
    # Split by pages first, then chunk within pages
    pages_text = text.split('--- Page ')
    chunk_id = 0
    
    for page_text in pages_text[1:]:  # Skip first empty split
        if '---' in page_text:
            page_num, content = page_text.split('\n', 1)
            page_num = int(page_num.replace('---', '').strip())
        else:
            page_num = 1
            content = page_text
        
        # Chunk this page's content
        for i in range(0, len(content), 400):
            chunk_text = content[i:i + 500].strip()
            if len(chunk_text) > 50:  # Only substantial chunks
                chunks.append(chunk_text)
                
                # Estimate page for this chunk (most chunks stay on same page)
                estimated_page = page_num
                metadata.append({
                    "chunk_id": chunk_id,
                    "estimated_page": estimated_page,
                    "start_pos": i,
                    "source_file": Path(pdf_path).name
                })
                chunk_id += 1
    
    return chunks, metadata

if __name__ == "__main__":
    pdf_file = "sample_resume.pdf"
    if os.path.exists(pdf_file):
        embeddings, chunks, metadata, pages = pdf_to_vectors(pdf_file)
        print(f"\n[OK] Processed: {len(chunks)} chunks, {pages} pages")
    else:
        print("[INFO] Put a sample PDF and run again!")
