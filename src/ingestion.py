import os
from langchain_text_splitters import RecursiveCharacterTextSplitter


def ingest_document(file_path, db):
    """Ingest a TXT document: load, chunk, embed, store."""
    if not file_path.endswith('.txt'):
        raise ValueError("Only TXT files supported")
    
    # Handle large files: read in chunks
    chunk_size = 1024 * 1024  # 1MB
    content = ""
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            content += chunk
    
    # Chunk the content
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_text(content)
    
    # Store in DB with chunks
    db.add_document_chunks(os.path.basename(file_path), content, chunks)
