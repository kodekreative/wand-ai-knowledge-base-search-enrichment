import sqlite3
import os
import hashlib
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class Database:
    def __init__(self, db_path="knowledge_base.db",
                 index_path="faiss_index.idx"):
        self.db_path = db_path
        self.index_path = index_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # For embeddings
        self.index = None
        self._init_db()
        self._load_index()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    filename TEXT UNIQUE,
                    content TEXT,
                    hash TEXT
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    doc_id INTEGER,
                    chunk_text TEXT,
                    FOREIGN KEY (doc_id) REFERENCES documents (id)
                )
            ''')
            conn.commit()

    def _load_index(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatIP(384)  # Cosine similarity, 384 dim

    def _save_index(self):
        faiss.write_index(self.index, self.index_path)

    def add_document_chunks(self, filename, content, chunks):
        doc_hash = hashlib.md5(content.encode()).hexdigest()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM documents WHERE filename = ?',
                           (filename,))
            existing = cursor.fetchone()
            if existing:
                cursor.execute('SELECT hash FROM documents WHERE id = ?',
                               (existing[0],))
                if cursor.fetchone()[0] == doc_hash:
                    return  # No change
                else:
                    # Update
                    cursor.execute(
                        'UPDATE documents SET content = ?, hash = ? '
                        'WHERE id = ?',
                        (content, doc_hash, existing[0]))
                    # Delete old chunks
                    cursor.execute('DELETE FROM chunks WHERE doc_id = ?',
                                   (existing[0],))
                    doc_id = existing[0]
            else:
                cursor.execute(
                    'INSERT INTO documents (filename, content, hash) '
                    'VALUES (?, ?, ?)',
                    (filename, content, doc_hash))
                doc_id = cursor.lastrowid
            
            # Add chunks
            for chunk in chunks:
                cursor.execute(
                    'INSERT INTO chunks (doc_id, chunk_text) VALUES (?, ?)',
                    (doc_id, chunk))
            
            # Rebuild index
            self._rebuild_index()
            conn.commit()

    def _rebuild_index(self):
        # Rebuild index from all chunks
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT chunk_text FROM chunks ORDER BY id')
            chunks = [row[0] for row in cursor.fetchall()]
        if chunks:
            embeddings = self.model.encode(chunks)
            embeddings = embeddings / np.linalg.norm(
                embeddings, axis=1, keepdims=True)
            self.index = faiss.IndexFlatIP(384)
            self.index.add(embeddings.astype('float32'))
            self._save_index()

    def search(self, query_embedding, top_k=5):
        if self.index.ntotal == 0:
            return []
        query_embedding = np.array([query_embedding])
        query_embedding = query_embedding / np.linalg.norm(
            query_embedding, axis=1, keepdims=True)
        distances, indices = self.index.search(
            query_embedding.astype('float32'), top_k)
        results = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT c.chunk_text, d.filename, d.content
                FROM chunks c
                JOIN documents d ON c.doc_id = d.id
                ORDER BY c.id
            ''')
            all_chunks = cursor.fetchall()
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(all_chunks):
                chunk_text, filename, content = all_chunks[idx]
                results.append((filename, chunk_text, dist))
        return results

    def get_all_documents(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT filename, content FROM documents')
            return cursor.fetchall()
