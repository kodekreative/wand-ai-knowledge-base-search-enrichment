def search_documents(query, db, top_k=5):
    """Semantic search: embed query, search FAISS, return results."""
    if not db.model:
        raise ValueError("Database not initialized")
    query_embedding = db.model.encode([query])[0]
    results = db.search(query_embedding, top_k)
    return [{"filename": r[0], "content": r[1], "score": float(r[2])}
            for r in results]
