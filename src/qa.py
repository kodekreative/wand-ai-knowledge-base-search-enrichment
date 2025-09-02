import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from .search import search_documents

load_dotenv()


def qa_answer(query, db):
    """Q&A using Groq API with Llama."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    # Simple: concatenate top search results
    search_results = search_documents(query, db, top_k=3)
    context = "\n".join([r["content"][:1000] for r in search_results])
    # Limit context

    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key)
    prompt = (
        f"Answer the question based on the context:\n{context}\n\n"
        f"Question: {query}"
    )
    response = llm.invoke(prompt)
    return response.content


def check_completeness(query, db):
    """Check if query is fully answered by top results."""
    results = search_documents(query, db, top_k=3)
    if not results:
        return {"complete": False, "reason": "No documents found"}

    # Simple heuristic: if top score > 0.5, consider complete
    top_score = results[0]["score"]
    return {"complete": top_score > 0.5, "top_score": top_score}
