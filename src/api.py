from fastapi import APIRouter, UploadFile, File, HTTPException
from .database import Database
from .ingestion import ingest_document
from .search import search_documents
from .qa import qa_answer, check_completeness
import os

router = APIRouter()
db = Database()


@router.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only TXT files allowed")
    content = await file.read()
    content_str = content.decode('utf-8')
    # Save temp file
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, 'w') as f:
        f.write(content_str)
    ingest_document(temp_path, db)
    os.remove(temp_path)
    return {"message": "Document ingested"}


@router.get("/search")
async def search(query: str, top_k: int = 5):
    results = search_documents(query, db, top_k)
    return {"results": results}


@router.post("/qa")
async def qa(query: str):
    answer = qa_answer(query, db)
    return {"answer": answer}


@router.post("/check_completeness")
async def completeness(query: str):
    result = check_completeness(query, db)
    return result
