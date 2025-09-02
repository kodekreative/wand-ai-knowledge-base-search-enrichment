from fastapi import FastAPI
from .api import router

app = FastAPI(
    title="AI Knowledge Base",
    description="Semantic search and Q&A API"
)

app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {"message": "Welcome to the AI Knowledge Base API"}

# Placeholder for future endpoints
