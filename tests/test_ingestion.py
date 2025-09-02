import pytest
from src.database import Database
from src.ingestion import ingest_document
import os
import tempfile


@pytest.fixture
def db():
    # Use temp DB file
    import tempfile
    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(db_fd)  # Close fd, keep path
    db = Database(db_path=db_path, index_path="temp.idx")
    yield db
    if os.path.exists(db_path):
        os.remove(db_path)
    if os.path.exists("temp.idx"):
        os.remove("temp.idx")


def test_ingest_document(db):
    # Create temp TXT file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                     delete=False) as f:
        f.write("This is a test document.")
        temp_path = f.name
    ingest_document(temp_path, db)
    docs = db.get_all_documents()
    assert len(docs) == 1
    assert docs[0][1] == "This is a test document."
    os.unlink(temp_path)


def test_search_documents(db):
    # Add doc first
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                     delete=False) as f:
        f.write("Python is a programming language.")
        temp_path = f.name
    ingest_document(temp_path, db)
    from src.search import search_documents
    results = search_documents("programming", db)
    assert len(results) > 0
    os.unlink(temp_path)
