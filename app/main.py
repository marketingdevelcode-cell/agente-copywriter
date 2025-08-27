from fastapi import FastAPI
from .routers import search

app = FastAPI(title="RAG Copywriter — ICP TI (pt-BR)", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(search.router)
