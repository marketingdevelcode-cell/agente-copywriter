import os, json, uuid
import numpy as np
from sentence_transformers import SentenceTransformer
from .utils import iter_knowledge, simple_chunk

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
KNOW_DIR = os.path.join(BASE_DIR, 'data', 'knowledge')
INDEX_DIR = os.path.join(BASE_DIR, 'data', 'index')
os.makedirs(INDEX_DIR, exist_ok=True)

MODEL_NAME = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

def build_index():
    model = SentenceTransformer(MODEL_NAME)
    rows = []
    for doc in iter_knowledge(KNOW_DIR):
        chunks = simple_chunk(doc['content'], max_chars=1200)
        for j, chunk in enumerate(chunks):
            row = {
                "uid": str(uuid.uuid4()),
                "id": f"{doc['id']}::chunk{j}",
                "title": doc['title'],
                "text": chunk,
                "metadata": doc['metadata']
            }
            rows.append(row)
    # embeddings
    texts = [r['text'] for r in rows]
    embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    # write jsonl
    out_path = os.path.join(INDEX_DIR, 'rag_index.jsonl')
    with open(out_path, 'w', encoding='utf-8') as f:
        for r, e in zip(rows, embs):
            r2 = {**r, "embedding": e.tolist()}
            f.write(json.dumps(r2, ensure_ascii=False) + '\n')
    return out_path, len(rows)

if __name__ == '__main__':
    p, n = build_index()
    print(f'Index criado: {p} ({n} chunks)')
