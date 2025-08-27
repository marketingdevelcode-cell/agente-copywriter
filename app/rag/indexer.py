import os, json, uuid
import numpy as np
from fastembed import TextEmbedding
from .utils import iter_knowledge, simple_chunk

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
KNOW_DIR = os.path.join(BASE_DIR, 'data', 'knowledge')
INDEX_DIR = os.path.join(BASE_DIR, 'data', 'index')
MODEL_NAME = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
CACHE_DIR = os.path.join(BASE_DIR, 'data', 'models')
os.makedirs(INDEX_DIR, exist_ok=True); os.makedirs(CACHE_DIR, exist_ok=True)

def build_index():
    embedder = TextEmbedding(model_name=MODEL_NAME, cache_dir=CACHE_DIR)
    rows = []
    for doc in iter_knowledge(KNOW_DIR):
        for j, chunk in enumerate(simple_chunk(doc['content'], max_chars=1200)):
            rows.append({"uid": str(uuid.uuid4()), "id": f"{doc['id']}::chunk{j}",
                         "title": doc['title'], "text": chunk, "metadata": doc['metadata']})
    embs = np.vstack(list(embedder.embed([r['text'] for r in rows]))).astype('float32')
    embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    out_path = os.path.join(INDEX_DIR, 'rag_index.jsonl')
    with open(out_path, 'w', encoding='utf-8') as f:
        for r, e in zip(rows, embs): r2 = {**r, "embedding": e.tolist()}; f.write(json.dumps(r2, ensure_ascii=False)+'\n')
    return out_path, len(rows)

if __name__ == '__main__':
    p, n = build_index(); print(f'Index criado: {p} ({n} chunks)')
