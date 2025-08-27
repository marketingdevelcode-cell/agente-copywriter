import os, json
import numpy as np
from typing import List, Dict, Any, Optional
from fastembed import TextEmbedding

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
INDEX_DIR = os.path.join(BASE_DIR, 'data', 'index')
CACHE_DIR = os.path.join(BASE_DIR, 'data', 'models')
MODEL_NAME = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

class RAGStore:
    def __init__(self):
        self.embedder = TextEmbedding(model_name=MODEL_NAME, cache_dir=CACHE_DIR)
        self.rows = []
        self.embs = None

    def load(self, path: Optional[str] = None):
        path = path or os.path.join(INDEX_DIR, 'rag_index.jsonl')
        rows, embs = [], []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                rows.append(obj)
                embs.append(obj['embedding'])
        self.rows = rows
        self.embs = np.array(embs, dtype=np.float32)
        norms = np.linalg.norm(self.embs, axis=1, keepdims=True) + 1e-9
        self.embs = self.embs / norms

    def _filter_rows(self, filters: Dict[str, Any]) -> List[int]:
        if not filters:
            return list(range(len(self.rows)))
        idxs = []
        for i, r in enumerate(self.rows):
            md = r.get('metadata', {})
            ok = True
            for key, val in filters.items():
                if val is None: 
                    continue
                if isinstance(val, list):
                    rv = md.get(key)
                    if rv is None:
                        ok = False; break
                    if isinstance(rv, list):
                        if len(set(v.lower() for v in val) & set(str(x).lower() for x in rv)) == 0:
                            ok = False; break
                    else:
                        if str(rv).lower() not in [str(x).lower() for x in val]:
                            ok = False; break
                else:
                    if str(md.get(key, '')).lower() != str(val).lower():
                        ok = False; break
            if ok:
                idxs.append(i)
        return idxs

    def search(self, query: str, top_k: int = 12, filters: Optional[Dict[str, Any]] = None):
        q_emb = list(self.embedder.embed([query]))[0].astype('float32')
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)

        mask = self._filter_rows(filters or {})
        if not mask:
            return []
        M = self.embs[mask]
        sims = M @ q_emb
        order = np.argsort(-sims)[:top_k]
        hits = []
        for idx in order:
            row = self.rows[mask[idx]]
            hits.append({
                "id": row["id"],
                "title": row["title"],
                "content": row["text"],
                "metadata": row.get("metadata", {}),
                "score": float(sims[idx]),
            })
        return hits
