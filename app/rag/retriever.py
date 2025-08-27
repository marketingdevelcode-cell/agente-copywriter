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
        self.rows, self.embs = [], None

    def load(self, path: Optional[str] = None):
        path = path or os.path.join(INDEX_DIR, 'rag_index.jsonl')
        rows, embs = [], []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line); rows.append(obj); embs.append(obj['embedding'])
        self.rows = rows; self.embs = np.array(embs, dtype=np.float32)
        self.embs /= (np.linalg.norm(self.embs, axis=1, keepdims=True) + 1e-9)

    def _filter_rows(self, filters: Dict[str, Any]) -> List[int]:
        if not filters: return list(range(len(self.rows)))
        idxs = []
        for i, r in enumerate(self.rows):
            md, ok = r.get('metadata', {}), True
            for k, v in (filters or {}).items():
                if v is None: continue
                rv = md.get(k)
                if isinstance(v, list):
                    if rv is None: ok=False; break
                    ok = bool(set(map(str.lower, map(str, v))) & set(map(str.lower, [rv] if not isinstance(rv, list) else rv)))
                    if not ok: break
                else:
                    ok = (str(rv).lower() == str(v).lower())
                    if not ok: break
            if ok: idxs.append(i)
        return idxs

    def search(self, query: str, top_k: int = 12, filters: Optional[Dict[str, Any]] = None):
        q = list(self.embedder.embed([query]))[0].astype('float32'); q /= (np.linalg.norm(q) + 1e-9)
        mask = self._filter_rows(filters or {}); 
        if not mask: return []
        M = self.embs[mask]; sims = M @ q; order = np.argsort(-sims)[:top_k]
        return [{"id": self.rows[mask[i]]["id"], "title": self.rows[mask[i]]["title"],
                 "content": self.rows[mask[i]]["text"],
                 "metadata": self.rows[mask[i]].get("metadata", {}), "score": float(sims[i])}
                for i in order]
