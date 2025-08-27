import os
from app.rag.indexer import build_index

if __name__ == '__main__':
    path, n = build_index()
    print(f'OK: {n} chunks → {path}')
