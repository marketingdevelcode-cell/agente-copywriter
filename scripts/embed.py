# scripts/embed.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.rag.indexer import build_index

if __name__ == '__main__':
    path, n = build_index()
    print(f'OK: {n} chunks → {path}')
