# RAG Copywriter — ICP Líderes de TI (pt-BR)

Base RAG e API de consulta para um agente **copywriter** pré-setado com o conhecimento do ICP (CIO/CTO/Diretores de TI). 
Inclui ontologia, chunking, metadados, indexação, retriever e prompts de geração/checagem.

## Como usar

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# 1) Indexar a base (gera embeddings em data/index/rag_index.jsonl)
python scripts/embed.py
# 2) Subir a API (FastAPI + Uvicorn)
uvicorn app.main:app --reload --port 8020
```

### Endpoints
- `GET /health` — status
- `POST /search` — busca híbrida (texto + filtros de metadados)
- `POST /search/generate-template` — recupera blocos + estrutura recomendada por formato

> Observação: este repositório prioriza **retrieval** + **estruturação**. A geração (model LLM) pode ser acoplada no seu orquestrador, usando os prompts em `prompts/` e os chunks retornados pelo `/search`.

## Estrutura
```
app/
  main.py
  rag/
    indexer.py
    retriever.py
    utils.py
  routers/
    search.py
  utils/
    types.py
data/
  knowledge/ ... (conteúdo ICP)
  index/ ... (arquivos gerados)
prompts/
  system_copywriter.md
  checker.md
scripts/
  embed.py
```

## Licença
MIT
