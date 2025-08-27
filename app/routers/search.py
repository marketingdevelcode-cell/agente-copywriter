from fastapi import APIRouter
from ..utils.types import SearchIn, SearchOut, Chunk, TemplateIn
from ..rag.retriever import RAGStore

router = APIRouter(prefix="/search", tags=["search"])
store = RAGStore()
_store_loaded = False

def ensure_loaded():
    global _store_loaded
    if not _store_loaded:
        store.load()
        _store_loaded = True

@router.post("", response_model=SearchOut)
def search(in_: SearchIn):
    ensure_loaded()
    filters = in_.filters.model_dump() if in_.filters else {}
    hits = store.search(in_.query, top_k=in_.top_k or 12, filters=filters)
    return SearchOut(query=in_.query, hits=[Chunk(**h) for h in hits])

@router.post("/generate-template")
def generate_template(inp: TemplateIn):
    # retorna estrutura sugerida por formato (para o orquestrador)
    fmt = (inp.format or '').lower()
    if fmt == 'email':
        return {
            "format": fmt,
            "structure": [
                "Abertura validando pressão/contexto",\ 
                "Promessa de valor sem hype (roadmap/governança/ROI)",                "Prova social enxuta (par do setor)",                "CTA de baixo atrito (15–20min)",                "Assinatura confiável"
            ]
        }
    if fmt == 'post':
        return {
            "format": fmt,
            "structure": [
                "Gancho dor/desejo", "Mini-framework (3 passos)",
                "Fecho com pergunta + CTA suave"
            ]
        }
    if fmt == 'landing_page':
        return {
            "format": fmt,
            "structure": [
                "H1 dor/resultado", "Subhead método + risco mitigado",
                "Para quem", "O que entregamos (etapas)",
                "Provas (ROI/segurança)", "Perguntas críticas (objeções)", "CTA"
            ]
        }
    return {"format": fmt or "unknown", "structure": []}
