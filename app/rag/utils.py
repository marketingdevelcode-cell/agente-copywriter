import os, json, re
from typing import Dict, Any, Iterator, List

def iter_knowledge(base_path: str) -> Iterator[Dict[str, Any]]:
    """Itera arquivos em data/knowledge (md e json) emitindo documentos com metadados básicos.
    Para json: se for lista, emite item por item; se for dict com 'items', emite cada item.
    """
    for root, _, files in os.walk(base_path):
        for fn in files:
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, base_path).replace('\\', '/')
            if fn.endswith('.md'):
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
                title = rel
                metadata = infer_metadata_from_path(rel)
                yield {
                    "id": rel,
                    "title": title,
                    "content": text,
                    "metadata": metadata
                }
            elif fn.endswith('.json'):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # pode ser lista ou dict com items
                items = data.get('items', data) if isinstance(data, dict) else data
                if isinstance(items, list):
                    for i, item in enumerate(items):
                        doc = {
                            "id": f"{rel}#{i}",
                            "title": item.get("title") or rel,
                            "content": item.get("content") or json.dumps(item, ensure_ascii=False),
                            "metadata": {**infer_metadata_from_path(rel), **item.get("metadata", {})}
                        }
                        yield doc

def infer_metadata_from_path(rel_path: str) -> Dict[str, Any]:
    rel = rel_path.lower()
    md = {
        "language": "pt-BR",
        "source_file": rel_path
    }
    if "core" in rel:
        md["content_type"] = "doutrina"
    elif "bancos" in rel:
        md["content_type"] = "bloco"
    elif "estruturas" in rel:
        md["content_type"] = "estrutura"
    elif "frases" in rel:
        md["content_type"] = "frases"
    else:
        md["content_type"] = "desconhecido"
    return md

def simple_chunk(text: str, max_chars: int = 1200) -> List[str]:
    """Divide texto em pedaços (~1200 chars) respeitando parágrafos."""
    parts, buff = [], []
    total = 0
    for para in re.split(r"\n\s*\n", text.strip()):
        if total + len(para) + 2 > max_chars and buff:
            parts.append("\n\n".join(buff))
            buff, total = [], 0
        buff.append(para)
        total += len(para) + 2
    if buff:
        parts.append("\n\n".join(buff))
    return parts
