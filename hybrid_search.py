"""
hybrid_search.py
Combines ExactMatch and SemanticSearch into a unified, deduplicated hybrid retriever.
"""
import re
from difflib import SequenceMatcher
from typing import Dict, List
from backend_exact_match import ExactMatchEngine
from labse import SemanticSearchEngine

def _clean_and_deduplicate_text(text: str, threshold: float = 0.8) -> str:
    if not text:
        return ""
    sentences = re.split(r'[။.]+', text)
    unique_sentences = []

    for s in sentences:
        s = s.strip()
        if not s:
            continue
             
        is_duplicate = False
        for existing in unique_sentences:
            sim = SequenceMatcher(None, s.replace(" ", ""), existing.replace(" ", "")).ratio()
            if sim >= threshold:
                is_duplicate = True
                break
                
        if not is_duplicate:
            unique_sentences.append(s)
            
    return "။ ".join(unique_sentences) + "။" if unique_sentences else ""

def _deduplicate_results(results: List[Dict], threshold: float = 0.85) -> List[Dict]:
    unique_results = []
    for item in results:
        text = (item.get("context") or item.get("content", "")).strip()
        if not text:
            continue
            
        is_duplicate = False
        for existing in unique_results:
            existing_text = (existing.get("context") or existing.get("content", "")).strip()
            sim = SequenceMatcher(None, text.replace(" ", ""), existing_text.replace(" ", "")).ratio()
            if sim >= threshold:
                is_duplicate = True
                break
                
        if not is_duplicate:
            unique_results.append(item)
            
    return unique_results

class HybridSearchEngine:
    def __init__(self, exact_json_path: str, labse_model_path: str, faiss_path: str, device: str = "cpu"):
        self.exact_engine = ExactMatchEngine(exact_json_path)
        self.semantic_engine = SemanticSearchEngine(
            model_path=labse_model_path,
            faiss_path=faiss_path,
            device=device
        )

    def search(self, query: str, top_k: int = 5) -> Dict:
        exact_res = self.exact_engine.search(query)
        semantic_res = self.semantic_engine.search(query, top_k=top_k)

        combined = []
        seen_ids = set()

        # 1. Add Exact Matches
        if exact_res and "results" in exact_res and not exact_res.get("error"):
            for item in exact_res["results"]:
                uid = f"exact_{item['number']}"
                if uid not in seen_ids:
                    cleaned_context = _clean_and_deduplicate_text(item["context"])
                    combined.append({
                        "match_type": "exact",
                        "section": item["section"],
                        "context": cleaned_context,
                        "section_number": item["number"],
                        "confidence": 1.0
                    })
                    seen_ids.add(uid)

        # 2. Add Semantic Matches
        for item in semantic_res:
            uid = f"semantic_{item['sentence'].strip().lower()}"
            if uid not in seen_ids:
                cleaned_content = _clean_and_deduplicate_text(item["sentence"])
                combined.append({
                    "match_type": "semantic",
                    "content": cleaned_content,
                    "similarity": item["similarity"],
                    "confidence": item["similarity"]
                })
                seen_ids.add(uid)

        # 3. Remove cross-result duplicates
        combined = _deduplicate_results(combined)

        # 4. Sort & enforce top_k
        combined.sort(key=lambda x: x["confidence"], reverse=True)
        exact_count = sum(1 for r in combined if r["match_type"] == "exact")
        semantic_limit = max(top_k - exact_count, 0)

        final_results = [r for r in combined if r["match_type"] == "exact"]
        final_results += [r for r in combined if r["match_type"] == "semantic"][:semantic_limit]

        return {
            "query": query,
            "missing_sections": exact_res.get("missing", []) if exact_res else [],
            "results": final_results,
            "metadata": {
                "exact_found": exact_count,
                "semantic_found": len(semantic_res)
            }
        }