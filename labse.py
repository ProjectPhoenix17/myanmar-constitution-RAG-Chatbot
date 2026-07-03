"""
labse.py
Handles vector-based semantic search using LaBSE embeddings and FAISS.
"""
import os
from typing import List, Dict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class SemanticSearchEngine:
    def __init__(self, model_path: str = "./labse-finetuned0207", faiss_path: str = "./faiss_index၁",
                 device: str = "cpu", similarity_threshold: float = 0.75):
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"FAISS index not found at {faiss_path}")
        if not os.path.exists(model_path):
            print(f"⚠️ Model path {model_path} not found. HuggingFace will attempt to download it automatically.")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_path, 
            model_kwargs={"device": device}, 
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vectorstore = FAISS.load_local(faiss_path, self.embeddings, allow_dangerous_deserialization=True)
        self.threshold = similarity_threshold

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=top_k * 3)
        seen_sentences = set()
        results = []

        for doc, score in docs_and_scores:
            sentence = doc.metadata.get("sentence", doc.metadata.get("sentence2", doc.page_content)).strip()
            if not sentence: 
                continue

            similarity = 1.0 - score
            if similarity < self.threshold: 
                continue

            norm_sentence = sentence.lower()
            if norm_sentence in seen_sentences: 
                continue
            seen_sentences.add(norm_sentence)

            results.append({"sentence": sentence, "similarity": similarity})
            if len(results) >= top_k:
                break

        return results