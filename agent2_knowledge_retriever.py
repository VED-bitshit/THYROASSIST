"""
Agent 2: Medical Knowledge Retriever (RAG)
Retrieves relevant medical guidelines and evidence
"""

import numpy as np
from typing import List, Dict
from dataclasses import dataclass
import re

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("sentence-transformers not available. Using fallback TF-IDF retrieval.")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# SAFE FALLBACKS (in case config is incomplete)
# ---------------------------------------------------------
try:
    from config import MEDICAL_GUIDELINES, RAGConfig
except Exception:
    MEDICAL_GUIDELINES = [
        {
            "id": "GEN-LOW",
            "title": "General Thyroid Monitoring",
            "content": "Routine monitoring is recommended for low risk thyroid patients."
        },
        {
            "id": "GEN-MOD",
            "title": "Moderate Thyroid Dysfunction",
            "content": "Patients with moderate thyroid dysfunction should undergo repeat testing."
        },
        {
            "id": "GEN-HIGH",
            "title": "High Risk Thyroid Disorders",
            "content": "High risk thyroid patients require urgent specialist referral."
        }
    ]

    class RAGConfig:
        embedding_model = "all-MiniLM-L6-v2"
        top_k_retrievals = 3


# ---------------------------------------------------------
# DATA STRUCTURE
# ---------------------------------------------------------
@dataclass
class RetrievedEvidence:
    citation_id: str
    title: str
    content: str
    relevance_score: float
    snippet: str


# ---------------------------------------------------------
# MAIN RETRIEVER
# ---------------------------------------------------------
class MedicalKnowledgeRetriever:
    """
    Agent 2: Retrieves relevant medical guidelines using RAG
    """

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.guidelines = MEDICAL_GUIDELINES
        self.use_embeddings = EMBEDDINGS_AVAILABLE

        if self.use_embeddings:
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            self.guideline_embeddings = self._create_embeddings()
        else:
            self.vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
            self.guideline_vectors = self._create_tfidf_vectors()

    def _create_embeddings(self) -> np.ndarray:
        texts = [g["content"] for g in self.guidelines]
        return self.embedding_model.encode(texts, show_progress_bar=False)

    def _create_tfidf_vectors(self) -> np.ndarray:
        texts = [g["content"] for g in self.guidelines]
        return self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = None) -> List[RetrievedEvidence]:
        top_k = top_k or self.config.top_k_retrievals

        if self.use_embeddings:
            return self._retrieve_with_embeddings(query, top_k)
        else:
            return self._retrieve_with_tfidf(query, top_k)

    def _retrieve_with_embeddings(self, query: str, top_k: int) -> List[RetrievedEvidence]:
        query_embedding = self.embedding_model.encode([query])[0]
        similarities = cosine_similarity(
            [query_embedding], self.guideline_embeddings
        )[0]

        indices = np.argsort(similarities)[-top_k:][::-1]

        return [
            RetrievedEvidence(
                citation_id=self.guidelines[i]["id"],
                title=self.guidelines[i]["title"],
                content=self.guidelines[i]["content"],
                relevance_score=float(similarities[i]),
                snippet=self._extract_snippet(self.guidelines[i]["content"], query),
            )
            for i in indices
        ]

    def _retrieve_with_tfidf(self, query: str, top_k: int) -> List[RetrievedEvidence]:
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.guideline_vectors)[0]

        indices = np.argsort(similarities)[-top_k:][::-1]

        return [
            RetrievedEvidence(
                citation_id=self.guidelines[i]["id"],
                title=self.guidelines[i]["title"],
                content=self.guidelines[i]["content"],
                relevance_score=float(similarities[i]),
                snippet=self._extract_snippet(self.guidelines[i]["content"], query),
            )
            for i in indices
        ]

    def _extract_snippet(self, content: str, query: str, max_length: int = 200) -> str:
        sentences = content.split(".")
        query_terms = set(query.lower().split())

        best_sentence = sentences[0]
        best_score = 0

        for sentence in sentences:
            overlap = len(query_terms.intersection(sentence.lower().split()))
            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence

        snippet = best_sentence.strip()
        return snippet[:max_length] + "..." if len(snippet) > max_length else snippet


# ---------------------------------------------------------
# 🔥 REQUIRED UI / SYSTEM ADAPTER FUNCTION 🔥
# ---------------------------------------------------------
def retrieve_evidence(risk_prediction) -> List[str]:
    """
    Adapter function REQUIRED by app.py.
    Converts RAG output → list[str] (safe for Agent 3 & 4)
    """

    # Accept dict or object
    if isinstance(risk_prediction, dict):
        risk_level = risk_prediction.get("risk_level", "LOW")
    else:
        risk_level = getattr(risk_prediction, "risk_level", "LOW")
        if hasattr(risk_level, "name"):
            risk_level = risk_level.name

    query_map = {
        "LOW": "routine thyroid monitoring",
        "MODERATE": "abnormal thyroid function repeat testing",
        "HIGH": "severe thyroid dysfunction urgent referral"
    }

    query = query_map.get(risk_level, "thyroid disorder")

    retriever = MedicalKnowledgeRetriever()
    results = retriever.retrieve(query)

    # Return plain text evidence (Agent 3-safe)
    return [
        f"{r.title}: {r.snippet}"
        for r in results
    ]


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------
if __name__ == "__main__":
    print(retrieve_evidence({"risk_level": "HIGH"}))