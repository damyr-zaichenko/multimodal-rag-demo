import subprocess
import spacy
from typing import List, Dict, Any, Tuple
from storage.faiss_chunk_store import FaissChunkStore
from embedding.embedder import ChunkEmbedder

class QueryEngine:
    def __init__(self, store: FaissChunkStore, embedder: ChunkEmbedder, model_name: str = "en_core_web_sm"):
        self.store = store
        self.embedder = embedder
        self.nlp = self._load_spacy_model(model_name)

    def _load_spacy_model(self, model_name: str):
        try:
            return spacy.load(model_name)
        except OSError:
            subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
            return spacy.load(model_name)

    def _extract_entities(self, query: str) -> List[str]:
        doc = self.nlp(query)
        return [ent.text for ent in doc.ents if ent.label_ in {"ORG", "PERSON", "PRODUCT"}]

    def _boost_score(self, score: float, meta: Dict[str, Any], entities: List[str], weight: float = 0.1) -> float:
        match_bonus = sum(1 for entity in entities if entity.lower() in meta.get("chunk", "").lower())
        return score - match_bonus * weight  # lower cosine distance is better

    def query(self, query_text: str, top_k: int = 10) -> Dict[str, Any]:
        query_vector = self.embedder.embed_text(query_text)
        raw_results: List[Tuple[Dict, float]] = self.store.search(query_vector, k=top_k)
        entities = self._extract_entities(query_text)
        reranked = sorted(raw_results, key=lambda x: self._boost_score(x[1], x[0], entities))

        return {
            "query": query_text,
            "entities": entities,
            "results": reranked
        }
    