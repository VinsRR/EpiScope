from abc import ABC, abstractmethod
from typing import List
from episcope.schemas import SearchResult

# JinaAI Rerankers can be useful for this: 
# - https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual 
# - https://jina.ai/reranker/

class Reranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    def rerank(self, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """Rerank a list of search results."""
        pass

class ResultRanker(Reranker):
    """Ranks and deduplicates search results."""
    
    @staticmethod
    def calculate_rank_score(result: SearchResult, query_weight: float = 1.0) -> float:
        """Calculate ranking score for a search result."""
        base_score = result.similarity_score * query_weight
        
        # Boost for artifacts
        artifact_bonus = 0.0
        if result.artifacts.get("urls"):
            artifact_bonus += 2.0
        if result.artifacts.get("dois"):
            artifact_bonus += 1.8
        if result.artifacts.get("accessions"):
            artifact_bonus += 1.5
        
        # Boost for availability terms
        availability_bonus = result.artifacts.get("availability_score", 0) * 0.5
        
        return base_score + artifact_bonus + availability_bonus
    
    def rerank(self, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """Deduplicate by text similarity and rank by score."""
        if not results:
            return []
        
        # Calculate rank scores
        for result in results:
            result.rank_score = self.calculate_rank_score(result)
        
        # Simple deduplication by normalized text signature
        unique_results = {}
        for result in results:
            # Create signature from first 200 chars of normalized text
            signature = result.artifacts.get("normalized_text", result.text)[:200].lower().strip()
            
            if signature not in unique_results or result.rank_score > unique_results[signature].rank_score:
                unique_results[signature] = result
        
        # Sort by rank score and return top-k
        final_results = list(unique_results.values())
        final_results.sort(key=lambda x: x.rank_score, reverse=True)
        
        return final_results[:top_k]
