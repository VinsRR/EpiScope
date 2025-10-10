from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import json

from ..utils.processing_utils import TextProcessor
from .specialized_retriever import RAGQuerier

@dataclass
class SearchResult:
    """Represents a search result chunk with metadata."""
    id: str
    text: str
    section_type: str = "other"
    title: str = ""
    similarity_score: float = 0.0
    source: str = ""  # semantic, keyword, context
    artifacts: Dict[str, Any] = field(default_factory=dict)
    rank_score: float = 0.0

class ChunkSearcher:
    """Handles different search strategies for finding relevant chunks."""
    
    def __init__(self, config, querier: RAGQuerier, text_processor: TextProcessor):
        self.config = config
        self.querier = querier
        self.text_processor = text_processor
    
    def search_semantic(self, queries: List[Tuple[str, float, str]], 
                       index_path: str, chunks_path: str) -> List[SearchResult]:
        """Perform semantic search using weighted queries."""
        if not self.config.use_semantic_search:
            return []
        
        results = []
        seen_ids = set()
        
        for query, weight, reason in queries:
            try:
                chunks = self.querier.query_structured_chunks(
                    query=query,
                    index_path=index_path,
                    chunks_path=chunks_path,
                    top_k=self.config.top_k_semantic,
                    similarity_threshold=self.config.similarity_threshold
                )
                
                for chunk in chunks:
                    chunk_id = str(chunk.get("id", ""))
                    if chunk_id in seen_ids:
                        continue
                    
                    artifacts = self.text_processor.extract_artifacts(chunk.get("text", ""))
                    
                    result = SearchResult(
                        id=chunk_id,
                        text=chunk.get("text", ""),
                        section_type=chunk.get("section_type", "other"),
                        title=chunk.get("title", ""),
                        similarity_score=chunk.get("similarity_score", 0.0),
                        source=f"semantic_{reason}",
                        artifacts=artifacts
                    )
                    
                    results.append(result)
                    seen_ids.add(chunk_id)
                    
            except Exception as e:
                logger.debug(f"Semantic search failed for query '{query}': {e}")
        
        return results
    
    def search_keyword(self, chunks_path: str) -> List[SearchResult]:
        """Perform keyword-based search."""
        if not self.config.use_keyword_search:
            return []
        
        keywords = ["data availability", "available", "repository", "accession", 
                   "deposited", "supplementary", "github", "zenodo"]
        
        try:
            chunks = self.querier.query_keyword_chunks(
                chunks_path=chunks_path,
                keywords=keywords,
                top_k=self.config.top_k_keyword
            )
            
            results = []
            for chunk in chunks:
                artifacts = self.text_processor.extract_artifacts(chunk.get("text", ""))
                
                result = SearchResult(
                    id=str(chunk.get("id", "")),
                    text=chunk.get("text", ""),
                    section_type=chunk.get("section_type", "other"),
                    title=chunk.get("title", ""),
                    similarity_score=0.0,
                    source="keyword",
                    artifacts=artifacts
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.debug(f"Keyword search failed: {e}")
            return []
    
    def extract_context_snippets(self, chunks_path: str) -> List[SearchResult]:
        """Extract context snippets around availability terms."""
        if not self.config.use_context_extraction:
            return []
        
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
        except Exception as e:
            logger.debug(f"Failed to read chunks for context extraction: {e}")
            return []
        
        results = []
        snippet_id = 0
        
        for chunk_idx, chunk in enumerate(chunks):
            text = chunk.get("text", chunk.get("content", ""))
            if not text:
                continue
            
            normalized = self.text_processor.normalize_text(text)
            sentences = self.text_processor.SENTENCE_SPLIT.split(normalized)
            
            for sent_idx, sentence in enumerate(sentences):
                artifacts = self.text_processor.extract_artifacts(sentence)
                
                # Only keep sentences with availability indicators or artifacts
                if artifacts["availability_score"] > 0 or artifacts["has_artifacts"]:
                    # Create context window (sentence before and after)
                    context_sentences = []
                    if sent_idx > 0:
                        context_sentences.append(sentences[sent_idx - 1])
                    context_sentences.append(sentence)
                    if sent_idx < len(sentences) - 1:
                        context_sentences.append(sentences[sent_idx + 1])
                    
                    context_text = " ".join(context_sentences).strip()
                    
                    result = SearchResult(
                        id=f"context_{snippet_id}",
                        text=context_text,
                        section_type=chunk.get("section_type", "other"),
                        title=chunk.get("title", ""),
                        similarity_score=0.0,
                        source="context",
                        artifacts=artifacts
                    )
                    
                    results.append(result)
                    snippet_id += 1
        
        return results
