
import os
file_path = "src/episcope/pipelines/precision_miner_pipeline.py"
with open(file_path, 'r') as f:
    content = f.read()
# remove the misplaced import and method
content = content.replace("from episcope.rag.chunker import chunk_paper", "")
misplaced_method = """
    def _prepare_index_and_chunks(self, sections, metadata, paper_id: str, base_dir: Path, strategy_name: str):
        """Prepare index and chunks files."""
        index_path, chunks_path = "", ""
        if self.indexer:
            try:
                index_path, chunks_path = self.indexer.create_paper_index(
                    sections=sections,
                    metadata=metadata,
                    output_dir=str(self.index_dir / strategy_name),
                    paper_id=paper_id
                )
            except Exception as e:
                logger.warning(f"Index creation failed for {paper_id}: {e}")
        
        return index_path, chunks_path
"
content = content.replace(misplaced_method, "")
with open(file_path, 'w') as f:
    f.write(content)
