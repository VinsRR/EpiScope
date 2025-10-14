import os
from pathlib import Path
import json
import faiss
import numpy as np

# It's safer to read the file and append the text directly
with open('/Users/vins/Documents/Projects/EpiScope/episcope/src/episcope/rag/indexing/faiss_indexer.py', 'r') as f:
    original_content = f.read()

# Define the new method and imports to add
new_imports = """
from episcope.rag.chunker import chunk_paper
from episcope.utils.data_blueprints import StructuredSection, PaperMetadata
from typing import Tuple
"""
new_method = """
    def create_paper_index(
        self,
        sections: list,
        metadata: dict,
        output_dir: str,
        paper_id: str,
        min_chunk_size: int = 50,
    ) -> Tuple[str, str]:
        """
        Chunks a structured paper, creates a FAISS index, and saves both to disk.
        """
        chunks = chunk_paper(sections, metadata, paper_id, min_chunk_size)
        if not chunks:
            logger.warning(f"No chunks were created for paper {paper_id}, index not generated.")
            return "", ""

        # Index the documents into the given namespace (paper_id)
        self.index_documents(chunks, namespace=paper_id)

        # Save the index and metadata to files
        self.index_dir = Path(output_dir)
        index_path = self.index_dir / f"{paper_id}_structured_index.faiss"
        chunks_path = self.index_dir / f"{paper_id}_structured_chunks.json"

        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata[paper_id], f, indent=2)

        if _HAS_FAISS and paper_id in self._faiss_indices:
            faiss.write_index(self._faiss_indices[paper_id], str(index_path))
        else:
            np.save(str(index_path).replace(".faiss", ".npy"), self._embeddings[paper_id])
        
        return str(index_path), str(chunks_path)
"

# Add imports at the top, avoiding duplicates
existing_imports = []
new_import_lines = []
for line in original_content.splitlines():
    if line.strip().startswith("from") or line.strip().startswith("import"):
        existing_imports.append(line.strip())

for line in new_imports.splitlines():
    if line.strip() and line.strip() not in existing_imports:
        new_import_lines.append(line)

final_content = "\n".join(new_import_lines) + "\n" + original_content + "\n" + new_method

# Write the updated content back to the file
with open('/Users/vins/Documents/Projects/EpiScope/episcope/src/episcope/rag/indexing/faiss_indexer.py', 'w') as f:
    f.write(final_content)