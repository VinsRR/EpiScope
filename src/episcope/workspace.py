from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

WORKSPACE_FILE = "episcope.toml"


@dataclass(frozen=True)
class WorkspaceConfig:
    root: Path
    name: str
    strategy_name: str = "local"
    papers_dir: str = "papers"
    outputs_dir: str = "outputs"
    logs_dir: str = "logs"
    metadata_backend: str = "memory"
    metadata_path: str = "metadata.json"
    index_backend: str = "file"
    index_dir: str = "index"
    loader: str = "unstructured"
    chunker: str = "paragraph"
    min_chunk_size: int = 20
    chunk_size: int = 600
    chunk_overlap: int = 100
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retrieval_mode: str = "dense_only"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "episcope_academic"
    mongo_db_name: str = "episcope_academic_db"
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.5-flash"

    @property
    def config_path(self) -> Path:
        return self.root / WORKSPACE_FILE

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return self.root / path


def create_workspace(
    path: str | Path,
    *,
    name: Optional[str] = None,
    force: bool = False,
) -> WorkspaceConfig:
    root = Path(path).expanduser().resolve()
    config = WorkspaceConfig(root=root, name=name or root.name)
    if config.config_path.exists() and not force:
        raise FileExistsError(f"Workspace already exists at {root}")

    root.mkdir(parents=True, exist_ok=True)
    for directory in (
        config.papers_dir,
        config.index_dir,
        config.outputs_dir,
        config.logs_dir,
    ):
        config.resolve_path(directory).mkdir(parents=True, exist_ok=True)
    write_workspace_config(config)
    return config


def load_workspace(path: str | Path) -> WorkspaceConfig:
    root_or_file = Path(path).expanduser()
    config_path = (
        root_or_file
        if root_or_file.name == WORKSPACE_FILE
        else root_or_file / WORKSPACE_FILE
    )
    if not config_path.exists():
        raise FileNotFoundError(f"No {WORKSPACE_FILE} found at {config_path}")

    data = _read_simple_toml(config_path)
    root = config_path.parent.resolve()
    workspace = data.get("workspace", {})
    storage = data.get("storage", {})
    indexing = data.get("indexing", {})
    retrieval = data.get("retrieval", {})
    server = data.get("server", {})
    llm = data.get("llm", {})
    return WorkspaceConfig(
        root=root,
        name=str(workspace.get("name") or root.name),
        strategy_name=str(workspace.get("strategy_name") or "local"),
        papers_dir=str(workspace.get("papers_dir") or "papers"),
        outputs_dir=str(workspace.get("outputs_dir") or "outputs"),
        logs_dir=str(workspace.get("logs_dir") or "logs"),
        metadata_backend=str(storage.get("metadata_backend") or "memory"),
        metadata_path=str(storage.get("metadata_path") or "metadata.json"),
        index_backend=str(storage.get("index_backend") or "file"),
        index_dir=str(storage.get("index_dir") or "index"),
        loader=str(indexing.get("loader") or "unstructured"),
        chunker=str(indexing.get("chunker") or "paragraph"),
        min_chunk_size=int(indexing.get("min_chunk_size") or 20),
        chunk_size=int(indexing.get("chunk_size") or 600),
        chunk_overlap=int(indexing.get("chunk_overlap") or 100),
        embed_model=str(
            indexing.get("embed_model")
            or "sentence-transformers/all-MiniLM-L6-v2"
        ),
        retrieval_mode=str(retrieval.get("mode") or "dense_only"),
        qdrant_url=str(server.get("qdrant_url") or "http://localhost:6333"),
        qdrant_collection=str(
            server.get("qdrant_collection") or "episcope_academic"
        ),
        mongo_db_name=str(server.get("mongo_db_name") or "episcope_academic_db"),
        llm_provider=str(llm.get("provider") or "gemini"),
        llm_model=str(llm.get("model") or "gemini-2.5-flash"),
    )


def find_workspace(start: str | Path | None = None) -> Optional[WorkspaceConfig]:
    current = Path(start or Path.cwd()).expanduser().resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        config_path = candidate / WORKSPACE_FILE
        if config_path.exists():
            return load_workspace(config_path)
    return None


def write_workspace_config(config: WorkspaceConfig) -> None:
    config.config_path.write_text(_render_workspace_toml(config), encoding="utf-8")


def _render_workspace_toml(config: WorkspaceConfig) -> str:
    return "\n".join(
        [
            "[workspace]",
            f'name = "{_escape(config.name)}"',
            'strategy_name = "local"',
            'papers_dir = "papers"',
            'outputs_dir = "outputs"',
            'logs_dir = "logs"',
            "",
            "[storage]",
            'metadata_backend = "memory"',
            'metadata_path = "metadata.json"',
            'index_backend = "file"',
            'index_dir = "index"',
            "",
            "[indexing]",
            'loader = "unstructured"',
            'chunker = "paragraph"',
            "min_chunk_size = 20",
            "chunk_size = 600",
            "chunk_overlap = 100",
            'embed_model = "sentence-transformers/all-MiniLM-L6-v2"',
            "",
            "[retrieval]",
            'mode = "dense_only"',
            "",
            "[server]",
            'qdrant_url = "http://localhost:6333"',
            'qdrant_collection = "episcope_academic"',
            'mongo_db_name = "episcope_academic_db"',
            "",
            "[llm]",
            'provider = "gemini"',
            'model = "gemini-2.5-flash"',
            "",
        ]
    )


def _read_simple_toml(path: Path) -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {}
    section: Optional[str] = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            data.setdefault(section, {})
            continue
        if section is None or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        data[section][key.strip()] = _parse_value(raw_value.strip())
    return data


def _parse_value(value: str) -> Any:
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    try:
        return int(value)
    except ValueError:
        return value


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
