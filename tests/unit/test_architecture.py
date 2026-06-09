from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "episcope"
ADAPTER_MODULES = {"api", "episcope", "ui"}
REPO_ONLY_MODULES = {"classification", "eval", "scripts", "notebooks"}


def _python_files() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def _absolute_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imports.add(node.module)
    return imports


def _package_part(path: Path) -> str:
    relative = path.relative_to(PACKAGE_ROOT)
    if len(relative.parts) == 1:
        return "__root__"
    return relative.parts[0]


def test_package_code_does_not_import_repo_only_modules() -> None:
    violations = []
    for path in _python_files():
        for imported in _absolute_imports(path):
            root = imported.split(".", 1)[0]
            if root in REPO_ONLY_MODULES:
                violations.append(
                    f"{path.relative_to(PACKAGE_ROOT)} imports {imported}"
                )

    assert violations == []


def test_reusable_package_layers_do_not_import_adapters() -> None:
    violations = []
    for path in _python_files():
        source_part = _package_part(path)
        if source_part in ADAPTER_MODULES:
            continue
        if path.name == "__main__.py":
            # `python -m episcope` entry-point shim; part of the CLI adapter.
            continue
        for imported in _absolute_imports(path):
            parts = imported.split(".")
            if (
                len(parts) >= 2
                and parts[0] == "episcope"
                and parts[1] in ADAPTER_MODULES
            ):
                violations.append(
                    f"{path.relative_to(PACKAGE_ROOT)} imports {imported}"
                )

    assert violations == []


def test_schemas_stay_independent_of_runtime_layers() -> None:
    forbidden = {
        "api",
        "clients",
        "db",
        "episcope",
        "rag",
        "services",
        "ui",
        "vectordb",
        "workflows",
    }
    violations = []
    for path in (PACKAGE_ROOT / "schemas").rglob("*.py"):
        for imported in _absolute_imports(path):
            parts = imported.split(".")
            if len(parts) >= 2 and parts[0] == "episcope" and parts[1] in forbidden:
                violations.append(
                    f"{path.relative_to(PACKAGE_ROOT)} imports {imported}"
                )

    assert violations == []


def test_rag_layer_does_not_import_workflows() -> None:
    violations = []
    for path in (PACKAGE_ROOT / "rag").rglob("*.py"):
        for imported in _absolute_imports(path):
            if imported == "episcope.workflows" or imported.startswith(
                "episcope.workflows."
            ):
                violations.append(
                    f"{path.relative_to(PACKAGE_ROOT)} imports {imported}"
                )

    assert violations == []
