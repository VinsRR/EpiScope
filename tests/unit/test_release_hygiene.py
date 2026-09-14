from pathlib import Path

from scripts.clean_release_artifacts import clean_release_artifacts


def test_clean_release_artifacts_removes_only_generated_targets(tmp_path: Path) -> None:
    generated = [
        tmp_path / "build",
        tmp_path / "dist",
        tmp_path / "src" / "epilens.egg-info",
        tmp_path / "src" / "old_name.egg-info",
    ]
    for directory in generated:
        directory.mkdir(parents=True)
        (directory / "generated.txt").write_text("generated", encoding="utf-8")

    source_file = tmp_path / "src" / "epilens" / "__init__.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("__version__ = 'test'\n", encoding="utf-8")

    removed = clean_release_artifacts(tmp_path)

    assert set(removed) == set(generated)
    assert all(not path.exists() for path in generated)
    assert source_file.read_text(encoding="utf-8") == "__version__ = 'test'\n"
