from pathlib import Path


def get_project_root() -> Path:
    """Find the project root by looking for pyproject.toml."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find project root (no pyproject.toml found)")


def get_data_path(filename: str) -> Path:
    """Get the path to a file in the data directory."""
    return get_project_root() / "data" / filename
