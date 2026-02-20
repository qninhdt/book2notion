"""Shared project paths and book discovery helpers."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BOOKS_ROOT = REPO_ROOT / "books"
OUTPUTS_ROOT = REPO_ROOT / "outputs"
PROMPTS_ROOT = REPO_ROOT / "prompts"


def list_output_book_dirs(book=None):
    """Return sorted output book directories, optionally filtered by name."""
    if not OUTPUTS_ROOT.exists():
        return []
    if book:
        return [OUTPUTS_ROOT / book]
    return sorted([p for p in OUTPUTS_ROOT.iterdir() if p.is_dir()])
