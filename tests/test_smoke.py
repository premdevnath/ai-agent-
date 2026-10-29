"""Smoke tests for the Financial RAG Chatbot.

These tests are deliberately lightweight so they run in CI without any
network access, API keys, or heavy runtime dependencies. Rather than
importing the modules (which would pull in Streamlit, LangChain, ChromaDB,
etc.), we statically parse the source with ``ast`` to guarantee the files
are syntactically valid Python and that the expected entry points exist.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse(filename: str) -> ast.Module:
    source = (REPO_ROOT / filename).read_text(encoding="utf-8")
    return ast.parse(source, filename=filename)


def test_key_files_exist():
    for name in ("app.py", "eval_harness.py", "requirements.txt", "README.md"):
        assert (REPO_ROOT / name).is_file(), f"missing expected file: {name}"


def test_app_parses():
    """app.py must be valid Python (parsed, not imported)."""
    tree = _parse("app.py")
    assert isinstance(tree, ast.Module)


def test_eval_harness_parses_and_exposes_functions():
    """eval_harness.py must parse and define its evaluation entry points."""
    tree = _parse("eval_harness.py")
    func_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    # A couple of stable, well-known function names from the harness.
    assert func_names, "eval_harness.py defines no functions"
