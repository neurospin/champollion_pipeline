"""Sphinx configuration for Champollion Pipeline documentation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

project = "champollion_pipeline"
copyright = "2026, Neurospin"
author = "Neurospin — Champollion team"
release = "0.2.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

html_theme = "furo"
html_static_path = ["_static"]

myst_enable_extensions = ["colon_fence", "deflist"]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "workflow.md"]

autodoc_mock_imports = [
    "champollion_utils",
    "contrastive",
    "soma",
    "numpy",
    "pandas",
    "joblib",
    "deep_folding",
    "anatomist",
    "huggingface_hub",
    "torch",
    "matplotlib",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
intersphinx_timeout = 5
