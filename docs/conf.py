"""Sphinx configuration for Champollion Pipeline documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "Champollion Pipeline"
copyright = "2024, CEA"
author = "CEA"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

html_theme = "furo"

myst_enable_extensions = ["colon_fence", "deflist"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_static_path = ["_static"]
