#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-5 (champollion-combine) help text and docstring name the versioned source.

Since commit f820f2a, stage 4 writes region embeddings under
<datasets_root>/derivatives/champollion_V1/<masks>/region_embeddings and stage 5
reads them from there, so the user-facing text must name that directory and no
longer the legacy unversioned {dataset}embeddings/ directory.

REQ-EMBVER-BDRABCZUK-1405DBCCD6E2: the help text of the embeddings_source
positional argument of champollion-combine shall contain the literal path
<datasets_root>/derivatives/champollion_V1/<masks>/region_embeddings.

REQ-EMBVER-BDRABCZUK-EC3185C41F06: the help text of the embeddings_source
positional argument of champollion-combine shall not contain the substring
{dataset}embeddings.

REQ-EMBVER-BDRABCZUK-F6F0E7A3FAAA: the PutTogetherEmbeddings class docstring
shall contain the literal path
<datasets_root>/derivatives/champollion_V1/<masks>/region_embeddings.

REQ-EMBVER-BDRABCZUK-E04E098F118F: the PutTogetherEmbeddings class docstring
shall not contain the substring {dataset}embeddings.
"""

import pytest

from champollion_pipeline.put_together_embeddings import PutTogetherEmbeddings

VERSIONED_SOURCE = "<datasets_root>/derivatives/champollion_V1/<masks>/region_embeddings"
LEGACY_SOURCE = "{dataset}embeddings"


@pytest.fixture
def script():
    return PutTogetherEmbeddings()


def _embeddings_source_help(script):
    actions = [a for a in script.parser._actions if a.dest == "embeddings_source"]
    assert len(actions) == 1, "PutTogetherEmbeddings must declare one embeddings_source positional"
    return actions[0].help or ""


def _rendered_help(script, monkeypatch):
    """What `champollion-combine --help` prints, with line wrapping undone."""
    monkeypatch.setenv("COLUMNS", "1000")
    return " ".join(script.parser.format_help().split())


def test_embeddings_source_help_names_versioned_region_embeddings(script):
    help_text = _embeddings_source_help(script)
    assert VERSIONED_SOURCE in help_text, f"embeddings_source help {help_text!r} does not name {VERSIONED_SOURCE}"


def test_combine_help_output_names_versioned_region_embeddings(script, monkeypatch):
    rendered = _rendered_help(script, monkeypatch)
    assert VERSIONED_SOURCE in rendered, f"champollion-combine --help does not name {VERSIONED_SOURCE}:\n{rendered}"


def test_embeddings_source_help_has_no_legacy_dataset_embeddings(script):
    help_text = _embeddings_source_help(script)
    assert LEGACY_SOURCE not in help_text, f"embeddings_source help {help_text!r} still names {LEGACY_SOURCE}"


def test_combine_help_output_has_no_legacy_dataset_embeddings(script, monkeypatch):
    rendered = _rendered_help(script, monkeypatch)
    assert LEGACY_SOURCE not in rendered, f"champollion-combine --help still names {LEGACY_SOURCE}:\n{rendered}"


def test_class_docstring_names_versioned_region_embeddings():
    doc = PutTogetherEmbeddings.__doc__ or ""
    assert VERSIONED_SOURCE in doc, f"PutTogetherEmbeddings docstring {doc!r} does not name {VERSIONED_SOURCE}"


def test_class_docstring_has_no_legacy_dataset_embeddings():
    doc = PutTogetherEmbeddings.__doc__ or ""
    assert LEGACY_SOURCE not in doc, f"PutTogetherEmbeddings docstring {doc!r} still names {LEGACY_SOURCE}"
