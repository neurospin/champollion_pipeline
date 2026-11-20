#!/usr/bin/env python3
"""
Per-model checkpoint epoch specification (MULTIMODEL epic, Task T1b).

User decision 2026-11-20: --checkpoint-epoch is repeatable. A bare ``N`` is the
default epoch of every model (at most one bare value), ``<id>=N`` sets the epoch of
one model id (split at the last '=' sign), applied to that id in every region; an id
matching no discovered model is refused; the default stays 80.

These tests exercise the pure functions of ``champollion_pipeline.models.epoch_spec``:
``parse_epoch_specs(values, default) -> (default_epoch, by_id)`` and
``resolve_model_epochs(default_epoch, by_id, model_ids) -> {model id: epoch}``.
No parser, no file system, no model. The module is imported inside each test so
that each test fails on its own while the module does not exist yet.
"""

import importlib

import pytest
from hypothesis import given
from hypothesis import strategies as st

pytestmark = pytest.mark.unit  # noqa: V107

MODULE = "champollion_pipeline.models.epoch_spec"


def epoch_spec():
    """Return the module under test (imported per test, see the module docstring)."""
    return importlib.import_module(MODULE)


def parse(values):
    """Call parse_epoch_specs with the default epoch 80 passed by the caller."""
    return epoch_spec().parse_epoch_specs(values, 80)


def resolve(default_epoch, by_id, model_ids):
    """Call resolve_model_epochs and return its result as a plain dict."""
    return dict(epoch_spec().resolve_model_epochs(default_epoch, by_id, model_ids))


def parse_refusal(values) -> str:
    """Return the message of the ValueError parse_epoch_specs raises for values."""
    with pytest.raises(ValueError) as excinfo:
        parse(values)
    return str(excinfo.value)


def resolve_refusal(default_epoch, by_id, model_ids) -> str:
    """Return the message of the ValueError resolve_model_epochs raises."""
    with pytest.raises(ValueError) as excinfo:
        resolve(default_epoch, by_id, model_ids)
    return str(excinfo.value)


class TestPerModelEpochChoice:
    """REQ-MULTIMODEL-7A5BD71C6EE7 (stakeholder): one model id's epoch apart from the default."""

    def test_one_model_id_gets_its_epoch_and_the_others_the_default(self):
        default_epoch, by_id = parse(["70", "run_3=60"])
        assert resolve(default_epoch, by_id, ["run_3", "run_4", "exp/v2"]) == {
            "run_3": 60,
            "run_4": 70,
            "exp/v2": 70,
        }


class TestParseDefaultEpoch:
    """REQ-MULTIMODEL-8EA76D2750CD (bare value) and -67ADD418B9D8 (default 80)."""

    def test_bare_value_sets_the_default_epoch(self):
        default_epoch, _ = parse(["run_3=60", "70"])
        assert default_epoch == 70

    def test_empty_list_gives_default_epoch_80(self):
        default_epoch, by_id = parse([])
        assert (default_epoch, dict(by_id)) == (80, {})

    def test_only_id_values_give_default_epoch_80(self):
        default_epoch, _ = parse(["run_3=60", "exp/v2=0"])
        assert default_epoch == 80


class TestParsePerIdEpoch:
    """REQ-MULTIMODEL-6E5211100000: '<id>=N', split at the last '=', maps <id> to N."""

    def test_id_value_maps_the_id_to_its_epoch(self):
        _, by_id = parse(["run_3=60", "run_4=0"])
        assert dict(by_id) == {"run_3": 60, "run_4": 0}

    def test_nested_model_id_is_kept_whole(self):
        _, by_id = parse(["exp/v2=55"])
        assert dict(by_id) == {"exp/v2": 55}

    def test_value_is_split_at_the_last_equals_sign(self):
        _, by_id = parse(["lr=0.1=60"])
        assert dict(by_id) == {"lr=0.1": 60}

    def test_bare_value_adds_no_model_id(self):
        _, by_id = parse(["70"])
        assert dict(by_id) == {}


class TestParseInvalidEpoch:
    """REQ-MULTIMODEL-78EA37174593 (bare value) and -19987BD20508 ('<id>=N')."""

    @pytest.mark.parametrize("value", ["-1", "abc", "8.5", "1e2"])
    def test_bare_value_not_a_non_negative_integer_is_refused(self, value):
        assert value in parse_refusal([value])

    @pytest.mark.parametrize("value", ["run_3=-1", "run_3=abc", "run_3=", "run_3=8.5"])
    def test_id_value_epoch_not_a_non_negative_integer_is_refused(self, value):
        assert value in parse_refusal(["70", value])


class TestParseRefusals:
    """REQ-MULTIMODEL-3F5C2D64922D (id twice), -57E3F9474546 (empty id), -6C80C4EC710E (two bare)."""

    def test_model_id_given_twice_is_refused_naming_it(self):
        assert "run_3" in parse_refusal(["run_3=60", "run_4=50", "run_3=70"])

    def test_empty_model_id_is_refused_naming_the_value(self):
        assert "=60" in parse_refusal(["70", "=60"])

    def test_two_bare_values_are_refused_naming_both(self):
        message = parse_refusal(["70", "run_3=60", "65"])
        assert "70" in message
        assert "65" in message


class TestResolveModelEpochs:
    """REQ-MULTIMODEL-C3C5EB074499 (id in the mapping) and -457E595CD603 (default epoch)."""

    def test_model_id_in_the_mapping_gets_its_epoch(self):
        assert resolve(80, {"run_3": 60}, ["run_3"]) == {"run_3": 60}

    def test_model_id_absent_from_the_mapping_gets_the_default_epoch(self):
        assert resolve(75, {"run_3": 60}, ["run_3", "run_4"]) == {"run_3": 60, "run_4": 75}

    def test_empty_model_id_gets_the_default_epoch(self):
        assert resolve(75, {}, [""]) == {"": 75}


class TestResolveUnknownModelId:
    """REQ-MULTIMODEL-430456A93945 (names the id) and -9FFD836D8D4B (lists the known ids)."""

    def test_unknown_model_id_is_refused_naming_it(self):
        assert "run_9" in resolve_refusal(80, {"run_9": 60}, ["run_3", "run_4"])

    def test_refusal_lists_the_known_model_ids(self):
        message = resolve_refusal(80, {"run_9": 60}, ["run_3", "exp/v2", ""])
        assert "run_3" in message
        assert "exp/v2" in message


class TestSingleEpochWithoutIdValues:
    """REQ-MULTIMODEL-E7CBCFCF6F71: no '<id>=N' value gives one same epoch to every model."""

    @given(
        values=st.one_of(st.just([]), st.integers(min_value=0, max_value=10_000).map(lambda n: [str(n)])),
        model_ids=st.lists(st.text(alphabet="abc_/0123", max_size=8), max_size=6),
    )
    def test_values_without_id_give_one_epoch_to_each_model(self, values, model_ids):
        default_epoch, by_id = parse(values)
        epochs = resolve(default_epoch, by_id, model_ids)
        assert set(epochs) == set(model_ids)
        assert len(set(epochs.values())) <= 1
