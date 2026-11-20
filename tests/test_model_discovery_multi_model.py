"""Model discovery over several models per region (TASK-MULTIMODEL-BDRABCZUK-762D497260FE).

find_region_models(models_path) returns one model entry per region model directory, read through the attributes
``path``, ``region``, ``model_id`` and ``label``. The model id is the POSIX path between the region folder (the
nearest ``*_left`` / ``*_right`` directory at or above the model directory, below the models path) and the model
directory; the label (logs, QC, provenance) is ``<region>/<model id>`` for a non-empty model id, else the region
name.

Requirements (wording in the ledger, ``alm get <id>``): the entry set, the model id, the empty model id (model
directory is the region folder; no side-suffixed folder), the sort order, the same-id refusal and the two
label rules. Each class below names its Requirement by title.
"""

from pathlib import Path

import pytest

from champollion_pipeline.models.checkpoint import find_region_model_dirs, find_region_models, resolve_region_name

pytestmark = pytest.mark.unit  # noqa: V107

HF_FOLD = "name08-32-57_200"


def make_model(models_path: Path, *parts: str) -> str:
    """Create models_path/<parts>/logs/ and return the model directory as discovery reports it."""
    model_dir = models_path.joinpath(*parts)
    (model_dir / "logs").mkdir(parents=True)
    return str(model_dir)


def entries_by_path(models_path: Path) -> dict:
    """find_region_models(models_path) keyed by model directory."""
    return {entry.path: entry for entry in find_region_models(str(models_path))}


def mixed_layout(models_path: Path) -> list[str]:
    """A models path mixing flat, nested multi-model, HF cache and side-less layouts."""
    return [
        make_model(models_path, "FCLp-subsc-FCLa-INSULA_left"),
        make_model(models_path, "STs_right", "run_3"),
        make_model(models_path, "STs_right", "exp", "v2"),
        make_model(models_path, "OCCIPITAL_left", HF_FOLD),
        make_model(models_path, "misc", "plain_model"),
    ]


class TestOneEntryPerModelDirectory:
    """Model discovery: one model entry per region model directory."""

    def test_entry_paths_are_the_discovered_model_directories(self, tmp_path):
        """Exactly one entry per directory find_region_model_dirs returns, no more, no fewer."""
        mixed_layout(tmp_path)

        paths = [entry.path for entry in find_region_models(str(tmp_path))]

        assert sorted(paths) == find_region_model_dirs(str(tmp_path))

    def test_entry_region_is_resolve_region_name_of_its_directory(self, tmp_path):
        """Each entry holds the region name resolve_region_name gives for its directory."""
        mixed_layout(tmp_path)

        entries = entries_by_path(tmp_path)

        assert {path: entry.region for path, entry in entries.items()} == {
            path: resolve_region_name(path, str(tmp_path)) for path in find_region_model_dirs(str(tmp_path))
        }

    def test_variant_folders_above_the_region_keep_one_entry_each(self, tmp_path):
        """Two variants of one region with distinct ids: two entries of that region."""
        first = make_model(tmp_path, "variant_a", "STs_right", "run_1")
        second = make_model(tmp_path, "variant_b", "STs_right", "run_2")

        entries = entries_by_path(tmp_path)

        assert sorted(entries) == sorted([first, second])
        assert {entry.region for entry in entries.values()} == {"STs_right"}


class TestModelIdIsPathFromRegionFolder:
    """Model id: path from the region folder to the model folder."""

    @pytest.mark.parametrize(
        ("parts", "expected_id"),
        [
            (("STs_right", "run_3"), "run_3"),
            (("STs_right", "exp", "v2"), "exp/v2"),
            (("variant", "STs_right", "a", "b", "c"), "a/b/c"),
            (("STs_right", "any name with spaces"), "any name with spaces"),
        ],
        ids=["one-level", "two-levels", "variant-above-region", "free-name"],
    )
    def test_model_id_is_posix_path_below_region_folder(self, tmp_path, parts, expected_id):
        """The id is the POSIX relative path from the *_left/*_right folder down to the model directory."""
        model_dir = make_model(tmp_path, *parts)

        entry = entries_by_path(tmp_path)[model_dir]

        assert (entry.region, entry.model_id) == ("STs_right", expected_id)

    def test_hf_cache_fold_hash_is_one_model_with_hash_id(self, tmp_path):
        """HF cache <Region>_<side>/<hash>/logs: one model of the region, model id <hash>."""
        model_dir = make_model(tmp_path, "Champollion_V1", "OCCIPITAL_left", HF_FOLD)

        entries = find_region_models(str(tmp_path))

        assert [(entry.path, entry.region, entry.model_id) for entry in entries] == [
            (model_dir, "OCCIPITAL_left", HF_FOLD)
        ]

    def test_nearest_side_suffixed_folder_is_the_region_folder(self, tmp_path):
        """With two *_left folders on the path, the nearest one to the model is the region folder."""
        model_dir = make_model(tmp_path, "outer_left", "STs_right", "run_1")

        entry = entries_by_path(tmp_path)[model_dir]

        assert (entry.region, entry.model_id) == ("STs_right", "run_1")

    def test_model_id_uses_forward_slashes(self, tmp_path):
        """The id separator is '/', whatever os.sep is."""
        model_dir = make_model(tmp_path, "STs_right", "exp", "v2")

        entry = entries_by_path(tmp_path)[model_dir]

        assert entry.model_id == "exp/v2"


class TestEmptyModelIdInRegionFolder:
    """Model id empty when the model folder is the region folder."""

    @pytest.mark.parametrize(
        "parts",
        [("STs_right",), ("FCLp-subsc-FCLa-INSULA_left",), ("variant", "STs_right")],
        ids=["right", "left", "below-variant"],
    )
    def test_logs_directly_in_region_folder_gives_empty_id(self, tmp_path, parts):
        """logs/ directly in the *_left/*_right folder: model id ''."""
        model_dir = make_model(tmp_path, *parts)

        entry = entries_by_path(tmp_path)[model_dir]

        assert (entry.region, entry.model_id) == (parts[-1], "")


class TestEmptyModelIdWithoutSideFolder:
    """Model id empty when no side-suffixed ancestor exists."""

    @pytest.mark.parametrize("parts", [("plain_model",), ("misc", "plain_model")], ids=["top", "nested"])
    def test_model_without_side_suffixed_folder_gives_empty_id(self, tmp_path, parts):
        """No *_left/*_right folder at or above the model directory: model id '', region = directory name."""
        model_dir = make_model(tmp_path, *parts)

        entry = entries_by_path(tmp_path)[model_dir]

        assert (entry.region, entry.model_id) == ("plain_model", "")

    def test_side_suffixed_models_path_itself_is_not_a_region_folder(self, tmp_path):
        """A models path named *_left is above the walk: the model below it has no region folder."""
        models_path = tmp_path / "models_left"
        model_dir = make_model(models_path, "misc", "plain_model")

        entry = entries_by_path(models_path)[model_dir]

        assert (entry.region, entry.model_id) == ("plain_model", "")


class TestEntriesSortedByRegionThenModelId:
    """Model entries sorted by region name, then model id."""

    def test_entries_sorted_by_region_then_model_id(self, tmp_path):
        """Order is (region, model id), not the order of the directory paths."""
        make_model(tmp_path, "a_variant", "Z_left", "run_1")
        make_model(tmp_path, "b_variant", "A_left", "run_3")
        make_model(tmp_path, "c_variant", "A_left", "exp", "v2")
        make_model(tmp_path, "d_variant", "A_left", "run_10")
        make_model(tmp_path, "M_right")

        keys = [(entry.region, entry.model_id) for entry in find_region_models(str(tmp_path))]

        assert keys == [
            ("A_left", "exp/v2"),
            ("A_left", "run_10"),
            ("A_left", "run_3"),
            ("M_right", ""),
            ("Z_left", "run_1"),
        ]

    def test_order_does_not_depend_on_creation_order(self, tmp_path):
        """Two trees with the same models created in opposite orders give the same sequence."""
        layout = [("B_left", "run_2"), ("A_right", "run_1"), ("B_left", "run_1"), ("A_right",)]
        for index, parts_list in enumerate((layout, list(reversed(layout)))):
            for parts in parts_list:
                make_model(tmp_path / str(index), *parts)

        first = [(e.region, e.model_id) for e in find_region_models(str(tmp_path / "0"))]
        second = [(e.region, e.model_id) for e in find_region_models(str(tmp_path / "1"))]

        assert first == second == [("A_right", ""), ("B_left", "run_1"), ("B_left", "run_2")]


class TestSameModelIdRefused:
    """Two models of one region with one model id refused."""

    @pytest.mark.parametrize(
        "tail",
        [(), ("run_1",), ("exp", "v2")],
        ids=["empty-id", "one-level-id", "nested-id"],
    )
    def test_same_region_same_id_raises_value_error_naming_both(self, tmp_path, tail):
        """Variant folders above the region folder giving one (region, id) twice: ValueError naming both dirs."""
        first = make_model(tmp_path, "variant_a", "STs_right", *tail)
        second = make_model(tmp_path, "variant_b", "STs_right", *tail)

        with pytest.raises(ValueError) as raised:
            find_region_models(str(tmp_path))

        assert first in str(raised.value)
        assert second in str(raised.value)

    def test_same_id_in_distinct_regions_is_accepted(self, tmp_path):
        """One id in two regions is two models, not a clash."""
        make_model(tmp_path, "STs_right", "run_1")
        make_model(tmp_path, "STs_left", "run_1")

        keys = [(entry.region, entry.model_id) for entry in find_region_models(str(tmp_path))]

        assert keys == [("STs_left", "run_1"), ("STs_right", "run_1")]

    def test_distinct_ids_in_one_region_are_accepted(self, tmp_path):
        """Two ids in one region under two variant folders are two models, not a clash."""
        make_model(tmp_path, "variant_a", "STs_right", "run_1")
        make_model(tmp_path, "variant_b", "STs_right", "run_2")

        keys = [(entry.region, entry.model_id) for entry in find_region_models(str(tmp_path))]

        assert keys == [("STs_right", "run_1"), ("STs_right", "run_2")]


class TestNonEmptyModelIdLabel:
    """Model label of a non-empty model id: <Region>/<id>."""

    def test_models_of_a_multi_model_region_are_labelled_region_slash_id(self, tmp_path):
        """Two models of STs_right: labels STs_right/<id>."""
        make_model(tmp_path, "STs_right", "run_3")
        make_model(tmp_path, "STs_right", "exp", "v2")

        labels = [entry.label for entry in find_region_models(str(tmp_path))]

        assert labels == ["STs_right/exp/v2", "STs_right/run_3"]

    @pytest.mark.parametrize(
        ("parts", "expected_label"),
        [
            (("STs_right", "run_3"), "STs_right/run_3"),
            (("Champollion_V1", "STs_right", HF_FOLD), f"STs_right/{HF_FOLD}"),
        ],
        ids=["named-run", "hf-cache"],
    )
    def test_only_model_of_a_region_with_an_id_is_labelled_region_slash_id(self, tmp_path, parts, expected_label):
        """One model in the region with a non-empty id (named run, HF cache hash): label <region>/<id>."""
        make_model(tmp_path, *parts)

        labels = [entry.label for entry in find_region_models(str(tmp_path))]

        assert labels == [expected_label]

    def test_single_model_region_beside_multi_model_region_is_labelled_region_slash_id(self, tmp_path):
        """The region count plays no part: every model with a non-empty id carries it in its label."""
        make_model(tmp_path, "STs_right", "run_1")
        make_model(tmp_path, "STs_right", "run_2")
        make_model(tmp_path, "OCCIPITAL_left", HF_FOLD)

        labels = {(entry.region, entry.model_id): entry.label for entry in find_region_models(str(tmp_path))}

        assert labels == {
            ("OCCIPITAL_left", HF_FOLD): f"OCCIPITAL_left/{HF_FOLD}",
            ("STs_right", "run_1"): "STs_right/run_1",
            ("STs_right", "run_2"): "STs_right/run_2",
        }


class TestEmptyModelIdLabel:
    """Model label of an empty model id: the region name."""

    def test_empty_id_beside_named_model_is_labelled_with_region_name(self, tmp_path):
        """<p>/v1/STs_right/logs (id '') beside <p>/v2/STs_right/run_1/logs: labels STs_right, STs_right/run_1."""
        make_model(tmp_path, "v1", "STs_right")
        make_model(tmp_path, "v2", "STs_right", "run_1")

        labels = [(entry.model_id, entry.label) for entry in find_region_models(str(tmp_path))]

        assert labels == [("", "STs_right"), ("run_1", "STs_right/run_1")]

    @pytest.mark.parametrize("parts", [("STs_right",), ("variant", "STs_right")], ids=["flat", "below-variant"])
    def test_only_model_of_a_region_in_its_region_folder_is_labelled_with_region_name(self, tmp_path, parts):
        """logs/ directly in the region folder, alone in its region: id '' and label = region name."""
        make_model(tmp_path, *parts)

        labels = [(entry.model_id, entry.label) for entry in find_region_models(str(tmp_path))]

        assert labels == [("", "STs_right")]

    def test_side_less_model_is_labelled_with_its_directory_name(self, tmp_path):
        """No *_left/*_right folder: id '' and label = the directory name resolve_region_name gives."""
        make_model(tmp_path, "misc", "plain_model")

        labels = [entry.label for entry in find_region_models(str(tmp_path))]

        assert labels == ["plain_model"]
