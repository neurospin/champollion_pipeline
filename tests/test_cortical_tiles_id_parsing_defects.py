"""Evidence that cortical_tiles 7b80c89 still has the ID and path parsing defects the pipeline guards for.

DEF-CORTILES-BDRABCZUK-9C7D2B9265BC (TASK-CORTILES-BDRABCZUK-96603E82E1F7 re-checked it): the dotted-ID guard and the
root-substring guard of run_cortical_tiles.py stay, because these calls still misbehave in the submodule. A test
failing here means cortical_tiles fixed the defect: remove the matching guard then.
"""

import pytest
from cortical_tiles.brainvisa.utils import quality_checks
from cortical_tiles.brainvisa.utils.skip_list import session_key

pytestmark = pytest.mark.unit  # noqa: V107


class TestRootTokensStillLeak:
    @pytest.mark.parametrize(
        ("root_token", "expected_key"),
        [("ses-pilot", "sub-1_ses-pilot"), ("dryrun-2", "sub-1_run-2")],
        ids=["ses", "run"],
    )
    def test_session_key_reads_the_root_of_the_path(self, root_token, expected_key):
        """A BIDS key takes its session part from the root folder of the graph path, not the subject's own."""
        graph = f"/proj/{root_token}/sub-1/t1mri/default/folds/R.arg"

        assert session_key("sub-1", graph, True) == expected_key


class TestDottedIdsStillBroken:
    def test_two_dotted_ids_abort_the_step(self, tmp_path):
        """A.1 and A.2 give one subject key A: the step raises ValueError instead of treating two subjects."""
        for name in ("pre_A.1.nii.gz", "pre_A.2.nii.gz"):
            (tmp_path / name).write_text("")
        target = tmp_path / "target"
        target.mkdir()

        with pytest.raises(ValueError, match="give one subject key 'A'"):
            quality_checks.get_not_processed_files(str(tmp_path), str(target), "pre_")

    def test_a_done_dotted_subject_is_listed_again(self, tmp_path):
        """A.1 with its result A.1.nii.gz present is still listed as not processed (the target is cut at the dot)."""
        (tmp_path / "A.1.nii.gz").write_text("")

        listed = quality_checks.get_not_processed_subjects_dict([{"subject": "A.1"}], str(tmp_path))

        assert listed == [{"subject": "A.1"}]
