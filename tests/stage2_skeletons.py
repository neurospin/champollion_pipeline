"""Stand-ins for the volumes aims.read returns on a stage 2 skeleton (soma.aims is a MagicMock in the tests).

Also the fixture that lets a test file run the real skeleton check.
"""

import numpy as np
import pytest

SIDES = "LR"
SKELETON_FILE = "{side}skeleton_{subject}.nii.gz"  # the name Morphologist gives it, e.g. Lskeleton_sub-002.nii.gz
SKELETON_VOXEL = 30  # a simple surface voxel


class SkeletonVolume(np.ndarray):
    """A volume read by aims.read: reachable through numpy and through its ``np`` attribute."""

    @property
    def np(self) -> np.ndarray:  # noqa: V106 - the attribute of an aims.Volume, read by the code under test
        return np.asarray(self)


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_stage2_skeletons():
    """Run the real skeleton check: a test file importing this overrides the conftest stub and lays its skeletons."""


def skeleton_volume(voxel: int = SKELETON_VOXEL) -> SkeletonVolume:
    """A small volume holding ``voxel`` in each position."""
    return np.full((2, 2, 2, 1), voxel, dtype=np.int16).view(SkeletonVolume)
