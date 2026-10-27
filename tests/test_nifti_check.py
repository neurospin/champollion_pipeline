#!/usr/bin/env python3
"""
Unit tests of the NIfTI input check (src/champollion_pipeline/utils/nifti_check.py).

Cases the stage-1 REQ tests do not reach (TASK-NTUI-BDRABCZUK-18A7A8483C9F):
NIfTI-2, big-endian NIfTI-1, a gzip stream cut in the voxel data, a path that
cannot be opened, and two failed checks in their fixed order.
"""

import gzip
import struct

from champollion_pipeline.utils.nifti_check import (
    INVALID_GEOMETRY,
    NOT_3D,
    TRUNCATED,
    UNREADABLE_HEADER,
    check_nifti_input,
)

_SHAPE = (4, 4, 4)


def _nifti1(order="<", dim=(3, *_SHAPE, 1, 1, 1, 1), pixdim=(1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0), n_voxels=64):
    """Return a single-file NIfTI-1 image (uint8 voxels) in byte order ``order``."""
    header = bytearray(348)
    struct.pack_into(f"{order}i", header, 0, 348)
    struct.pack_into(f"{order}8h", header, 40, *dim)
    struct.pack_into(f"{order}h", header, 72, 8)
    struct.pack_into(f"{order}8f", header, 76, *pixdim)
    struct.pack_into(f"{order}f", header, 108, 352.0)
    header[344:348] = b"n+1\x00"
    return bytes(header) + b"\x00" * 4 + b"\x01" * n_voxels


def _nifti2(order="<"):
    """Return a single-file NIfTI-2 image of shape _SHAPE (uint8 voxels)."""
    header = bytearray(540)
    struct.pack_into(f"{order}i", header, 0, 540)
    header[4:12] = b"n+2\x00\r\n\x1a\n"
    struct.pack_into(f"{order}h", header, 14, 8)
    struct.pack_into(f"{order}8q", header, 16, 3, *_SHAPE, 1, 1, 1, 1)
    struct.pack_into(f"{order}8d", header, 104, 1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0)
    struct.pack_into(f"{order}q", header, 168, 544)
    return bytes(header) + b"\x00" * 4 + b"\x01" * 64


def test_nifti2_passes(tmp_path):
    path = tmp_path / "sub-01.nii"
    path.write_bytes(_nifti2())
    assert check_nifti_input(path) == ()


def test_big_endian_nifti1_passes(tmp_path):
    path = tmp_path / "sub-01.nii"
    path.write_bytes(_nifti1(order=">"))
    assert check_nifti_input(path) == ()


def test_gzip_passes(tmp_path):
    path = tmp_path / "sub-01.nii.gz"
    path.write_bytes(gzip.compress(_nifti1()))
    assert check_nifti_input(path) == ()


def test_gzip_cut_in_voxel_data_is_truncated(tmp_path):
    path = tmp_path / "sub-01.nii.gz"
    data = gzip.compress(_nifti1(dim=(3, 100, 100, 20, 1, 1, 1, 1), n_voxels=200_000), compresslevel=0)
    path.write_bytes(data[: len(data) // 2])
    assert check_nifti_input(path) == (TRUNCATED,)


def test_directory_path_is_unreadable(tmp_path):
    assert check_nifti_input(tmp_path) == (UNREADABLE_HEADER,)


def test_header_image_pair_magic_is_unreadable(tmp_path):
    path = tmp_path / "sub-01.nii"
    path.write_bytes(_nifti1().replace(b"n+1\x00", b"ni1\x00"))
    assert check_nifti_input(path) == (UNREADABLE_HEADER,)


def test_two_failed_checks_in_fixed_order(tmp_path):
    path = tmp_path / "sub-01.nii"
    path.write_bytes(_nifti1(dim=(4, *_SHAPE, 2, 1, 1, 1), pixdim=(1.0, 0.0, 1.0, 1.0, 0, 0, 0, 0), n_voxels=128))
    assert check_nifti_input(path) == (NOT_3D, INVALID_GEOMETRY)
