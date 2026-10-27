"""Header-level QC of one raw NIfTI input file, standard library only.

check_nifti_input reads the NIfTI-1 or NIfTI-2 header (plain or gzip, either
byte order) and reports which checks fail: unreadable header, not a 3-D
volume, invalid geometry, or truncated voxel data. No nibabel, numpy or soma
import, so it runs in every pixi environment.
"""

import gzip
import math
import os
import struct
import zlib
from typing import BinaryIO

UNREADABLE_HEADER = "unreadable_header"
NOT_3D = "not_3d"
INVALID_GEOMETRY = "invalid_geometry"
TRUNCATED = "truncated"

_GZIP_MAGIC = b"\x1f\x8b"
_HEADER_SIZE = 540
_CHUNK_SIZE = 1 << 20
_READ_ERRORS = (OSError, EOFError, zlib.error, struct.error)


def _read_up_to(stream: BinaryIO, size: int) -> bytes:
    """Read ``size`` bytes, or fewer at end of stream."""
    data = b""
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            break
        data += chunk
    return data


def _parse_header(head: bytes) -> tuple[list[int], int, list[float], float] | None:
    """Return (dim, bitpix, pixdim, vox_offset), or None when ``head`` is no NIfTI-1/2 single-file header."""
    for order in ("<", ">"):
        if len(head) < 4:
            return None
        sizeof_hdr = struct.unpack_from(f"{order}i", head, 0)[0]
        if sizeof_hdr == 348 and len(head) >= 348 and head[344:348] == b"n+1\x00":
            return (
                list(struct.unpack_from(f"{order}8h", head, 40)),
                struct.unpack_from(f"{order}h", head, 72)[0],
                list(struct.unpack_from(f"{order}8f", head, 76)),
                struct.unpack_from(f"{order}f", head, 108)[0],
            )
        if sizeof_hdr == 540 and len(head) >= 540 and head[4:8] == b"n+2\x00":
            return (
                list(struct.unpack_from(f"{order}8q", head, 16)),
                struct.unpack_from(f"{order}h", head, 14)[0],
                list(struct.unpack_from(f"{order}8d", head, 104)),
                float(struct.unpack_from(f"{order}q", head, 168)[0]),
            )
    return None


def _expected_size(dim: list[int], bitpix: int, vox_offset: float) -> float:
    """Return vox_offset + prod(dim[1..nd]) * bitpix / 8, as a float."""
    nd = min(max(dim[0], 0), 7)
    return vox_offset + math.prod(dim[1 : nd + 1]) * bitpix / 8


def _count_gzip_bytes(stream: BinaryIO, already: int, needed: float) -> int:
    """Count decompressed bytes, from ``already``, until ``needed`` is reached or the stream ends or breaks."""
    available = already
    target = math.ceil(needed)
    try:
        while available < target:
            chunk = stream.read(min(_CHUNK_SIZE, target - available))
            if not chunk:
                break
            available += len(chunk)
    except _READ_ERRORS:
        pass
    return available


def _reasons(dim: list[int], pixdim: list[float], expected: float, available: int) -> tuple[str, ...]:
    """Return the failed checks among NOT_3D, INVALID_GEOMETRY, TRUNCATED, in that order."""
    nd = min(max(dim[0], 0), 7)
    reasons = []
    if dim[0] < 3 or math.prod(dim[4 : nd + 1]) > 1:
        reasons.append(NOT_3D)
    if any(dim[i] < 1 or not (math.isfinite(pixdim[i]) and pixdim[i] > 0) for i in (1, 2, 3)):
        reasons.append(INVALID_GEOMETRY)
    if available < expected:
        reasons.append(TRUNCATED)
    return tuple(reasons)


def check_nifti_input(path: str | os.PathLike) -> tuple[str, ...]:
    """Return the failed QC reasons of one NIfTI file; ``()`` when it passes every check.

    ``(UNREADABLE_HEADER,)`` when no NIfTI-1/2 single-file header can be read
    (no other check runs); otherwise the failed subset of
    ``(NOT_3D, INVALID_GEOMETRY, TRUNCATED)`` in that order. Never raises for
    the file's content or for an error while reading it.
    """
    try:
        with open(path, "rb") as handle:
            if _read_up_to(handle, 2) == _GZIP_MAGIC:
                handle.seek(0)
                with gzip.GzipFile(fileobj=handle, mode="rb") as stream:
                    head = _read_up_to(stream, _HEADER_SIZE)
                    header = _parse_header(head)
                    if header is None:
                        return (UNREADABLE_HEADER,)
                    dim, bitpix, pixdim, vox_offset = header
                    expected = _expected_size(dim, bitpix, vox_offset)
                    available = _count_gzip_bytes(stream, len(head), expected) if math.isfinite(expected) else 0
            else:
                handle.seek(0)
                header = _parse_header(_read_up_to(handle, _HEADER_SIZE))
                if header is None:
                    return (UNREADABLE_HEADER,)
                dim, bitpix, pixdim, vox_offset = header
                expected = _expected_size(dim, bitpix, vox_offset)
                available = os.fstat(handle.fileno()).st_size
    except _READ_ERRORS:
        return (UNREADABLE_HEADER,)
    return _reasons(dim, pixdim, expected, available)
