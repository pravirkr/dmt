"""Shared test helpers for the Python test suite."""

from __future__ import annotations

import numpy as np

_CANDIDATE_SORT_FIELDS = [
    "beam_idx",
    "subband_level",
    "subband_idx",
    "dm_idx",
    "filter_idx",
    "begin_sample",
    "end_sample",
    "peak_sample",
]


def pack_lsb_first(values: np.ndarray, nbits: int) -> np.ndarray:
    """Pack unsigned ``nbits``-wide samples along the last axis, LSB-first."""
    if nbits == 8:
        return np.ascontiguousarray(values, dtype=np.uint8)
    if nbits == 16:
        return (
            np.ascontiguousarray(values, dtype="<u2")
            .view(np.uint8)
            .reshape(*values.shape[:-1], 2 * values.shape[-1])
        )
    per_byte = 8 // nbits
    nsamps = values.shape[-1]
    row_bytes = (nsamps * nbits + 7) // 8
    lead = values.shape[:-1]
    padded = np.zeros((*lead, row_bytes * per_byte), dtype=np.uint32)
    padded[..., :nsamps] = values
    shifts = np.arange(per_byte, dtype=np.uint32) * nbits
    lanes = padded.reshape(*lead, row_bytes, per_byte) << shifts
    return lanes.sum(axis=-1).astype(np.uint8)


def sorted_candidates(parts: list[np.ndarray]) -> np.ndarray:
    """Concatenate and canonically sort structured candidate arrays."""
    return np.sort(np.concatenate(parts), order=_CANDIDATE_SORT_FIELDS)
