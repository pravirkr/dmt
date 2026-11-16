"""Trial DM grids for FDMT and DDMT.

The C++ generators (``dmt/common/dm_grids.hpp``), re-exported:

- :func:`generate_dm_grid`: dmt's rule. Trials are spaced so a pulse halfway
  between two keeps ``1 - max_snr_loss`` of its S/N after the boxcar
  search, modelled over every channel (dispersion misalignment,
  intra-channel and FDMT box smearing at ``|DM - chan_dm|``, sampling, the
  boxcar widths and the noise they add). Negative DMs, coherently
  dedispersed data (``chan_dm``) and dead channels (``kill_mask``).
- :func:`generate_levin_dm_grid`: Levin (2012), the classic rule as in
  Heimdall, kept as it is.
- :func:`calculate_snr_loss`: the loss a grid leaves at any DM, on the model
  of :func:`generate_dm_grid`.

Both generators take ``piecewise=True`` for :class:`DDMTFFT`'s NUFFT path.
Frequencies are in MHz, times (``tsamp``, ``pulse_width``) in seconds.
"""

from __future__ import annotations

from .libdmt import calculate_snr_loss, generate_dm_grid, generate_levin_dm_grid

DM_CONSTANT = 1 / 2.41e-4  # TEMPO2, Manchester&Taylor (1972)

__all__ = [
    "DM_CONSTANT",
    "calculate_snr_loss",
    "generate_dm_grid",
    "generate_levin_dm_grid",
]
