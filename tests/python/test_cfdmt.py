import numpy as np
import pytest

import dmtlib
from dmtlib import BasebandFormat, CohFDMT, CohFDMTConfig, CohFDMTPlan

K_DISP = 1.0 / 2.41e-4  # dmt's dispersion constant (kDispConst)


def small_config(**kw: object) -> CohFDMTConfig:
    """16 x 1 MHz at 392-408 MHz, t_p = 4 us, DM 10-11: five coarse trials."""
    return CohFDMTConfig(400.0, 1.0, 16, 4.0e-6, 10.0, 11.0, **kw)


def pulse_voltages(
    plan: CohFDMTPlan, dm: float, j0: float, noise: float = 0.0
) -> np.ndarray:
    """One block of voltages whose pulse lands on output sample j0."""
    t_ref = plan.output_time_offset + j0 * plan.tsamp
    t_inf = t_ref - K_DISP * dm / plan.f_ref**2
    return dmtlib.simulate_baseband(
        plan.f_center,
        plan.bw_sub,
        plan.nsub,
        plan.block_nsamps,
        [(dm, t_inf, 1.0e6, 0.0)],
        noise,
        7,
        4,
    )


def quantise(v: np.ndarray, fmt: BasebandFormat, peak: float = 100.0) -> np.ndarray:
    scale = peak / max(np.abs(v.real).max(), np.abs(v.imag).max())
    return dmtlib.pack_baseband(v, fmt, scale)


class TestPlan:
    def test_geometry(self) -> None:
        plan = CohFDMTPlan(small_config())
        assert plan.n_p == 4
        assert plan.nchans == 64
        assert plan.tsamp == pytest.approx(4.0e-6)
        assert plan.ndm_coh > 1
        assert plan.stride_nsamps + plan.overlap_nsamps == plan.block_nsamps
        assert plan.stride_nsamps == plan.output_nsamps * plan.n_p
        assert plan.dmt_size == plan.ndm * plan.output_nsamps
        assert plan.input_size() == 4 * plan.block_nsamps * plan.nsub
        assert plan.intra_channel_smear <= 1.0

    def test_coarse_windows_tile_the_range(self) -> None:
        plan = CohFDMTPlan(small_config())
        coh = plan.dm_grid_coh
        step = plan.dm_step_coh
        assert coh[0] - step / 2 == pytest.approx(10.0)
        assert coh[-1] + step / 2 == pytest.approx(11.0)
        np.testing.assert_allclose(np.diff(coh), step, rtol=1e-4)

    def test_multi_subband_grid_is_coarser_than_single_band_rule(self) -> None:
        # GUPPI node, DM 0-500: the single-band rule with the subband tbin
        # over-partitions by ~nsub / 2.
        plan = CohFDMTPlan(
            CohFDMTConfig(1406.25, 1500.0 / 512.0, 64, 10.0e-6, 0.0, 500.0)
        )
        delay = K_DISP * 500.0 * (plan.f_min**-2 - plan.f_max**-2)
        old = np.ceil(delay * plan.tbin / (2 * 1e-5**2))
        assert old / plan.ndm_coh > 16

    def test_variance_grids(self) -> None:
        plan = CohFDMTPlan(small_config())
        var = plan.get_effective_variance_grid()
        var16 = plan.get_effective_variance_grid(16)
        assert var.shape == (plan.ndm,)
        assert np.all(var >= plan.nchans)
        assert np.all(var16 > 16 * var)
        np.testing.assert_allclose(plan.get_effective_sigma_grid() ** 2, var, rtol=1e-6)
        assert plan.get_lag_correlation(2)[0] == pytest.approx(1.0)

    def test_filter_leakage_sets_the_overlap(self) -> None:
        tight = CohFDMTPlan(small_config(filter_leakage=1e-6))
        default = CohFDMTPlan(small_config())
        assert small_config().filter_leakage == pytest.approx(1e-4)
        assert default.noverlap < tight.noverlap
        assert default.noverlap % default.n_p == 0

    @pytest.mark.parametrize(
        "kw",
        [
            {"nsub": 0},
            {"dm_max": 5.0},
            {"format": BasebandFormat("PRIT")},
            {"format": BasebandFormat(nbits=16)},
            {"subband_groups": [4, 4]},
            {"block_nsamps": 1000},
            {"filter_leakage": 0.0},
            {"filter_leakage": 1.0},
        ],
    )
    def test_invalid_configs_raise(self, kw: dict) -> None:
        cfg = small_config()
        for k, v in kw.items():
            setattr(cfg, k, v)
        with pytest.raises(ValueError):
            CohFDMTPlan(cfg)


class TestExecute:
    @pytest.mark.parametrize("where", ["centre", "edge"])
    def test_recovers_dispersed_impulse(self, where: str) -> None:
        coh = CohFDMT(small_config(normalize=False), 4)
        plan = coh.plan
        c = plan.dm_grid_coh
        dm = c[2] if where == "centre" else c[1] + 0.49 * plan.dm_step_coh
        j0 = 200.0
        v = pulse_voltages(plan, dm, j0)
        out = coh.execute(quantise(v, plan.format))
        assert out.shape == (plan.ndm, plan.output_nsamps)
        row, t = np.unravel_index(np.argmax(out), out.shape)
        fine = plan.dm_grid_final[1] - plan.dm_grid_final[0]
        assert abs(plan.dm_grid_final[row] - dm) <= 1.5 * fine
        assert abs(t - j0) <= 1

    def test_normalised_noise_matches_variance_grid(self) -> None:
        coh = CohFDMT(small_config(block_nsamps=1 << 18), 4)
        plan = coh.plan
        v = dmtlib.simulate_baseband(
            plan.f_center, plan.bw_sub, plan.nsub, plan.block_nsamps, [], 16.0, 3, 4
        )
        out = coh.execute(dmtlib.pack_baseband(v, plan.format, 1.0))
        ratio = out.var(axis=1) / plan.get_effective_variance_grid()
        assert ratio.mean() == pytest.approx(1.0, abs=0.02)
        assert abs(out.mean()) < 0.05 * np.sqrt(plan.get_effective_variance_grid().mean())

    def test_skipback_blocks_tile_one_long_block(self) -> None:
        cfg = small_config(normalize=False)
        small = CohFDMT(cfg, 4)
        p = small.plan
        nb = 3
        total = p.block_nsamps + (nb - 1) * p.stride_nsamps
        big = CohFDMT(small_config(normalize=False, block_nsamps=total), 4)
        assert big.plan.output_nsamps == nb * p.output_nsamps
        v = dmtlib.simulate_baseband(
            p.f_center, p.bw_sub, p.nsub, total, [], 8.0, 5, 4
        )
        whole = big.execute(dmtlib.pack_baseband(v, p.format, 1.0))
        parts = [
            small.execute(
                dmtlib.pack_baseband(
                    v, p.format, 1.0, t_begin=b * p.stride_nsamps, t_count=p.block_nsamps
                )
            )
            for b in range(nb)
        ]
        np.testing.assert_allclose(
            np.concatenate(parts, axis=1), whole, atol=1e-5 * np.abs(whole).max()
        )

    def test_layouts_encodings_and_groups_agree(self) -> None:
        base = small_config()
        plan = CohFDMTPlan(base)
        v = pulse_voltages(plan, plan.dm_grid_coh[1], 150.0, noise=0.5)
        scale = 50.0 / np.abs(v).max()
        ref = CohFDMT(base, 4).execute(dmtlib.pack_baseband(v, base.format, scale))
        for order in ("PRITF", "TFPRI"):
            for signed in (True, False):
                fmt = BasebandFormat(order, is_signed=signed)
                got = CohFDMT(small_config(format=fmt), 4).execute(
                    dmtlib.pack_baseband(v, fmt, scale)
                )
                np.testing.assert_array_equal(got, ref)
        grouped = CohFDMT(small_config(subband_groups=[6, 10]), 4)
        groups = [
            dmtlib.pack_baseband(v, base.format, scale, sub_begin=0, sub_count=6),
            dmtlib.pack_baseband(v, base.format, scale, sub_begin=6, sub_count=10),
        ]
        np.testing.assert_array_equal(grouped.execute(groups), ref)

    def test_out_buffer_reuse_and_int8_input(self) -> None:
        coh = CohFDMT(small_config(), 2)
        raw = np.random.default_rng(1).normal(0, 10, coh.input_size()).astype(np.int8)
        buf = np.empty(coh.dmt_size, dtype=np.float32)
        out = coh.execute(raw, out=buf)
        assert out.base is buf or np.shares_memory(out, buf)
        np.testing.assert_array_equal(out, coh.execute(raw.view(np.uint8)))
        assert coh.memory_usage()["total"] > 0

    def test_bad_inputs_raise(self) -> None:
        coh = CohFDMT(small_config())
        with pytest.raises(ValueError):
            coh.execute(np.zeros(coh.input_size() - 4, dtype=np.uint8))
        with pytest.raises(TypeError):
            coh.execute(np.zeros(coh.input_size(), dtype=np.float32))
        with pytest.raises(ValueError):
            coh.execute(
                np.zeros(coh.input_size(), dtype=np.uint8),
                out=np.empty(coh.dmt_size - 1, dtype=np.float32),
            )
