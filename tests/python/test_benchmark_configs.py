"""Prevent archived benchmark measurements from being silently relabeled."""

import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def plot_suite() -> Iterator[ModuleType]:
    pytest.importorskip("matplotlib")
    path = Path(__file__).resolve().parents[2] / "bench/scripts/plot_suite.py"
    spec = importlib.util.spec_from_file_location("dmt_test_plot_suite", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        del sys.modules[spec.name]


@pytest.mark.parametrize("mismatch", [{"nchans": 1024}, {"tsamp": 64e-6}])
def test_plot_rejects_conflicting_geometry(
    tmp_path: Path, plot_suite: ModuleType, mismatch: dict[str, float]
) -> None:
    machine = tmp_path / "machine"
    machine.mkdir()
    benchmark = {
        "name": "suite/ndms/FDMT/cpu1/nsamps:32768/dmmax:2000/nbits:32/real_time",
        "nchans": 4096,
        "real_time": 1,
        "time_unit": "s",
        **mismatch,
    }
    result_file = machine / "suite_cpu.json"
    result_file.write_text(json.dumps({"benchmarks": [benchmark]}))
    with pytest.raises(ValueError, match="does not match selected preset"):
        plot_suite.load(tmp_path)
    # The default preset's geometry (meerkat_lband).
    benchmark.update(nchans=4096, tsamp=612.48e-6)
    result_file.write_text(json.dumps({"benchmarks": [benchmark]}))
    results, machines = plot_suite.load(tmp_path)
    assert len(results) == 1
    assert results[0].time == 1
    assert "machine" in machines
