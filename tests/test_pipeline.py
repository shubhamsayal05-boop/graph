import numpy as np

from avldrive import (BRAND_DNA, TRANSMISSION_CONFIG, VehicleConfig,
                      load_measurement, resolve_channels, run_assessment)
from avldrive.pipeline import MissingChannelsError
from avldrive.spectrum import compute_band_spectrum
from avldrive.dsp import LP


def _cfg(transmission):
    tx = TRANSMISSION_CONFIG[transmission]
    return VehicleConfig(transmission=transmission, propulsion=tx["propulsion"]).as_dict()


def test_load_resolves_channels(synthetic_mf4):
    cfg = _cfg("AT")
    df, found = load_measurement(synthetic_mf4, cfg)
    assert "AcceleratorPedal" in found and "AccelerationChassis" in found
    assert "AccelerationChassisCompensated" in df.columns
    assert "RoadGradient" in df.columns


def test_at_detects_gear_shift_and_engine_torque(synthetic_mf4):
    cfg = _cfg("AT")
    df, _ = load_measurement(synthetic_mf4, cfg)
    assert "EngineTorqueEstimated" in df.columns
    res = run_assessment(df, TRANSMISSION_CONFIG["AT"]["modes"], cfg, BRAND_DNA["Luxury Sedan"])
    assert res.overall is not None
    assert "Gear shift" in res.mode_results


def test_bev_excludes_engine_modes(synthetic_mf4):
    cfg = _cfg("BEV")
    df, _ = load_measurement(synthetic_mf4, cfg)
    # BEV has no engine torque estimate.
    assert "EngineTorqueEstimated" not in df.columns
    res = run_assessment(df, TRANSMISSION_CONFIG["BEV"]["modes"], cfg, BRAND_DNA["Eco EV"])
    assert "Gear shift" not in res.mode_results
    assert "Idle" not in res.mode_results


def test_transmission_changes_enabled_modes():
    at = set(TRANSMISSION_CONFIG["AT"]["modes"])
    bev = set(TRANSMISSION_CONFIG["BEV"]["modes"])
    assert "Gear shift" in at and "Gear shift" not in bev
    assert "Recuperation" in bev and "Recuperation" not in at


def test_surge_spectrum_peak(synthetic_mf4):
    cfg = _cfg("AT")
    df, _ = load_measurement(synthetic_mf4, cfg)
    spec = compute_band_spectrum(LP(df["AccelerationChassisCompensated"].to_numpy(float), 2.0), (0.3, 2.0))
    assert spec is not None
    assert abs(spec["dominant_freq"] - 1.2) < 0.2


def test_missing_channels_raises(tmp_path):
    from asammdf import MDF, Signal
    t = np.arange(0, 5, 0.01)
    m = MDF()
    m.append([Signal(np.sin(t), t, name="SomeUnrelatedChannel", unit="-")])
    p = tmp_path / "bad.mf4"
    m.save(str(p), overwrite=True)
    try:
        load_measurement(str(p), _cfg("AT"))
        assert False, "expected MissingChannelsError"
    except MissingChannelsError as e:
        assert "AcceleratorPedal" in e.missing
