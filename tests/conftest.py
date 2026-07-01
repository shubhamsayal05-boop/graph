import numpy as np
import pytest
from asammdf import MDF, Signal

from avldrive.config import FS


@pytest.fixture(scope="session")
def synthetic_mf4(tmp_path_factory):
    """A synthetic drive with two tip-ins, a launch and injected disturbances."""
    fs = FS
    t = np.arange(0, 40.0, 1.0 / fs)
    pedal = np.zeros_like(t)
    for t0 in (8.0, 22.0):
        pedal += 60 * (1 / (1 + np.exp(-(t - t0) * 8))) * np.exp(-np.maximum(0, t - t0) / 6)
    pedal = np.clip(pedal, 0, 100)
    accel = np.zeros_like(t)
    for t0 in (8.0, 22.0):
        d = t - (t0 + 0.25)
        accel += 2.5 * (d > 0) * (1 - np.exp(-np.maximum(0, d) / 0.4)) * np.exp(-np.maximum(0, d) / 6)
    accel += (0.15 * np.sin(2 * np.pi * 1.2 * t) + 0.10 * np.sin(2 * np.pi * 5 * t)
              + 0.05 * np.sin(2 * np.pi * 14 * t))
    rng = np.random.default_rng(0)
    accel += 0.02 * rng.standard_normal(len(t))
    speed = np.cumsum(np.maximum(0, accel)) * (1.0 / fs) * 3.6
    espeed = 1000 + speed * 40 + pedal * 15
    sigs = [
        Signal(accel, t, name="AccelerationChassis", unit="m/s^2"),
        Signal(accel * 0.3, t, name="AccelerationVertical", unit="m/s^2"),
        Signal(pedal, t, name="AcceleratorPedal", unit="%"),
        Signal(np.zeros_like(t), t, name="BrakePosition", unit="%"),
        Signal(espeed, t, name="EngineSpeed", unit="rpm"),
        Signal(speed, t, name="VehicleSpeed", unit="km/h"),
        Signal(np.clip(np.round(speed / 25) + 1, 1, 6), t, name="GearEngaged", unit="-"),
    ]
    path = tmp_path_factory.mktemp("data") / "synthetic.mf4"
    m = MDF()
    m.append(sigs)
    m.save(str(path), overwrite=True)
    return str(path)
