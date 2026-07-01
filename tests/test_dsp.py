import numpy as np

from avldrive.dsp import BP, HP, LP, SMO, crest_factor, rms
from avldrive.config import FS


def _tone(freq, dur=20.0, amp=1.0):
    t = np.arange(0, dur, 1.0 / FS)
    return t, amp * np.sin(2 * np.pi * freq * t)


def test_lp_passes_low_blocks_high():
    _, low = _tone(1.0)
    _, high = _tone(20.0)
    sig = low + high
    filtered = LP(sig, 5.0)
    # Low-frequency content preserved, high attenuated.
    assert rms(filtered) < rms(sig)
    assert abs(rms(LP(low, 5.0)) - rms(low)) / rms(low) < 0.1


def test_bp_isolates_band():
    _, mid = _tone(5.0)
    _, low = _tone(0.5)
    _, high = _tone(30.0)
    filtered = BP(low + mid + high, 2.0, 10.0)
    # The 5 Hz component should dominate the band-passed signal.
    assert rms(filtered) > 0.5 * rms(mid)


def test_hp_blocks_dc():
    x = np.ones(4000) * 3.0
    assert rms(HP(x, 10.0)) < 0.05


def test_smo_preserves_length_and_smooths():
    x = np.random.randn(1000)
    y = SMO(x, 21)
    assert len(y) == len(x)
    assert np.std(y) < np.std(x)


def test_crest_factor_of_sine():
    _, s = _tone(2.0, amp=1.0)
    # Crest factor of a pure sine is ~sqrt(2).
    assert abs(crest_factor(s) - np.sqrt(2)) < 0.1
