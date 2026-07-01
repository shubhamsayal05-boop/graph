# AVL-DRIVE-Style Objective Drivability Assessment

An independent, modular re-implementation of the **AVL-DRIVE™ 4.6 SR1**
objective drivability-benchmarking methodology, built as a Streamlit application
on top of a reusable, unit-tested Python engine (`avldrive`).

> This project mirrors AVL-DRIVE's structure, terminology, channels, signal
> processing, frequency bands, operation modes and 1–10 DRIVE-Rating direction.
> It does **not** reproduce AVL's proprietary calibration data or exact
> algorithms, and is intended for demonstration/engineering-education use.

## Highlights

- **Transmission-aware**: pick an architecture (`AT`, `CVT`, `eCVT`, `DCT`,
  `DHT`, `MT`, `BEV`) and the tool re-assigns its enabled operation modes,
  propulsion, gearbox character and relevant channels.
- **Authentic signal processing**: AVL filter notation `SMO/LP/HP/BP` on a
  100 Hz grid; acceleration disturbances **2–50 Hz**, LF **2–10 Hz**,
  HF **>10 Hz**, crest factor (HF), surge/correlation **<2 Hz**, response delay.
- **Operation modes**: Tip in/out, Drive away, Acceleration, Constant speed,
  Deceleration, Gear shift, Sailing, Recuperation, Idle, Vehicle stationary,
  Engine start/shut-off, Maneuvering.
- **DRIVE-Rating tree**: criteria → operation mode → overall, using a weight
  tree (criteria + mode weights) **plus extreme-value weighting**. Ratings are
  1–10 (10 = best) against a **brand-DNA target** (Luxury Sedan / Sports Car /
  Eco EV).
- **Calculated channels**: `AccelerationChassisCompensated`, `RoadGradient`,
  road-load `TractiveForce`, `WheelTorque`, `EngineTorqueEstimated`,
  `GearRatio`, `WheelSlip`.
- **Powerful UI**: multi-measurement **benchmarking**, a live **weight-tree
  editor** (JSON import/export), Welch **PSD frequency mapping**, a transient
  **microscope**, **statistical trend** views, cached processing, and a
  one-click **PDF executive report**.

### Engineering workflows

- **Calibration Advisor** — ranks criteria by *improvement potential* (how much
  the overall DR rises if each is lifted to target, through the weight tree),
  gives target-gap tables and root-cause → calibration-lever hints
  (anti-jerk/lash, torque ramp, lock-up, mount isolation, shift handover…).
- **Drivability Verification** — acceptance-spec presets (Production sign-off /
  Development milestone / Prototype) with editable thresholds, a PASS/FAIL gate,
  a requirement table and a **worst-event issue log**; exportable verdict.
- **Benchmark Library** — save assessments as named references (best-in-class),
  import/export the library as JSON, view a **drivability fingerprint** radar
  and a ranking table.
- **A/B comparison** — baseline-vs-candidate calibration diff at overall,
  operation-mode and criterion level with **regression/improvement** flags.

## Project layout

```
avldrive/                 Reusable, Streamlit-free engine
  config.py               Transmissions, criteria, weights, brand-DNA, channels
  dsp.py                  AVL filters (SMO/LP/HP/BP), RMS, crest factor
  channels.py             Channel resolution + calculated channels
  criteria.py             Criteria metrics and 1-10 rating
  operation_modes.py      Trigger engine / operation-mode detection
  assessment.py           Weighted + extreme-value DRIVE-Rating aggregation
  spectrum.py             Welch PSD mapping
  advisor.py              Calibration Advisor (priority ranking + lever hints)
  verification.py         Acceptance specs, PASS/FAIL gate, issue log
  benchmark.py            Reference library, fingerprint, ranking
  compare.py              A/B regression comparison
  reporting.py            PDF / text executive summary
  pipeline.py             load_measurement + run_assessment orchestration
app.py                    Streamlit UI (presentation only)
tests/                    pytest unit + pipeline tests
requirements.txt
```

## Install & run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Import one measurement to assess it, or several to benchmark them side by side.

## Tests

```bash
pip install pytest
pytest -q
```

The suite generates a synthetic `.mf4` (via `asammdf`) and exercises the DSP,
the rating aggregation, transmission-specific reconfiguration and the full
import→assess pipeline.

## Channel mapping

The engine resolves real measurement channels to AVL logical channels via a
candidate list (see `avldrive/config.py::CHANNEL_CANDIDATES`), so exports with
varying names (e.g. `Accel_Filt_X`, `veh_Spd_Kph`) still map. The minimum
required channels are `AcceleratorPedal` and `AccelerationChassis`.
```
