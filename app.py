"""AVL-DRIVE-style objective drivability assessment tool.

This Streamlit app is modelled as closely as practical on AVL-DRIVE(TM) 4.6 SR1
(the AVL objective drivability benchmarking tool). Terminology, channel names,
signal-processing filters, frequency bands, operation modes, criteria and the
1-10 rating direction follow the AVL-DRIVE Function Description (AT) shipped in
the repository. It is an independent re-implementation for demonstration and
does not reproduce AVL's proprietary calibration data or exact algorithms.
"""
import streamlit as st
from asammdf import MDF
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, welch
import plotly.graph_objects as go
import os
from io import BytesIO
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. CONFIGURATION MODE CONSTANTS (AVL-DRIVE "Configuration" tab)
# -----------------------------------------------------------------------------
# Transmission types mirror the .ect template families provided (AT/CVT/DCT/DHT/
# MT/BEV). Propulsion and vehicle data feed AVL "Calculated Channels".
TRANSMISSION_TYPES = ["AT", "CVT", "DCT", "DHT", "MT", "BEV"]
PROPULSION_TYPES = ["ICE", "Hybrid", "BEV"]

# AVL-DRIVE rates every criterion against the customer "brand-DNA" target. These
# profiles set how strictly each criterion is scored (higher sensitivity = a
# small fault costs more rating points). Direction always follows AVL: "the
# higher the disturbance / the longer the delay / the worse the correlation, the
# lower the rating". The delay coefficients reproduce the earlier tuning
# (200 ms -> Luxury Sedan 7.6, Sports Car 4.8, Eco EV 8.8).
BRAND_DNA = {
    "Luxury Sedan": {"disturbances": 3.0, "lf": 3.5, "hf": 2.5, "crest": 0.7,
                     "correlation": 5.0, "delay": 12.0},
    "Sports Car": {"disturbances": 4.5, "lf": 5.0, "hf": 4.0, "crest": 1.0,
                   "correlation": 7.0, "delay": 26.0},
    "Eco EV": {"disturbances": 2.5, "lf": 3.0, "hf": 2.5, "crest": 0.6,
               "correlation": 4.0, "delay": 6.0},
}

# AVL-DRIVE import-template DRIVE channels. Each logical signal maps to a list of
# candidate raw channel names so real .mf4 exports (which use varying naming)
# still resolve. The first present candidate wins.
CHANNEL_CANDIDATES = {
    "AccelerationChassis": ["AccelerationChassis", "AccelerationChassisComp",
                            "AccelChassis", "Accel_Filt_X", "Ax_Sensor_g", "Ax"],
    "AccelerationVertical": ["AccelerationVertical", "AccelerationWheel_Z",
                             "Az_Sensor_g", "Az"],
    "AccelerationLateral": ["AccelerationLateral", "Ay_Sensor_g", "Ay"],
    "AcceleratorPedal": ["AcceleratorPedal", "Acc_Pedal_Pct", "AccPedal",
                         "PedalPosition", "iv_dki"],
    "BrakePosition": ["BrakePosition", "BrakePdlPosn", "Brake", "iv_BrakePdlPosn"],
    "EngineSpeed": ["EngineSpeed", "Engine_RPM", "EngSpeed", "nmot", "Nmot"],
    "VehicleSpeed": ["VehicleSpeed", "veh_Spd_Kph", "VehicleSpeedMPH", "VehSpeed"],
    "TurbineSpeed": ["TurbineSpeed"],
    "GearEngaged": ["GearEngaged", "Gear Engaged", "Current_Gear", "Gear", "GearDMU"],
    "SelectorLeverDMU": ["SelectorLeverDMU", "SelectorLever"],
    "TCC_State": ["TCC_State", "TCC"],
    "EngineTorque": ["Engine_Torque", "EngineTorque", "Torque"],
    "WheelSpeed_FL": ["WheelSpeed_FL", "WheelSpeedFL"],
    "WheelSpeed_FR": ["WheelSpeed_FR", "WheelSpeedFR"],
    "WheelSpeed_RL": ["WheelSpeed_RL", "WheelSpeedRL"],
    "WheelSpeed_RR": ["WheelSpeed_RR", "WheelSpeedRR"],
}
# Minimum channels required to run an assessment.
REQUIRED_LOGICAL = ["AcceleratorPedal", "AccelerationChassis"]

FS = 100.0  # AVL-DRIVE analysis grid: 100 Hz (Nyquist 50 Hz).

st.set_page_config(page_title="AVL-DRIVE-Style Drivability Lab", layout="wide")
st.title("⚙️ AVL-DRIVE-Style Objective Drivability Assessment")
st.caption("Independent re-implementation of AVL-DRIVE™ 4.6 SR1 methodology — "
           "operation modes, criteria and 1–10 brand-DNA ratings.")

# -----------------------------------------------------------------------------
# 2. SIGNAL PROCESSING (AVL filter notation: SMO / LP / HP / BP)
# -----------------------------------------------------------------------------
def _butter_filt(data, btype, cut, fs=FS, order=3):
    """Zero-phase Butterworth filter with Nyquist-safe cutoff clamping."""
    x = np.asarray(data, dtype=float)
    nyq = 0.5 * fs
    if btype == "band":
        lo = max(cut[0] / nyq, 1e-4)
        hi = min(cut[1] / nyq, 0.999)
        if lo >= hi:
            return x
        b, a = butter(order, [lo, hi], btype="band")
    elif btype == "low":
        b, a = butter(order, min(cut / nyq, 0.999), btype="low")
    else:  # high
        b, a = butter(order, max(cut / nyq, 1e-4), btype="high")
    padlen = 3 * max(len(a), len(b))
    if len(x) <= padlen:
        return x
    return filtfilt(b, a, x)

def LP(x, c, fs=FS):
    """Low-pass: passes frequencies below c Hz."""
    return _butter_filt(x, "low", c, fs)

def HP(x, c, fs=FS):
    """High-pass: passes frequencies above c Hz."""
    return _butter_filt(x, "high", c, fs)

def BP(x, lo, hi, fs=FS):
    """Band-pass: passes frequencies between lo and hi Hz."""
    return _butter_filt(x, "band", (lo, hi), fs)

def SMO(x, n):
    """AVL smoothing filter: weighted (triangular) moving average over n points."""
    x = np.asarray(x, dtype=float)
    n = max(1, int(n))
    if n <= 1 or len(x) < n:
        return x
    w = np.concatenate([np.arange(1, n // 2 + 2), np.arange((n + 1) // 2, 0, -1)])
    w = w[:n].astype(float)
    w /= w.sum()
    return np.convolve(x, w, mode="same")

def rms(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0

def crest_factor(x):
    """Peak-to-RMS ratio in the higher-frequency domain (AVL: a high value
    indicates occasional bumps)."""
    r = rms(x)
    if r <= 1e-9:
        return 0.0
    return float(np.max(np.abs(np.asarray(x, dtype=float))) / r)

# -----------------------------------------------------------------------------
# 3. CALCULATED CHANNELS (AVL-DRIVE Function Description, section 1.3)
# -----------------------------------------------------------------------------
def build_calculated_channels(df, config):
    """Derives AVL calculated channels from the imported raw channels."""
    # AccelerationChassisCompensated: chassis acceleration with the slow road-
    # gradient / gravity offset removed (AVL compensates the longitudinal signal
    # before assessing disturbances). Approximated by removing the < 0.3 Hz drift.
    ax = df["AccelerationChassis"].to_numpy(dtype=float)
    df["AccelerationChassisCompensated"] = ax - LP(ax, 0.3)

    # GearRatio = EngineSpeed / VehicleSpeed (AVL uses turbine speed if enabled).
    if "EngineSpeed" in df and "VehicleSpeed" in df:
        v = df["VehicleSpeed"].to_numpy(dtype=float)
        n = df["EngineSpeed"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            gr = np.where(np.abs(v) > 1.0, n / np.where(v == 0, np.nan, v), np.nan)
        df["GearRatio"] = gr

    # WheelSlip: deviation between driven and non-driven wheel means (percent).
    fl, fr = df.get("WheelSpeed_FL"), df.get("WheelSpeed_FR")
    rl, rr = df.get("WheelSpeed_RL"), df.get("WheelSpeed_RR")
    if all(w is not None for w in (fl, fr, rl, rr)):
        front = (fl.to_numpy(float) + fr.to_numpy(float)) / 2.0
        rear = (rl.to_numpy(float) + rr.to_numpy(float)) / 2.0
        driven, non_driven = (front, rear)  # assume FWD unless configured RWD
        if config.get("drive") == "RWD":
            driven, non_driven = rear, front
        with np.errstate(divide="ignore", invalid="ignore"):
            df["WheelSlip"] = np.where(np.abs(non_driven) > 1.0,
                                       (driven - non_driven) / non_driven * 100.0, 0.0)
    return df

# -----------------------------------------------------------------------------
# 4. CRITERIA (AVL-DRIVE assessment; 1-10 scale, 10 = best)
# -----------------------------------------------------------------------------
def rate(value, sensitivity, offset=0.0):
    """Convert a fault metric into a 1-10 rating. Higher value -> lower rating
    (AVL assessment direction). ``offset`` is a fault-free allowance."""
    return float(np.clip(10.0 - max(0.0, value - offset) * sensitivity, 1.0, 10.0))

def compute_disturbance_criteria(accel_comp, dna):
    """Acceleration disturbances family on AccelerationChassisCompensated.

    Bands per AVL Function Description:
      disturbances (total): 2-50 Hz
      disturbances LF:      2-10 Hz
      disturbances HF:      >10 Hz (10-50 Hz)
      crest factor:         peak/RMS in the HF domain
    """
    total = BP(accel_comp, 2.0, 49.5)
    lf = BP(accel_comp, 2.0, 10.0)
    hf = HP(accel_comp, 10.0)
    rms_total, rms_lf, rms_hf = rms(total), rms(lf), rms(hf)
    cf = crest_factor(hf)
    return {
        "disturbances": {"metric": rms_total, "unit": "m/s²",
                         "rating": rate(rms_total, dna["disturbances"]),
                         "signal": total},
        "disturbances_lf": {"metric": rms_lf, "unit": "m/s²",
                            "rating": rate(rms_lf, dna["lf"]), "signal": lf},
        "disturbances_hf": {"metric": rms_hf, "unit": "m/s²",
                            "rating": rate(rms_hf, dna["hf"]), "signal": hf},
        "crest_factor": {"metric": cf, "unit": "-",
                         "rating": rate(cf, dna["crest"], offset=3.0)},
    }

def compute_correlation_criterion(pedal, accel_comp, dna):
    """Surge/Correlation: pedal↔acceleration correlation at low frequency (<2 Hz).
    AVL: the worse the correlation (higher parameter), the lower the rating."""
    p = LP(np.asarray(pedal, float), 2.0)
    a = LP(np.asarray(accel_comp, float), 2.0)
    # Assess only active driving (pedal applied); idle sections dilute the
    # correlation. AVL assesses correlation during load-increase events.
    active = np.asarray(pedal, float) > 5.0
    if active.sum() > 50:
        p, a = p[active], a[active]
    if np.std(p) < 1e-6 or np.std(a) < 1e-6:
        return {"metric": 0.0, "unit": "-", "rating": 10.0, "corr": 1.0}
    corr = float(np.corrcoef(p, a)[0, 1])
    param = 1.0 - max(0.0, corr)  # 0 = perfect correlation (best)
    return {"metric": param, "unit": "-", "rating": rate(param, dna["correlation"]),
            "corr": corr}

def response_delay(event_df, dna):
    """Response delay [s]: dead-time from accelerator tip-in to physical
    acceleration onset, using AcceleratorPedal vs AccelerationChassis_SMO(20)."""
    ev = event_df.reset_index(drop=True)
    pedal = ev["AcceleratorPedal"].to_numpy(float)
    accel_smo = SMO(ev["AccelerationChassisCompensated"].to_numpy(float), 20)
    t = ev["timestamp"].to_numpy(float)
    trig = np.argmax(pedal > 5.0)
    if not (pedal > 5.0).any():
        return None
    t_pedal = t[trig]
    jerk = np.gradient(accel_smo, 1.0 / FS)
    post = np.arange(trig, len(t))
    onset = post[jerk[post] > 0.5]
    if onset.size == 0:
        return None
    delay = max(0.0, float(t[onset[0]] - t_pedal))
    return delay

# -----------------------------------------------------------------------------
# 5. TRIGGER ENGINE / OPERATION MODE DETECTION (simplified AVL trigger engine)
# -----------------------------------------------------------------------------
def detect_operation_modes(df):
    """Segments the drive into AVL-style operation-mode events. Returns a list of
    dicts: {mode, t_start, t_end}."""
    t = df["timestamp"].to_numpy(float)
    pedal = df["AcceleratorPedal"].to_numpy(float)
    speed = df["VehicleSpeed"].to_numpy(float) if "VehicleSpeed" in df else np.zeros_like(t)
    pedal_rate = np.gradient(pedal, 1.0 / FS)
    events = []

    def cluster(idxs, gap=3.0, pre=0.5, post=2.5):
        if len(idxs) == 0:
            return []
        starts = [t[idxs[0]]]
        for i in idxs[1:]:
            if t[i] - starts[-1] > gap:
                starts.append(t[i])
        return [(s - pre, s + post) for s in starts]

    # Tip in: pedal increase faster than ~40 %/s with meaningful pedal travel.
    for s, e in cluster(np.where((pedal_rate > 40) & (pedal > 10))[0]):
        events.append({"mode": "Tip in", "t_start": s, "t_end": e})
    # Tip out: pedal release faster than ~40 %/s.
    for s, e in cluster(np.where((pedal_rate < -40) & (pedal < 60))[0], post=2.0):
        events.append({"mode": "Tip out", "t_start": s, "t_end": e})
    # Drive away: vehicle rises through 5 km/h having been near standstill (<3
    # km/h) shortly before. Crossing is detected over the threshold, not as a
    # single-sample jump.
    if "VehicleSpeed" in df:
        cross = np.where((speed[:-1] < 5.0) & (speed[1:] >= 5.0))[0] + 1
        win = int(1.5 * FS)
        launch = [i for i in cross if speed[max(0, i - win):i].min() < 3.0]
        for s, e in cluster(np.array(launch, dtype=int), gap=5.0, pre=0.5, post=4.0):
            events.append({"mode": "Drive away", "t_start": s, "t_end": e})
    events.sort(key=lambda ev: ev["t_start"])
    return events

# -----------------------------------------------------------------------------
# 6. SPECTRUM MAPPING (Welch PSD)
# -----------------------------------------------------------------------------
def compute_band_spectrum(signal_data, band, fs=FS):
    signal = np.asarray(signal_data, dtype=float)
    signal = signal[np.isfinite(signal)]
    nperseg = int(min(len(signal), 1024))
    if nperseg < 16:
        return None
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return None
    bf, bp_ = freqs[mask], psd[mask]
    i = int(np.argmax(bp_))
    return {"freqs": freqs, "psd": psd, "band": band,
            "dominant_freq": float(bf[i]), "dominant_power": float(bp_[i])}

def interpret_surge_source(f):
    if f < 1.0:
        return ("Low-frequency chugging (< 1.0 Hz) — engine combustion firing / "
                "lugging loops or coarse driveline lash.")
    if f < 1.5:
        return ("Mid-band surge (1.0-1.5 Hz) — mixed powertrain source.")
    return ("Faster surge oscillation (> 1.5 Hz) — electric motor / e-drive "
            "controller damping (limit-cycle) loop.")

def render_psd_block(container, signal_data, band, title, color, caption_fn=None):
    spec = compute_band_spectrum(signal_data, band)
    low, high = band
    if spec is None:
        container.info(f"{title}: slice too short to resolve a dominant frequency peak.")
        return None
    dom = spec["dominant_freq"]
    container.markdown(
        f"**{title} — dominant frequency: `{dom:.2f} Hz`** "
        f"(band {low:g}–{high:g} Hz, PSD peak {spec['dominant_power']:.3e} (m/s²)²/Hz)")
    if caption_fn is not None:
        container.caption(caption_fn(dom))
    mask = (spec["freqs"] >= low) & (spec["freqs"] <= high)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spec["freqs"][mask], y=spec["psd"][mask],
                             name=f"{title} PSD", line=dict(color=color), fill="tozeroy"))
    fig.add_vline(x=dom, line=dict(color="crimson", dash="dash"),
                  annotation_text=f"{dom:.2f} Hz", annotation_position="top")
    fig.update_layout(title=f"{title} Power Spectral Density (Welch)",
                      xaxis_title="Frequency (Hz)", yaxis_title="PSD (m/s²)²/Hz", height=320)
    container.plotly_chart(fig, use_container_width=True)
    return dom

# -----------------------------------------------------------------------------
# 7. REPORT EXPORT
# -----------------------------------------------------------------------------
def build_summary_report(meta, scores):
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                                topMargin=2 * cm, bottomMargin=2 * cm)
        styles = getSampleStyleSheet()
        story = [Paragraph("AVL-DRIVE-Style Executive Drivability Summary", styles["Title"]),
                 Spacer(1, 6), Paragraph(f"Generated: {generated}", styles["Normal"]),
                 Spacer(1, 12), Paragraph("Configuration &amp; File Metadata", styles["Heading2"])]
        head_style = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b57")),
                      ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                      ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                      ("FONTSIZE", (0, 0), (-1, -1), 9),
                      ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white])]
        mt = Table([["Field", "Value"]] + [[k, str(v)] for k, v in meta.items()],
                   colWidths=[6 * cm, 9 * cm])
        mt.setStyle(TableStyle(head_style))
        story += [mt, Spacer(1, 16), Paragraph("Criteria Ratings (1-10, 10 = best)", styles["Heading2"])]
        stbl = Table([["Criterion", "Rating", "Metric"]] +
                     [[n, f"{v:.1f} / 10", d] for n, v, d in scores],
                     colWidths=[6 * cm, 3 * cm, 6 * cm])
        stbl.setStyle(TableStyle(head_style))
        story.append(stbl)
        doc.build(story)
        return buf.getvalue(), "application/pdf", "pdf"
    except Exception:
        lines = ["AVL-DRIVE-STYLE EXECUTIVE DRIVABILITY SUMMARY", "=" * 46,
                 f"Generated: {generated}", "", "CONFIGURATION & FILE METADATA", "-" * 46]
        lines += [f"{k:<26}: {v}" for k, v in meta.items()]
        lines += ["", "CRITERIA RATINGS (1-10, 10 = best)", "-" * 46]
        lines += [f"{n:<26}: {v:>4.1f} / 10   {d}" for n, v, d in scores]
        return ("\n".join(lines) + "\n").encode("utf-8"), "text/plain", "txt"

# -----------------------------------------------------------------------------
# 8. CHANNEL RESOLUTION
# -----------------------------------------------------------------------------
def resolve_channels(mdf):
    """Maps available raw channels to AVL logical channels using the candidate
    lists. Returns (logical->raw dict)."""
    available = list(mdf.channels_db.keys())
    lower = {name.lower(): name for name in available}
    found = {}
    for logical, cands in CHANNEL_CANDIDATES.items():
        for c in cands:
            if c in mdf.channels_db:
                found[logical] = c
                break
            if c.lower() in lower:
                found[logical] = lower[c.lower()]
                break
    return found

# -----------------------------------------------------------------------------
# 9. CONFIGURATION MODE (sidebar)
# -----------------------------------------------------------------------------
st.sidebar.header("🔧 Configuration Mode")
transmission = st.sidebar.selectbox("Transmission type", TRANSMISSION_TYPES, index=0)
propulsion = st.sidebar.selectbox("Propulsion", PROPULSION_TYPES, index=0)
drive_layout = st.sidebar.selectbox("Driven axle", ["FWD", "RWD"], index=0)
brand = st.sidebar.selectbox(
    "Brand-DNA target profile", list(BRAND_DNA.keys()),
    index=list(BRAND_DNA.keys()).index("Luxury Sedan"),
    help="Sets how strictly criteria are scored against the target character. "
         "'Sports Car' punishes delay and disturbances far more than 'Luxury Sedan'.")
dna = BRAND_DNA[brand]
config = {"transmission": transmission, "propulsion": propulsion, "drive": drive_layout,
          "brand": brand}
st.sidebar.caption(
    f"Target **{brand}** — delay ×{dna['delay']:.0f}, disturbances ×{dna['disturbances']:.1f}, "
    f"LF ×{dna['lf']:.1f}, HF ×{dna['hf']:.1f}")

uploaded_file = st.sidebar.file_uploader("Import measurement (.mf4, .dat)", type=["mf4", "dat"])

# -----------------------------------------------------------------------------
# 10. MAIN PIPELINE
# -----------------------------------------------------------------------------
if uploaded_file is None:
    st.info("⬅️ Configure the vehicle and import a `.mf4` / `.dat` measurement to begin.")
    st.markdown(
        "**AVL-DRIVE alignment notes** — analysis grid 100 Hz (Nyquist 50 Hz); "
        "acceleration disturbances 2–50 Hz, LF 2–10 Hz, HF >10 Hz; surge/correlation "
        "assessed <2 Hz; ratings 1–10 (10 = best) against the selected brand-DNA target.")
else:
    temp_filename = f"temp_{uploaded_file.name}"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        with st.spinner("Importing channels and building the 100 Hz analysis grid..."):
            mdf = MDF(temp_filename)
            found = resolve_channels(mdf)
            missing_required = [c for c in REQUIRED_LOGICAL if c not in found]
            if missing_required:
                st.error("Missing required DRIVE channels: " + ", ".join(missing_required) +
                         f". Detected channels: {sorted(found.keys())}")
                st.stop()

            mdf_res = mdf.filter(list(found.values())).resample(raster=1.0 / FS)
            df = mdf_res.to_dataframe()
            df = df.rename(columns={raw: logical for logical, raw in found.items()})
            df = df.reset_index().rename(columns={"index": "timestamp", "time": "timestamp"})
            if "timestamp" not in df.columns:
                df = df.rename(columns={df.columns[0]: "timestamp"})
            df = build_calculated_channels(df, config)

        st.sidebar.success(f"Imported {len(found)} DRIVE channels.")
        with st.sidebar.expander("Resolved channel mapping"):
            st.write({logical: raw for logical, raw in found.items()})

        accel_comp = df["AccelerationChassisCompensated"]
        pedal = df["AcceleratorPedal"]

        # ---- Global criteria (whole measurement) -----------------------------
        dist = compute_disturbance_criteria(accel_comp, dna)
        corr = compute_correlation_criterion(pedal, accel_comp, dna)

        # ---- Operation-mode detection & response delay -----------------------
        events = detect_operation_modes(df)
        delays = []
        for ev in events:
            if ev["mode"] in ("Tip in", "Drive away"):
                seg = df[(df["timestamp"] >= ev["t_start"]) & (df["timestamp"] <= ev["t_end"])]
                d = response_delay(seg, dna)
                if d is not None:
                    delays.append(d)
        mean_delay = float(np.mean(delays)) if delays else 0.0
        rating_delay = rate(mean_delay, dna["delay"])

        criteria = {
            "Acceleration disturbances (2-50 Hz)": dist["disturbances"],
            "Acceleration disturbances LF (2-10 Hz)": dist["disturbances_lf"],
            "Acceleration disturbances HF (>10 Hz)": dist["disturbances_hf"],
            "Crest factor (HF)": dist["crest_factor"],
            "Surge / Correlation (<2 Hz)": {"metric": corr["metric"], "unit": "-",
                                            "rating": corr["rating"]},
            "Response delay": {"metric": mean_delay, "unit": "s", "rating": rating_delay},
        }
        overall = float(np.mean([c["rating"] for c in criteria.values()]))

        # ---- Scoreboard ------------------------------------------------------
        st.markdown(f"### 📊 Driveability Rating — *{brand}* ({transmission}/{propulsion})")
        st.metric("🏁 Overall Driveability Rating", f"{round(overall, 1)} / 10")
        cols = st.columns(3)
        for i, (name, c) in enumerate(criteria.items()):
            unit = c["unit"]
            detail = f"{c['metric']:.3f} {unit}" if unit != "-" else f"param {c['metric']:.3f}"
            if name == "Surge / Correlation (<2 Hz)":
                detail = f"r = {corr['corr']:.2f}"
            if name == "Response delay":
                detail = f"mean lag {int(mean_delay * 1000)} ms ({len(delays)} events)"
            cols[i % 3].metric(name, f"{round(c['rating'], 1)} / 10", detail)

        # ---- Operation-mode list (AVL "List of Operation Modes") -------------
        st.markdown("### 🧭 Detected Operation Modes")
        if events:
            om_df = pd.DataFrame([
                {"Operation Mode": ev["mode"],
                 "Start [s]": round(ev["t_start"], 2),
                 "End [s]": round(ev["t_end"], 2),
                 "Duration [s]": round(ev["t_end"] - ev["t_start"], 2)}
                for ev in events])
            st.dataframe(om_df, use_container_width=True, hide_index=True)
        else:
            st.info("No discrete Tip-in / Tip-out / Drive-away events triggered in this measurement.")

        # ---- Report export ---------------------------------------------------
        duration_s = float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) if len(df) else 0.0
        report_meta = {
            "Source File": uploaded_file.name,
            "Transmission / Propulsion": f"{transmission} / {propulsion}",
            "Brand-DNA Target": brand,
            "Analysis Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Log Duration (s)": f"{duration_s:.1f}",
            "Samples (100 Hz grid)": len(df),
            "Operation Modes Detected": len(events),
            "Mean Response Lag (ms)": int(mean_delay * 1000),
        }
        report_scores = [("Overall Driveability", overall, "Mean of criteria")]
        report_scores += [(name, c["rating"],
                           (f"{c['metric']:.3f} {c['unit']}" if c["unit"] != "-"
                            else f"param {c['metric']:.3f}")) for name, c in criteria.items()]
        report_bytes, report_mime, report_ext = build_summary_report(report_meta, report_scores)
        safe_stem = os.path.splitext(uploaded_file.name)[0]
        st.sidebar.markdown("---")
        st.sidebar.download_button(
            "📥 Export AVL-Style Executive Summary Report", data=report_bytes,
            file_name=f"drivability_summary_{safe_stem}.{report_ext}",
            mime=report_mime, use_container_width=True)

        # ---- Diagnostic tabs -------------------------------------------------
        st.markdown("### 🎛️ Diagnostic Analytics")
        tab1, tab2, tab3 = st.tabs(
            ["Disturbance Time Traces", "Frequency Mapping (PSD)", "Transient Response Microscope"])

        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["timestamp"], y=dist["disturbances_lf"]["signal"],
                                     name="Disturbances LF (2–10 Hz)", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=df["timestamp"], y=dist["disturbances_hf"]["signal"],
                                     name="Disturbances HF (>10 Hz)", line=dict(color="crimson")))
            fig.update_layout(title="Isolated Acceleration Disturbances (AccelerationChassisCompensated)",
                              xaxis_title="Time (s)", yaxis_title="Acceleration (m/s²)", height=420)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            c1, c2 = st.columns(2)
            render_psd_block(c1, LP(accel_comp.to_numpy(float), 2.0), band=(0.3, 2.0),
                             title="Surge (<2 Hz)", color="orange", caption_fn=interpret_surge_source)
            render_psd_block(c2, dist["disturbances_lf"]["signal"], band=(2.0, 10.0),
                             title="Disturbances LF (2–10 Hz)", color="blue")
            render_psd_block(st, dist["disturbances_hf"]["signal"], band=(10.0, 50.0),
                             title="Disturbances HF (10–50 Hz)", color="crimson")

        with tab3:
            tip_events = [ev for ev in events if ev["mode"] in ("Tip in", "Drive away")]
            if tip_events:
                labels = [f"{ev['mode']} @ {ev['t_start'] + 0.5:.2f}s" for ev in tip_events]
                sel = st.selectbox("Select transient event", options=list(range(len(labels))),
                                   format_func=lambda i: labels[i])
                ev = tip_events[sel]
                w = df[(df["timestamp"] >= ev["t_start"]) & (df["timestamp"] <= ev["t_end"] + 1.0)]
                ft = go.Figure()
                ft.add_trace(go.Scatter(x=w["timestamp"], y=w["AcceleratorPedal"],
                                        name="AcceleratorPedal (%)", line=dict(color="green"), yaxis="y2"))
                ft.add_trace(go.Scatter(x=w["timestamp"],
                                        y=SMO(w["AccelerationChassisCompensated"].to_numpy(float), 20),
                                        name="AccelerationChassis_SMO(20) (m/s²)",
                                        line=dict(color="black", width=2)))
                ft.update_layout(title="Pedal Tip-In vs Acceleration Response",
                                 xaxis_title="Time (s)",
                                 yaxis=dict(title="Acceleration (m/s²)"),
                                 yaxis2=dict(title="Pedal (%)", overlaying="y", side="right", range=[0, 100]),
                                 height=450)
                st.plotly_chart(ft, use_container_width=True)
            else:
                st.info("No transient (Tip-in / Drive-away) events available for microscope view.")

    except Exception as e:
        st.error(f"Execution Error: {e}")
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
