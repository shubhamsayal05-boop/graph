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
# 1. CHANNEL LAYOUT CONFIGURATION
# -----------------------------------------------------------------------------
CHANNEL_MAPPING = {
    "long_accel": "Ax_Sensor_g",         # Target longitudinal accelerometer channel
    "vert_accel": "Az_Sensor_g",         # Target vertical accelerometer channel
    "pedal_pos": "Acc_Pedal_Pct",        # Target accelerator pedal position (%)
    "engine_speed": "Engine_RPM",        # Target engine/motor speed channel
    "gear_status": "Current_Gear"        # Target gear status channel
}

# -----------------------------------------------------------------------------
# 1b. VEHICLE CONFIGURATION PROFILES (multi-vehicle baselines)
# -----------------------------------------------------------------------------
# Each profile defines how aggressively RMS error and response delay are
# penalised when translating into the 1-10 rating scale. A "Sports Car" holds
# the sharpest target (small faults cost more points, response delay is
# punished severely), a "Luxury Sedan" is a balanced comfort-biased baseline,
# and an "Eco EV" is the most forgiving on latency (heavier / efficiency-first
# calibration where a longer, smoother torque build-up is acceptable).
#   score_delay = 10 - mean_delay_s * delay_coeff
# Reference points at a 200 ms delay:
#     Luxury Sedan: 10 - 0.2 * 12.0 = 7.6
#     Sports Car:   10 - 0.2 * 26.0 = 4.8   (delay punished far more severely)
#     Eco EV:       10 - 0.2 *  6.0 = 8.8
MODE_PROFILES = {
    "Luxury Sedan": {"delay": 12.0, "surge": 4.5, "lf": 3.5, "hf": 2.5},
    "Sports Car": {"delay": 26.0, "surge": 6.5, "lf": 5.0, "hf": 4.0},
    "Eco EV": {"delay": 6.0, "surge": 3.5, "lf": 3.0, "hf": 2.5},
}

st.set_page_config(page_title="AI Automotive Calibration Lab", layout="wide")
st.title("⚙️ Advanced AI Automotive Calibration & Benchmarking Engine")
st.subheader("Comprehensive Diagnostic Suite (Response Delay, Surge, LF/HF Disturbances)")

# -----------------------------------------------------------------------------
# 2. ADVANCED SIGNAL FILTERS & MATH PROCESSING
# -----------------------------------------------------------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    """Generates standard bandpass filter coefficients."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(data, lowcut, highcut, fs, order=4):
    """Applies a zero-phase bandpass filter to isolate specific NVH/drivability faults."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, data)

def process_advanced_metrics(df, fs=100.0):
    """Advanced feature engineering matrix to separate distinct vehicle issues."""
    g_to_ms2 = 9.81
    ax_raw = df['long_accel'] * g_to_ms2
    
    # 1. Standard Filtering
    df['ax_base_filtered'] = butter_lowpass_filter(ax_raw, cutoff=25.0, fs=fs)
    dt = 1.0 / fs
    df['jerk'] = np.gradient(df['ax_base_filtered'], dt)
    
    # 2. Extract Surge (0.5 to 2.0 Hz)
    df['surge_signal'] = apply_bandpass(ax_raw, lowcut=0.5, highcut=2.0, fs=fs)
    
    # 3. Extract Acceleration Disturbances LF (2.0 to 8.0 Hz) - Driveline shuffle
    df['disturbances_lf'] = apply_bandpass(ax_raw, lowcut=2.0, highcut=8.0, fs=fs)
    
    # 4. Extract Acceleration Disturbances HF (8.0 to 20.0 Hz) - High frequency harshness
    df['disturbances_hf'] = apply_bandpass(ax_raw, lowcut=8.0, highcut=20.0, fs=fs)
    
    return df

def compute_band_spectrum(signal_data, fs=100.0, band=(0.5, 2.0)):
    """Runs a Welch periodogram over a filtered signal and isolates the
    dominant frequency peak inside the requested band.

    Returns a dict with the full spectrum (for plotting) and the dominant peak
    frequency/power inside the band, or ``None`` if the slice is too short.
    """
    signal = np.asarray(signal_data, dtype=float)
    signal = signal[np.isfinite(signal)]

    # Choose a segment length that gives good low-frequency resolution while
    # still fitting within the captured drive slice. A larger nperseg tightens
    # the frequency resolution around the band of interest.
    nperseg = int(min(len(signal), 1024))
    if nperseg < 16:
        return None

    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    low, high = band
    band_mask = (freqs >= low) & (freqs <= high)
    if not np.any(band_mask):
        return None

    band_freqs = freqs[band_mask]
    band_psd = psd[band_mask]
    peak_idx = int(np.argmax(band_psd))

    return {
        "freqs": freqs,
        "psd": psd,
        "band": band,
        "dominant_freq": float(band_freqs[peak_idx]),
        "dominant_power": float(band_psd[peak_idx]),
    }

def interpret_surge_source(dominant_freq):
    """Rough physical interpretation of the dominant surge frequency. Low-order
    combustion firing / lugging surges tend to sit near the bottom of the band,
    whereas faster limit-cycle oscillations from an electric motor controller
    damping loop tend to push toward the top of the band."""
    if dominant_freq < 1.0:
        return (
            "Low-frequency chugging (< 1.0 Hz) — consistent with engine "
            "combustion firing / lugging loops or coarse driveline lash."
        )
    if dominant_freq < 1.5:
        return (
            "Mid-band surge (1.0-1.5 Hz) — mixed powertrain source; inspect "
            "both combustion torque delivery and controller damping."
        )
    return (
        "Faster surge oscillation (> 1.5 Hz) — consistent with an electric "
        "motor / e-drive controller damping (limit-cycle) loop."
    )

def render_psd_block(container, signal_data, band, title, color, fs=100.0, caption_fn=None):
    """Renders a PSD (Welch) sub-chart with the dominant band frequency called
    out as text above the plot. Returns the dominant frequency (Hz) or None."""
    spec = compute_band_spectrum(signal_data, fs=fs, band=band)
    low, high = band
    if spec is None:
        container.info(f"{title}: slice too short to resolve a dominant frequency peak.")
        return None

    dom = spec["dominant_freq"]
    # Dominant frequency stated as text directly above the chart.
    container.markdown(
        f"**{title} — dominant frequency: `{dom:.2f} Hz`** "
        f"(band {low:g}–{high:g} Hz, PSD peak {spec['dominant_power']:.3e} (m/s²)²/Hz)"
    )
    if caption_fn is not None:
        container.caption(caption_fn(dom))

    band_mask = (spec["freqs"] >= low) & (spec["freqs"] <= high)
    fig_psd = go.Figure()
    fig_psd.add_trace(go.Scatter(
        x=spec["freqs"][band_mask],
        y=spec["psd"][band_mask],
        name=f"{title} PSD",
        line=dict(color=color),
        fill="tozeroy",
    ))
    fig_psd.add_vline(
        x=dom,
        line=dict(color="crimson", dash="dash"),
        annotation_text=f"{dom:.2f} Hz",
        annotation_position="top",
    )
    fig_psd.update_layout(
        title=f"{title} Power Spectral Density (Welch)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="PSD (m/s²)²/Hz",
        height=320,
    )
    container.plotly_chart(fig_psd, use_container_width=True)
    return dom

def build_summary_report(meta, scores):
    """Builds an AVL-style executive summary report. Returns a
    ``(bytes, mime, extension)`` tuple. Uses reportlab for a PDF when available,
    otherwise falls back to a plain-text report so the export always works."""
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        )

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            leftMargin=2 * cm, rightMargin=2 * cm,
            topMargin=2 * cm, bottomMargin=2 * cm,
        )
        styles = getSampleStyleSheet()
        story = [
            Paragraph("AVL-Style Executive Calibration Summary", styles["Title"]),
            Spacer(1, 6),
            Paragraph(f"Generated: {generated}", styles["Normal"]),
            Spacer(1, 12),
            Paragraph("File Metadata", styles["Heading2"]),
        ]

        meta_rows = [["Field", "Value"]] + [[k, str(v)] for k, v in meta.items()]
        meta_table = Table(meta_rows, colWidths=[6 * cm, 9 * cm])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b57")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ]))
        story += [meta_table, Spacer(1, 16), Paragraph("Calibration Ratings (0-10)", styles["Heading2"])]

        score_rows = [["Metric", "Rating", "Detail"]] + [
            [name, f"{val:.1f} / 10", detail] for name, val, detail in scores
        ]
        score_table = Table(score_rows, colWidths=[5 * cm, 3 * cm, 7 * cm])
        score_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b57")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ]))
        story.append(score_table)

        doc.build(story)
        return buffer.getvalue(), "application/pdf", "pdf"
    except Exception:
        # Fallback: structured plain-text report (no external dependency).
        lines = [
            "AVL-STYLE EXECUTIVE CALIBRATION SUMMARY",
            "=" * 44,
            f"Generated: {generated}",
            "",
            "FILE METADATA",
            "-" * 44,
        ]
        lines += [f"{k:<24}: {v}" for k, v in meta.items()]
        lines += ["", "CALIBRATION RATINGS (0-10)", "-" * 44]
        lines += [f"{name:<24}: {val:>4.1f} / 10   {detail}" for name, val, detail in scores]
        text = "\n".join(lines) + "\n"
        return text.encode("utf-8"), "text/plain", "txt"

def calculate_response_delay(event_df, fs=100.0):
    """Calculates the exact dead-time (seconds) from pedal tip-in to vehicle physical acceleration."""
    # Find point where pedal crosses a 5% threshold
    pedal_trigger_idx = (event_df['pedal_pos'] > 5.0).idxmax()
    pedal_time = event_df.loc[pedal_trigger_idx, 'timestamp']
    
    # Slice search window from pedal trigger onwards
    post_trigger_df = event_df.loc[pedal_trigger_idx:]
    
    # Detect physical vehicle acceleration start (when filtered jerk crosses 0.5 m/s3)
    accel_start_mask = post_trigger_df['jerk'] > 0.5
    if accel_start_mask.any():
        accel_start_idx = accel_start_mask.idxmax()
        accel_time = post_trigger_df.loc[accel_start_idx, 'timestamp']
        delay = accel_time - pedal_time
        return max(0.0, delay)
    return 0.0

# -----------------------------------------------------------------------------
# 3. FILE PARSING PIPELINE
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🚗 Vehicle Configuration Profile")
vehicle_mode = st.sidebar.selectbox(
    "Vehicle Configuration Profile",
    options=list(MODE_PROFILES.keys()),
    index=list(MODE_PROFILES.keys()).index("Luxury Sedan"),
    help=(
        "Sets the baseline against which drivability faults are scored. "
        "'Sports Car' punishes response delay and disturbances far more "
        "severely, 'Luxury Sedan' is a balanced baseline, and 'Eco EV' is the "
        "most forgiving on latency."
    ),
)
profile = MODE_PROFILES[vehicle_mode]
st.sidebar.caption(
    f"Active profile: **{vehicle_mode}** — "
    f"delay penalty ×{profile['delay']:.1f}, "
    f"surge ×{profile['surge']:.1f}, LF ×{profile['lf']:.1f}, HF ×{profile['hf']:.1f}"
)

uploaded_file = st.sidebar.file_uploader("Upload Automotive Log File (.mf4, .dat)", type=["mf4", "dat"])

if uploaded_file is not None:
    temp_filename = f"temp_{uploaded_file.name}"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    try:
        with st.spinner("Extracting multi-rate CAN vectors and synchronizing matrix..."):
            mdf = MDF(temp_filename)
            channels_to_extract = list(CHANNEL_MAPPING.values())
            mdf_resampled = mdf.filter(channels_to_extract).resample(raster=0.01) # 100Hz Grid
            df = mdf_resampled.to_dataframe()
            
            reverse_mapping = {v: k for k, v in CHANNEL_MAPPING.items()}
            df = df.rename(columns=reverse_mapping).reset_index().rename(columns={"index": "timestamp"})
            
            # Run Mathematical Engine
            df = process_advanced_metrics(df, fs=100.0)
            
        st.sidebar.success("All channels locked and filtered!")

        # -----------------------------------------------------------------------------
        # 4. COMPREHENSIVE SCOREBOARD (AVL-DRIVE CONVERSION MAP)
        # -----------------------------------------------------------------------------
        # Compute RMS values of isolated error signals
        surge_rms = np.sqrt(np.mean(df['surge_signal']**2))
        lf_rms = np.sqrt(np.mean(df['disturbances_lf']**2))
        hf_rms = np.sqrt(np.mean(df['disturbances_hf']**2))
        
        # Human translation equations (Transforms root-mean-square errors into a 1-10 rating scale).
        # Penalty coefficients are driven by the selected vehicle mode profile.
        score_surge = max(1.0, min(10.0, 10.0 - (surge_rms * profile['surge'])))
        score_lf = max(1.0, min(10.0, 10.0 - (lf_rms * profile['lf'])))
        score_hf = max(1.0, min(10.0, 10.0 - (hf_rms * profile['hf'])))
        
        st.markdown(f"### 📊 Vehicle Calibration Quality Overview — *{vehicle_mode}*")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        m_col1.metric("Surge Rating (0.5-2Hz)", f"{round(score_surge, 1)} / 10", f"RMS: {round(surge_rms, 3)} m/s²")
        m_col2.metric("Disturbances LF (2-8Hz)", f"{round(score_lf, 1)} / 10", f"RMS: {round(lf_rms, 3)} m/s²")
        m_col3.metric("Disturbances HF (8-20Hz)", f"{round(score_hf, 1)} / 10", f"RMS: {round(hf_rms, 3)} m/s²")
        
        # Detect Tip-In instances for Transient Delay Engine
        pedal_derivative = np.gradient(df['pedal_pos'], 0.01)
        tip_in_timestamps = df[(pedal_derivative > 150) & (df['pedal_pos'] > 15)]['timestamp'].values
        
        # Simple clustering mechanism for tip-ins
        tip_ins = []
        if len(tip_in_timestamps) > 0:
            tip_ins.append(tip_in_timestamps[0])
            for t in tip_in_timestamps[1:]:
                if t - tip_ins[-1] > 4.0: tip_ins.append(t)
                
        # Calculate mean response lag over the drive file
        delays = []
        for t_start in tip_ins:
            ev_df = df[(df['timestamp'] >= t_start - 0.5) & (df['timestamp'] <= t_start + 2.0)].reset_index(drop=True)
            if not ev_df.empty:
                delays.append(calculate_response_delay(ev_df))
                
        mean_delay = np.mean(delays) if delays else 0.0
        # Response delay penalty scales with the selected vehicle profile.
        score_delay = max(1.0, min(10.0, 10.0 - (mean_delay * profile['delay'])))
        m_col4.metric("Response Delay Rating", f"{round(score_delay, 1)} / 10", f"Mean Lag: {int(mean_delay*1000)} ms")

        # Overall calibration rating (mean of the four sub-scores).
        score_overall = float(np.mean([score_surge, score_lf, score_hf, score_delay]))
        st.metric("🏁 Overall Calibration Rating", f"{round(score_overall, 1)} / 10")

        # -----------------------------------------------------------------------------
        # 4b. AUTOMATED EXECUTIVE SUMMARY REPORT (sidebar export)
        # -----------------------------------------------------------------------------
        duration_s = float(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]) if len(df) else 0.0
        report_meta = {
            "Source File": uploaded_file.name,
            "Vehicle Profile": vehicle_mode,
            "Analysis Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Log Duration (s)": f"{duration_s:.1f}",
            "Samples (100 Hz grid)": len(df),
            "Tip-In Events Detected": len(tip_ins),
            "Mean Response Lag (ms)": int(mean_delay * 1000),
        }
        report_scores = [
            ("Overall Calibration", score_overall, "Mean of sub-ratings"),
            ("Surge (0.5-2 Hz)", score_surge, f"RMS {surge_rms:.3f} m/s²"),
            ("Disturbances LF (2-8 Hz)", score_lf, f"RMS {lf_rms:.3f} m/s²"),
            ("Disturbances HF (8-20 Hz)", score_hf, f"RMS {hf_rms:.3f} m/s²"),
            ("Response Delay", score_delay, f"Mean lag {int(mean_delay * 1000)} ms"),
        ]
        report_bytes, report_mime, report_ext = build_summary_report(report_meta, report_scores)
        safe_stem = os.path.splitext(uploaded_file.name)[0]
        st.sidebar.markdown("---")
        st.sidebar.download_button(
            "📥 Export AVL-Style Executive Summary Report",
            data=report_bytes,
            file_name=f"calibration_summary_{safe_stem}.{report_ext}",
            mime=report_mime,
            use_container_width=True,
        )

        # -----------------------------------------------------------------------------
        # 5. DIAGNOSTIC GRAPH ARCHITECTURE
        # -----------------------------------------------------------------------------
        st.markdown("### 🎛️ Dynamic Spectrum Analytics")
        tab1, tab2, tab3 = st.tabs(["Surge & Low-Freq Analysis", "Harshness & High-Freq Analysis", "Transient Response Microscope"])
        
        with tab1:
            fig_low = go.Figure()
            fig_low.add_trace(go.Scatter(x=df['timestamp'], y=df['surge_signal'], name="Surge (Chugging Components)", line=dict(color="orange")))
            fig_low.add_trace(go.Scatter(x=df['timestamp'], y=df['disturbances_lf'], name="Disturbances LF (Driveline Shuffle)", line=dict(color="blue")))
            fig_low.update_layout(title="Isolated Powertrain Structural Backlash & Surge Signals", xaxis_title="Time (s)", yaxis_title="Acceleration (m/s²)", height=400)
            st.plotly_chart(fig_low, use_container_width=True)

            # -----------------------------------------------------------------
            # FREQUENCY MAPPING (Welch PSD) — Surge (0.5-2Hz) & LF (2-8Hz)
            # -----------------------------------------------------------------
            st.markdown("#### 🔬 Frequency Mapping (Welch PSD)")
            psd_col1, psd_col2 = st.columns(2)
            surge_dom = render_psd_block(
                psd_col1, df['surge_signal'], band=(0.5, 2.0),
                title="Surge (0.5–2 Hz)", color="orange", fs=100.0,
                caption_fn=interpret_surge_source,
            )
            lf_dom = render_psd_block(
                psd_col2, df['disturbances_lf'], band=(2.0, 8.0),
                title="Disturbances LF (2–8 Hz)", color="blue", fs=100.0,
            )
            
        with tab2:
            fig_high = go.Figure()
            fig_high.add_trace(go.Scatter(x=df['timestamp'], y=df['disturbances_hf'], name="Disturbances HF (Harshness Components)", line=dict(color="crimson")))
            fig_high.update_layout(title="High-Frequency Signal Noise & Combustion Disturbance Profile", xaxis_title="Time (s)", yaxis_title="Acceleration (m/s²)", height=400)
            st.plotly_chart(fig_high, use_container_width=True)

            # -----------------------------------------------------------------
            # FREQUENCY MAPPING (Welch PSD) — HF harshness (8-20Hz)
            # -----------------------------------------------------------------
            st.markdown("#### 🔬 Harshness Frequency Mapping (Welch PSD)")
            hf_dom = render_psd_block(
                st, df['disturbances_hf'], band=(8.0, 20.0),
                title="Disturbances HF (8–20 Hz)", color="crimson", fs=100.0,
            )
            
        with tab3:
            if tip_ins:
                selected_event = st.selectbox("Select Tip-In Event Timestamp", options=tip_ins)
                window_df = df[(df['timestamp'] >= selected_event - 0.5) & (df['timestamp'] <= selected_event + 3.0)]
                
                fig_trans = go.Figure()
                fig_trans.add_trace(go.Scatter(x=window_df['timestamp'], y=window_df['pedal_pos'], name="Pedal Input (%)", line=dict(color="green"), yaxis="y2"))
                fig_trans.add_trace(go.Scatter(x=window_df['timestamp'], y=window_df['ax_base_filtered'], name="Vehicle Acceleration (m/s²)", line=dict(color="black", width=2)))
                fig_trans.update_layout(
                    title="Pedal Step Input vs Acceleration Launch Response",
                    xaxis_title="Time (s)",
                    yaxis=dict(title="Vehicle Acceleration (m/s²)"),
                    yaxis2=dict(title="Pedal Input (%)", overlaying="y", side="right", range=[0, 100]),
                    height=450
                )
                st.plotly_chart(fig_trans, use_container_width=True)
            else:
                st.info("No explicit dynamic pedal tip-in maneuvers found in this log slice.")

    except Exception as e:
        st.error(f"Execution Error: {e}")
    finally:
        # Clean up the temporary log file written to disk during upload.
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
else:
    st.info("⬅️ Upload a `.mf4` or `.dat` log file from the sidebar to begin analysis.")
