"""AVL-DRIVE-style objective drivability assessment — Streamlit UI.

Thin presentation layer over the ``avldrive`` package. The user selects a
transmission/powertrain architecture; the tool re-assigns its operation modes,
criteria, propulsion and channels, then rates imported measurements with an
AVL-style weighted + extreme-value DRIVE-Rating tree. Supports multi-measurement
benchmarking, a live weight-tree editor and diagnostic frequency analysis.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import avldrive as avl
from avldrive.criteria import event_criteria
from avldrive.dsp import LP, SMO
from avldrive.criteria import disturbance_metrics

st.set_page_config(page_title="AVL-DRIVE-Style Drivability Lab", layout="wide",
                   page_icon="⚙️")


# =============================================================================
# Cached processing
# =============================================================================
@st.cache_data(show_spinner=False)
def inspect_channels(file_bytes: bytes, filename: str):
    """List every raw channel in a measurement (cached)."""
    suffix = os.path.splitext(filename)[1] or ".mf4"
    return avl.list_channels_from_bytes(file_bytes, suffix)


@st.cache_data(show_spinner=False)
def process_file(file_bytes: bytes, filename: str, cfg: dict, dna: dict,
                 mode_weights: dict, criteria_weights: dict, enabled_modes: tuple,
                 mapping: dict):
    """Import + assess a single measurement (cached on all inputs)."""
    suffix = os.path.splitext(filename)[1] or ".mf4"
    df, found = avl.load_measurement_from_bytes(file_bytes, suffix, cfg, mapping)
    result = avl.run_assessment(df, list(enabled_modes), cfg, dna,
                                mode_weights, criteria_weights)
    return df, found, result


def channel_mapper(filename: str, available: list[str], auto: dict) -> dict:
    """Interactive logical→raw channel mapping. Returns the effective mapping.

    Defaults to auto-resolved channels; the expander opens automatically when a
    required channel is unmapped so the engineer can pick the right signal.
    """
    from avldrive.config import CHANNEL_CANDIDATES, REQUIRED_LOGICAL
    key = f"map_{filename}"
    prev = st.session_state.get(key, {})
    effective = {**auto, **{k: v for k, v in prev.items() if v}}
    missing = [c for c in REQUIRED_LOGICAL if c not in effective]

    with st.expander(f"🔌 Channel mapping — {filename}  ({len(available)} channels)",
                     expanded=bool(missing)):
        if missing:
            st.warning("Required channels not auto-detected: " + ", ".join(missing) +
                       ". Map them to the correct raw channels below.")
        with st.expander("Browse all channels in file"):
            st.dataframe(pd.DataFrame({"Channel": available}), use_container_width=True,
                         hide_index=True, height=240)
        options = ["(none)"] + available
        order = REQUIRED_LOGICAL + [c for c in CHANNEL_CANDIDATES if c not in REQUIRED_LOGICAL]
        mapping: dict[str, str] = {}
        cols = st.columns(2)
        for i, logical in enumerate(order):
            default = prev.get(logical) or auto.get(logical)
            idx = options.index(default) if default in options else 0
            required_tag = " *" if logical in REQUIRED_LOGICAL else ""
            sel = cols[i % 2].selectbox(f"{logical}{required_tag}", options, index=idx,
                                        key=f"{key}_{logical}")
            if sel != "(none)":
                mapping[logical] = sel
        st.session_state[key] = mapping
    return mapping


# =============================================================================
# UI sections
# =============================================================================
def select_transmission() -> tuple[str, dict]:
    st.markdown("### 1️⃣ Transmission / powertrain architecture")
    transmission = st.selectbox(
        "The tool re-assigns its operation modes, criteria, propulsion and "
        "relevant channels to the selected architecture.",
        options=list(avl.TRANSMISSION_CONFIG.keys()),
        format_func=lambda k: f"{k} — {avl.TRANSMISSION_CONFIG[k]['label']}",
        key="transmission")
    tx = avl.TRANSMISSION_CONFIG[transmission]

    a, b, c = st.columns(3)
    a.metric("Propulsion", tx["propulsion"])
    b.metric("Gearbox", tx["gearbox"])
    c.metric("Operation modes", len(tx["modes"]))
    templates = avl.templates_for(transmission)
    with st.expander(f"🔧 Assigned configuration — **{transmission} · {tx['label']}**", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Enabled operation modes**\n\n" + "\n".join(f"- {m}" for m in tx["modes"]))
        c2.markdown("**Powertrain features**\n\n" + "\n".join(f"- {f}" for f in tx["features"]))
        c3.markdown("**Relevant DRIVE channels**\n\n" + "\n".join(f"- {ch}" for ch in avl.relevant_channels(transmission)))
        st.markdown(f"**Loaded operation-mode templates ({len(templates)}) — {transmission} family**")
        st.dataframe(pd.DataFrame([
            {"Template": e["template"], "Sub-operation mode": e["label"],
             "Base mode": e["mode"], "Default weight": e["weight"]} for e in templates]),
            use_container_width=True, hide_index=True)
    return transmission, tx


def sidebar_config(transmission: str, tx: dict) -> tuple[dict, dict]:
    st.sidebar.header("🔧 Configuration Mode")
    st.sidebar.caption(f"Architecture: **{transmission} · {tx['label']}** ({tx['propulsion']})")
    default_brand = "Eco EV" if tx["propulsion"] == "BEV" else "Luxury Sedan"
    brand = st.sidebar.selectbox("Brand-DNA target", list(avl.BRAND_DNA.keys()),
                                 index=list(avl.BRAND_DNA.keys()).index(default_brand))
    drive = st.sidebar.selectbox("Driven axle", ["FWD", "RWD", "AWD"], index=0)

    st.sidebar.markdown("**Vehicle specific data**")
    mass = st.sidebar.number_input("Vehicle mass [kg]", 500.0, 4000.0, 1600.0, 10.0)
    wheel_radius = st.sidebar.number_input("Dynamic wheel radius [m]", 0.20, 0.60, 0.32, 0.01)
    with st.sidebar.expander("Road-load coefficients (F = A0 + B0·v + C0·v²)"):
        A0 = st.number_input("A0 [N]", 0.0, 1000.0, 120.0, 5.0)
        B0 = st.number_input("B0 [N/(km/h)]", 0.0, 50.0, 0.0, 0.5)
        C0 = st.number_input("C0 [N/(km/h)²]", 0.0, 1.0, 0.045, 0.005, format="%.3f")
    clutch = None
    if transmission == "MT":
        with st.sidebar.expander("Clutch pedal operating points [%]"):
            clutch = {"disengaged": st.number_input("Disengaged", 0.0, 100.0, 80.0, 1.0),
                      "touchpoint": st.number_input("Touchpoint", 0.0, 100.0, 45.0, 1.0),
                      "engaged": st.number_input("Engaged", 0.0, 100.0, 20.0, 1.0)}

    cfg = avl.VehicleConfig(transmission=transmission, propulsion=tx["propulsion"], drive=drive,
                            brand=brand, mass=mass, wheel_radius=wheel_radius,
                            A0=A0, B0=B0, C0=C0, clutch=clutch).as_dict()
    return cfg, avl.BRAND_DNA[brand]


def weight_editor(transmission: str) -> tuple[dict, dict]:
    """Transmission-specific AVL weight-tree editor.

    Loads the selected transmission's operation-mode templates (the ``.ect``
    family), lets the engineer weight each one (0–5), and rolls those up into the
    operation-mode weights used by the DRIVE-Rating aggregation. Criteria weights
    are global. Supports JSON import/export.
    """
    st.sidebar.markdown("---")
    templates = avl.templates_for(transmission)
    st.session_state.setdefault("template_weights", {})
    if transmission not in st.session_state.template_weights:
        st.session_state.template_weights[transmission] = avl.default_template_weights(transmission)
    st.session_state.setdefault("criteria_weights", avl.default_criteria_weights())
    tw = st.session_state.template_weights[transmission]

    with st.sidebar.expander(f"⚖️ Weight-tree editor — {transmission} templates"):
        up = st.file_uploader("Import weights (JSON)", type=["json"], key=f"weights_up_{transmission}")
        if up is not None:
            try:
                data = json.load(up)
                imported = data.get("template_weights", {})
                tw.update(imported.get(transmission, imported if not any(k in avl.TEMPLATE_CATALOG for k in imported) else {}))
                st.session_state.criteria_weights.update(data.get("criteria_weights", {}))
                st.success("Weights imported.")
            except Exception as e:
                st.error(f"Invalid weights file: {e}")
        if st.button("Reset to AVL defaults", key=f"reset_{transmission}"):
            st.session_state.template_weights[transmission] = avl.default_template_weights(transmission)
            st.session_state.criteria_weights = avl.default_criteria_weights()
            tw = st.session_state.template_weights[transmission]

        st.caption(f"{len(templates)} operation-mode templates loaded for **{transmission}** (weights 0–5). "
                   "Each base mode takes the strongest template weight.")
        groups: dict[str, list] = {}
        for e in templates:
            groups.setdefault(e["mode"], []).append(e)
        for mode, entries in groups.items():
            st.markdown(f"**{mode}**")
            for e in entries:
                tw[e["template"]] = st.slider(e["label"], 0, 5, int(tw.get(e["template"], e["weight"])),
                                              key=f"tw_{e['template']}")

        st.caption("Criteria weights (0–5)")
        for c, meta in avl.CRITERIA_META.items():
            st.session_state.criteria_weights[c] = st.slider(
                meta["label"], 0, 5, int(st.session_state.criteria_weights.get(c, meta["weight"])),
                key=f"cw_{c}")

        st.download_button("Export weights (JSON)",
                           data=json.dumps({"transmission": transmission,
                                            "template_weights": {transmission: tw},
                                            "criteria_weights": st.session_state.criteria_weights}, indent=2),
                           file_name=f"avldrive_weights_{transmission}.json", mime="application/json")

    mode_weights = avl.mode_weights_from_templates(transmission, tw)
    st.session_state.mode_weights = mode_weights
    return mode_weights, st.session_state.criteria_weights


# =============================================================================
# Result rendering
# =============================================================================
def _mode_table(mode_results, mode_weights):
    rated = [(m, r) for m, r in mode_results.items() if r["dr"] is not None]
    rated.sort(key=lambda x: x[1]["dr"])
    return rated


def render_single(df, found, result, cfg, dna, tx, transmission, filename):
    mode_weights = st.session_state.mode_weights
    st.markdown(f"## 🏁 AVL-DRIVE Rating — *{cfg['brand']}* on **{transmission} ({tx['propulsion']})**")
    overall = result.overall
    st.metric("Overall AVL-DRIVE Rating (weighted + extreme-value)",
              f"{round(overall, 1) if overall else '—'} / 10")

    rated = _mode_table(result.mode_results, mode_weights)
    if rated:
        st.markdown("### 📊 Main Operation-Mode DRIVE Ratings")
        cols = st.columns(min(4, len(rated)))
        for i, (m, r) in enumerate(rated):
            cols[i % len(cols)].metric(m, f"{round(r['dr'], 1)} / 10",
                                       f"{r['n_events']} event(s) · w{mode_weights.get(m, 3)}")
        st.dataframe(pd.DataFrame([
            {"Operation Mode": m, "DRIVE Rating": round(r["dr"], 1), "Events": r["n_events"],
             "Mode Weight": mode_weights.get(m, 3)} for m, r in rated]),
            use_container_width=True, hide_index=True)

        st.markdown("### 🔬 Criteria breakdown")
        sel = st.selectbox("Operation mode", [m for m, _ in rated], key=f"cb_{filename}")
        crit = result.mode_results[sel]["criteria"]
        st.dataframe(pd.DataFrame([
            {"Criterion": v["label"], "DRIVE Rating": round(v["rating"], 1),
             "Metric": f"{v['metric']:.3f} {v['unit']}", "Weight": v["weight"]}
            for v in crit.values()]), use_container_width=True, hide_index=True)
    else:
        st.warning("No enabled operation modes were triggered in this measurement.")

    render_timeline(result.events)
    render_report_button(df, found, result, cfg, tx, transmission, filename)
    render_calibration_advisor(result)
    render_verification(df, result, dna)
    render_diagnostics(df, result, dna)


def render_timeline(events):
    st.markdown("### 🧭 Operation-mode timeline")
    if events:
        st.dataframe(pd.DataFrame([
            {"Operation Mode": ev["mode"], "Start [s]": round(ev["t_start"], 2),
             "End [s]": round(ev["t_end"], 2), "Duration [s]": round(ev["t_end"] - ev["t_start"], 2)}
            for ev in events]), use_container_width=True, hide_index=True)
    else:
        st.info("No operation-mode events triggered.")


def render_report_button(df, found, result, cfg, tx, transmission, filename):
    meta = {"Source File": filename, "Architecture": f"{transmission} — {tx['label']}",
            "Propulsion / Gearbox": f"{tx['propulsion']} / {tx['gearbox']}",
            "Brand-DNA Target": cfg["brand"],
            "Analysis Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Log Duration (s)": f"{result.duration_s:.1f}", "Samples (100 Hz)": result.n_samples,
            "Overall AVL-DRIVE Rating": f"{round(result.overall, 1) if result.overall else '—'} / 10",
            "Operation Modes Triggered": len(result.mode_results)}
    rows = [[m, round(r["dr"], 1) if r["dr"] else "—", r["n_events"],
             st.session_state.mode_weights.get(m, 3)] for m, r in result.mode_results.items()]
    rb, mime, ext = avl.build_summary_report(meta, rows)
    st.sidebar.markdown("---")
    st.sidebar.download_button("📥 Export Executive Summary Report", data=rb,
                               file_name=f"drivability_{transmission}_{os.path.splitext(filename)[0]}.{ext}",
                               mime=mime, use_container_width=True, key=f"dl_{filename}")


def render_diagnostics(df, result, dna):
    st.markdown("### 🎛️ Diagnostic analytics")
    accel = df["AccelerationChassisCompensated"].to_numpy(float)
    dm = disturbance_metrics(accel)
    t1, t2, t3, t4 = st.tabs(["Disturbance traces", "Frequency mapping (PSD)",
                              "Transient microscope", "Statistical trends"])
    with t1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["timestamp"], y=dm["lf"][1], name="Disturbances LF (2–10 Hz)", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=df["timestamp"], y=dm["hf"][1], name="Disturbances HF (>10 Hz)", line=dict(color="crimson")))
        fig.update_layout(title="Isolated acceleration disturbances (AccelerationChassisCompensated)",
                          xaxis_title="Time (s)", yaxis_title="Acceleration (m/s²)", height=420)
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        c1, c2 = st.columns(2)
        _psd(c1, LP(accel, 2.0), (0.3, 2.0), "Surge (<2 Hz)", "orange", avl.interpret_surge_source)
        _psd(c2, dm["lf"][1], (2.0, 10.0), "Disturbances LF (2–10 Hz)", "blue")
        _psd(st, dm["hf"][1], (10.0, 50.0), "Disturbances HF (10–50 Hz)", "crimson")
    with t3:
        tips = [ev for ev in result.events if ev["mode"] in ("Tip in", "Drive away")]
        if tips:
            labels = [f"{ev['mode']} @ {ev['t_start'] + 0.5:.2f}s" for ev in tips]
            i = st.selectbox("Transient event", list(range(len(labels))), format_func=lambda i: labels[i])
            ev = tips[i]
            w = df[(df["timestamp"] >= ev["t_start"]) & (df["timestamp"] <= ev["t_end"] + 1.0)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=w["timestamp"], y=w["AcceleratorPedal"], name="AcceleratorPedal (%)",
                                     line=dict(color="green"), yaxis="y2"))
            fig.add_trace(go.Scatter(x=w["timestamp"], y=SMO(w["AccelerationChassisCompensated"].to_numpy(float), 20),
                                     name="AccelerationChassis_SMO(20) (m/s²)", line=dict(color="black", width=2)))
            fig.update_layout(title="Pedal tip-in vs acceleration response", xaxis_title="Time (s)",
                              yaxis=dict(title="Acceleration (m/s²)"),
                              yaxis2=dict(title="Pedal (%)", overlaying="y", side="right", range=[0, 100]), height=450)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No transient (Tip-in / Drive-away) events available.")
    with t4:
        render_trends(df, result, dna)


def render_trends(df, result, dna):
    modes = [m for m, r in result.mode_results.items() if r["n_events"] > 0]
    if not modes:
        st.info("No rated operation modes to trend.")
        return
    mode = st.selectbox("Operation mode", modes, key="trend_mode")
    events = [ev for ev in result.events if ev["mode"] == mode]
    rows = []
    for ev in events:
        seg = df[(df["timestamp"] >= ev["t_start"]) & (df["timestamp"] <= ev["t_end"])]
        crit = event_criteria(seg, mode, dna)
        for c, v in crit.items():
            rows.append({"criterion": avl.CRITERIA_META[c]["label"], "rating": v["rating"],
                         "t_start": round(ev["t_start"], 1)})
    if not rows:
        st.info("Not enough event data to trend.")
        return
    tdf = pd.DataFrame(rows)
    crit_sel = st.selectbox("Criterion", sorted(tdf["criterion"].unique()), key="trend_crit")
    sub = tdf[tdf["criterion"] == crit_sel]
    c1, c2 = st.columns(2)
    fig1 = go.Figure(go.Histogram(x=sub["rating"], nbinsx=10, marker_color="teal"))
    fig1.update_layout(title=f"Rating distribution — {crit_sel}", xaxis_title="DRIVE Rating",
                       yaxis_title="Event count", height=340, xaxis=dict(range=[1, 10]))
    c1.plotly_chart(fig1, use_container_width=True)
    fig2 = go.Figure(go.Scatter(x=sub["t_start"], y=sub["rating"], mode="markers+lines", marker_color="teal"))
    fig2.update_layout(title=f"Rating over time — {crit_sel}", xaxis_title="Event start (s)",
                       yaxis_title="DRIVE Rating", height=340, yaxis=dict(range=[1, 10]))
    c2.plotly_chart(fig2, use_container_width=True)


def _psd(container, signal, band, title, color, caption_fn=None):
    spec = avl.compute_band_spectrum(signal, band)
    low, high = band
    if spec is None:
        container.info(f"{title}: slice too short to resolve a dominant peak.")
        return
    dom = spec["dominant_freq"]
    container.markdown(f"**{title} — dominant frequency: `{dom:.2f} Hz`** "
                       f"(band {low:g}–{high:g} Hz, PSD peak {spec['dominant_power']:.3e} (m/s²)²/Hz)")
    if caption_fn:
        container.caption(caption_fn(dom))
    mask = (spec["freqs"] >= low) & (spec["freqs"] <= high)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spec["freqs"][mask], y=spec["psd"][mask], name=f"{title} PSD",
                             line=dict(color=color), fill="tozeroy"))
    fig.add_vline(x=dom, line=dict(color="crimson", dash="dash"),
                  annotation_text=f"{dom:.2f} Hz", annotation_position="top")
    fig.update_layout(title=f"{title} PSD (Welch)", xaxis_title="Frequency (Hz)",
                      yaxis_title="PSD (m/s²)²/Hz", height=320)
    container.plotly_chart(fig, use_container_width=True)


def render_benchmark(results_by_file, mode_weights):
    st.markdown("## 📈 Benchmark comparison")
    overall_rows = [{"Measurement": name, "Overall DR": round(r["result"].overall, 1) if r["result"].overall else None}
                    for name, r in results_by_file.items()]
    odf = pd.DataFrame(overall_rows)
    fig = go.Figure(go.Bar(x=odf["Measurement"], y=odf["Overall DR"], marker_color="#1f77b4",
                           text=odf["Overall DR"], textposition="outside"))
    fig.update_layout(title="Overall AVL-DRIVE Rating by measurement", yaxis=dict(range=[0, 10]),
                      yaxis_title="DRIVE Rating", height=380)
    st.plotly_chart(fig, use_container_width=True)

    all_modes = sorted({m for r in results_by_file.values() for m in r["result"].mode_results})
    grid = []
    for m in all_modes:
        row = {"Operation Mode": m, "Weight": mode_weights.get(m, 3)}
        for name, r in results_by_file.items():
            mr = r["result"].mode_results.get(m)
            row[name] = round(mr["dr"], 1) if mr and mr["dr"] is not None else None
        grid.append(row)
    st.markdown("### Per-operation-mode DRIVE ratings")
    st.dataframe(pd.DataFrame(grid), use_container_width=True, hide_index=True)


# =============================================================================
# Calibration / Verification / Benchmark features
# =============================================================================
def render_calibration_advisor(result):
    st.markdown("### 🛠️ Calibration Advisor")
    mode_weights = st.session_state.mode_weights
    base_overall, opps = avl.improvement_opportunities(result.mode_results, mode_weights)
    if not opps:
        st.success("All assessed criteria already meet the improvement target (≥ 9.0). "
                   "No high-priority calibration actions.")
    else:
        st.caption("Ranked by how much the **overall** AVL-DRIVE Rating would rise if each "
                   "criterion were lifted to 9.0 (accounts for the weight tree + extreme-value weighting).")
        st.dataframe(pd.DataFrame([
            {"Priority": i + 1, "Operation Mode": o["mode"], "Criterion": o["label"],
             "Current DR": o["current_rating"], "Δ Overall if fixed": o["potential_gain"],
             "Metric": f"{o['metric']:.3f} {o['unit']}", "Mode w": o["mode_weight"]}
            for i, o in enumerate(opps[:12])]), use_container_width=True, hide_index=True)
        top = opps[0]
        st.info(f"**Top action — {top['mode']} · {top['label']}** "
                f"(potential +{top['potential_gain']:.2f} overall DR)\n\n{top['hint']}")
        with st.expander("Calibration lever hints for all flagged criteria"):
            seen = set()
            for o in opps:
                if o["criterion"] in seen:
                    continue
                seen.add(o["criterion"])
                st.markdown(f"- **{o['label']}** — {o['hint']}")

    with st.expander("Target-gap table (all criteria vs a target DR)"):
        target = st.slider("Target DRIVE Rating", 5.0, 10.0, 8.0, 0.5, key="advisor_target")
        gaps = avl.target_gaps(result.mode_results, target)
        st.dataframe(pd.DataFrame([
            {"Operation Mode": g["mode"], "Criterion": g["criterion"], "DR": g["rating"],
             "Gap to target": g["gap_to_target"], "Metric": f"{g['metric']:.3f} {g['unit']}"}
            for g in gaps]), use_container_width=True, hide_index=True)


def render_verification(df, result, dna):
    st.markdown("### ✅ Drivability Verification (acceptance gate)")
    preset = st.selectbox("Acceptance spec preset", list(avl.VERIFICATION_PRESETS.keys()),
                          key="verif_preset")
    base = avl.VERIFICATION_PRESETS[preset]
    c1, c2, c3 = st.columns(3)
    spec = {
        "overall_min": c1.number_input("Overall DR ≥", 1.0, 10.0, float(base["overall_min"]), 0.5),
        "mode_min": c2.number_input("Each mode DR ≥", 1.0, 10.0, float(base["mode_min"]), 0.5),
        "criterion_min": c3.number_input("Each criterion DR ≥", 1.0, 10.0, float(base["criterion_min"]), 0.5),
    }
    verdict = avl.verify(result.overall, result.mode_results, spec)
    if verdict["passed"]:
        st.success(f"**PASS** — all {verdict['n_checks']} requirements met for '{preset}'.")
    else:
        st.error(f"**FAIL** — {verdict['n_fail']} of {verdict['n_checks']} requirements not met for '{preset}'.")

    only_fail = st.checkbox("Show only failing requirements", value=True, key="verif_only_fail")
    rows = [i for i in verdict["items"] if i["actual"] is not None and (not only_fail or not i["pass"])]
    if rows:
        st.dataframe(pd.DataFrame([
            {"Requirement": i["requirement"], "Level": i["level"], "Actual DR": i["actual"],
             "Target": i["target"], "Result": "PASS" if i["pass"] else "FAIL"} for i in rows]),
            use_container_width=True, hide_index=True)
    else:
        st.caption("No failing requirements to show.")

    log = avl.issue_log(df, result.events, dna, criterion_min=spec["criterion_min"])
    st.markdown("#### 🚩 Worst-event issue log")
    if log:
        st.dataframe(pd.DataFrame([
            {"Operation Mode": r["mode"], "Start [s]": r["t_start"], "End [s]": r["t_end"],
             "Event DR": r["event_dr"], "Worst criterion": avl.CRITERIA_META.get(r["worst_criterion"], {}).get("label", r["worst_criterion"]),
             "Worst DR": r["worst_rating"]} for r in log]),
            use_container_width=True, hide_index=True)
    else:
        st.caption("No individual events below the criterion threshold.")

    verif_payload = {"preset": preset, "spec": spec, "passed": verdict["passed"],
                     "n_fail": verdict["n_fail"], "n_checks": verdict["n_checks"],
                     "items": verdict["items"]}
    st.download_button("📥 Export verification result (JSON)",
                       data=json.dumps(verif_payload, indent=2),
                       file_name="drivability_verification.json", mime="application/json",
                       key="verif_dl")


def _radar(modes, series):
    fig = go.Figure()
    for name, vals in series.items():
        r = [v if v is not None else 0 for v in vals]
        if not r:
            continue
        fig.add_trace(go.Scatterpolar(r=r + [r[0]], theta=modes + [modes[0]],
                                      fill="toself", name=name))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 10], visible=True)),
                      height=480, title="Drivability fingerprint — per-mode DRIVE Rating",
                      showlegend=True)
    return fig


def render_benchmark_library(current_ref):
    st.markdown("## 📚 Benchmark Library")
    if "benchmark_library" not in st.session_state:
        st.session_state.benchmark_library = []
    lib = st.session_state.benchmark_library

    c1, c2 = st.columns([2, 1])
    ref_name = c1.text_input("Reference name", value=current_ref["name"], key="ref_name")
    if c2.button("➕ Save current as reference"):
        rec = dict(current_ref)
        rec["name"] = ref_name or current_ref["name"]
        st.session_state.benchmark_library = [r for r in lib if r["name"] != rec["name"]] + [rec]
        lib = st.session_state.benchmark_library
        st.success(f"Saved reference '{rec['name']}' ({len(lib)} in library).")

    up = st.file_uploader("Import benchmark library (JSON)", type=["json"], key="lib_up")
    if up is not None:
        try:
            imported = avl.library_from_json(up.getvalue().decode("utf-8"))
            names = {r["name"] for r in lib}
            st.session_state.benchmark_library = lib + [r for r in imported if r["name"] not in names]
            lib = st.session_state.benchmark_library
            st.success(f"Imported {len(imported)} reference(s).")
        except Exception as e:
            st.error(f"Invalid library file: {e}")

    compare_set = [current_ref] + [r for r in lib if r["name"] != current_ref["name"]]
    if len(compare_set) >= 1:
        modes, series = avl.fingerprint(compare_set)
        if modes:
            st.plotly_chart(_radar(modes, series), use_container_width=True)
    if lib:
        st.markdown("### 🏆 Reference ranking")
        st.dataframe(pd.DataFrame(avl.ranking([current_ref] + lib)),
                     use_container_width=True, hide_index=True)
        st.download_button("📥 Export benchmark library (JSON)",
                           data=avl.library_to_json(lib), file_name="benchmark_library.json",
                           mime="application/json")
    else:
        st.caption("Save references to build a best-in-class benchmark library.")


def render_ab_compare(results_by_file):
    st.markdown("## 🔀 A/B Calibration Comparison (regression check)")
    names = list(results_by_file.keys())
    c1, c2, c3 = st.columns(3)
    a = c1.selectbox("Baseline (A)", names, index=0, key="ab_a")
    b = c2.selectbox("Candidate (B)", names, index=min(1, len(names) - 1), key="ab_b")
    thr = c3.number_input("Regression threshold [DR]", 0.1, 3.0, 0.5, 0.1, key="ab_thr")
    if a == b:
        st.caption("Select two different measurements to compare.")
        return
    cmp = avl.compare_results(results_by_file[a]["result"], results_by_file[b]["result"], thr)
    m1, m2, m3 = st.columns(3)
    m1.metric("Overall Δ (B − A)", f"{cmp['overall_delta']:+.2f}" if cmp["overall_delta"] is not None else "—",
              cmp["overall_verdict"])
    m2.metric("Modes improved", cmp["n_improvements"])
    m3.metric("Modes regressed", cmp["n_regressions"])
    st.markdown("### Per-operation-mode change")
    st.dataframe(pd.DataFrame([
        {"Operation Mode": r["mode"], f"A · {a}": r["a"], f"B · {b}": r["b"],
         "Δ (B−A)": r["delta"], "Verdict": r["verdict"]} for r in cmp["mode_rows"]],),
        use_container_width=True, hide_index=True)
    with st.expander("Per-criterion change"):
        st.dataframe(pd.DataFrame([
            {"Operation Mode": r["mode"], "Criterion": r["criterion"], f"A": r["a"], f"B": r["b"],
             "Δ (B−A)": r["delta"], "Verdict": r["verdict"]} for r in cmp["criterion_rows"]]),
            use_container_width=True, hide_index=True)


# =============================================================================
# Main
# =============================================================================
def main():
    st.title("⚙️ AVL-DRIVE-Style Objective Drivability Assessment")
    st.caption("Independent re-implementation of AVL-DRIVE™ 4.6 SR1 methodology · "
               f"engine v{avl.__version__}")

    transmission, tx = select_transmission()
    cfg, dna = sidebar_config(transmission, tx)
    mode_weights, criteria_weights = weight_editor(transmission)

    st.sidebar.markdown("---")
    uploads = st.sidebar.file_uploader("Import measurement(s) (.mf4, .dat)",
                                       type=["mf4", "dat"], accept_multiple_files=True)

    st.markdown("### 2️⃣ Import measurement(s) & assess")
    if not uploads:
        st.info("⬅️ Import one or more `.mf4` / `.dat` measurements in the sidebar. "
                "Upload several to benchmark them side by side.")
        return

    from avldrive.config import REQUIRED_LOGICAL
    results_by_file = {}
    for up in uploads:
        try:
            available = inspect_channels(up.getvalue(), up.name)
        except Exception as e:  # pragma: no cover - defensive UI guard
            st.error(f"**{up.name}** — could not read channels: {e}")
            continue

        auto = avl.resolve_channel_names(available)
        mapping = channel_mapper(up.name, available, auto)
        missing = [c for c in REQUIRED_LOGICAL if c not in mapping]
        if missing:
            st.warning(f"**{up.name}** — waiting for channel mapping "
                       f"(required: {', '.join(missing)}).")
            continue
        try:
            df, found, result = process_file(up.getvalue(), up.name, cfg, dna,
                                             mode_weights, criteria_weights,
                                             tuple(tx["modes"]), mapping)
            results_by_file[up.name] = {"df": df, "found": found, "result": result}
        except avl.MissingChannelsError as e:
            st.error(f"**{up.name}** — {e}")
        except Exception as e:  # pragma: no cover - defensive UI guard
            st.error(f"**{up.name}** — execution error: {e}")

    if not results_by_file:
        return

    if len(results_by_file) > 1:
        render_benchmark(results_by_file, mode_weights)
        st.markdown("---")
        render_ab_compare(results_by_file)
        st.markdown("---")
        sel = st.selectbox("Detailed view for measurement", list(results_by_file.keys()))
    else:
        sel = next(iter(results_by_file))

    r = results_by_file[sel]
    st.sidebar.success(f"Imported {len(r['found'])} DRIVE channels from {sel}.")
    with st.sidebar.expander("Resolved channel mapping"):
        st.write(r["found"])
    render_single(r["df"], r["found"], r["result"], cfg, dna, tx, transmission, sel)

    st.markdown("---")
    current_ref = avl.result_to_reference(sel, transmission, cfg["brand"], r["result"])
    render_benchmark_library(current_ref)


if __name__ == "__main__":
    main()
else:
    main()
