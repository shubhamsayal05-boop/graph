"""AVL-DRIVE-style objective drivability assessment engine.

A modular, independent re-implementation of AVL-DRIVE(TM) 4.6 SR1 methodology:
transmission-aware operation modes, AVL signal processing and frequency bands,
criteria with the 1-10 DRIVE-Rating direction, and a weighted + extreme-value
rating tree. The core is Streamlit-free and unit-testable; ``app.py`` provides
the interactive UI.
"""
from __future__ import annotations

from . import config, criteria, dsp, operation_modes, spectrum
from .assessment import aggregate_dr, assess
from .channels import build_calculated_channels, resolve_channels
from .config import (BRAND_DNA, CRITERIA_META, MODE_CRITERIA, MODE_WEIGHTS,
                     TRANSMISSION_CONFIG, VehicleConfig, default_criteria_weights,
                     default_mode_weights, relevant_channels)
from .operation_modes import detect_events
from .pipeline import (AssessmentResult, MissingChannelsError, load_measurement,
                       load_measurement_from_bytes, run_assessment)
from .reporting import build_summary_report
from .spectrum import compute_band_spectrum, interpret_surge_source

__version__ = "1.0.0"

__all__ = [
    "config", "dsp", "criteria", "operation_modes", "spectrum",
    "TRANSMISSION_CONFIG", "BRAND_DNA", "MODE_WEIGHTS", "MODE_CRITERIA", "CRITERIA_META",
    "VehicleConfig", "relevant_channels", "default_mode_weights", "default_criteria_weights",
    "resolve_channels", "build_calculated_channels", "detect_events",
    "assess", "aggregate_dr", "compute_band_spectrum", "interpret_surge_source",
    "build_summary_report", "load_measurement", "load_measurement_from_bytes",
    "run_assessment", "AssessmentResult", "MissingChannelsError",
]
