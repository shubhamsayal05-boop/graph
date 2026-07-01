"""Tests for the per-transmission operation-mode template catalog."""
from avldrive import (TEMPLATE_CATALOG, default_template_weights,
                      mode_weights_from_templates, parse_ect_filename,
                      templates_for)
from avldrive.config import TRANSMISSION_CONFIG


def test_every_transmission_has_templates():
    for tx in TRANSMISSION_CONFIG:
        assert templates_for(tx), f"{tx} has no templates"


def test_at_templates_match_ect_family():
    names = {e["template"] for e in templates_for("AT")}
    assert "AT_V46_TITO_AD.ect".replace(".ect", "") in {n for n in names}
    assert "AT_V46_Gearshift" not in names  # AT uses Manual_Power_* not Gearshift
    assert any(e["token"] == "TITO_AD" for e in templates_for("AT"))
    assert any("Manual_Power_On_Up" == e["token"] for e in templates_for("AT"))


def test_mt_templates_match_ect_family():
    tokens = {e["token"] for e in templates_for("MT")}
    assert "Gearshift" in tokens
    assert "TITO_AD_Brazil" in tokens
    assert "ESS" in tokens
    # MT has no torque-converter garage-shift template.
    assert "Garage_Shift" not in tokens


def test_bev_has_no_engine_or_gearshift_templates():
    modes = {e["mode"] for e in templates_for("BEV")}
    assert "Engine start" not in modes


def test_templates_differ_by_transmission():
    assert templates_for("AT") != templates_for("MT")
    assert {e["token"] for e in templates_for("AT")} != {e["token"] for e in templates_for("BEV")}


def test_weight_rollup_uses_strongest_template():
    tx = "AT"
    tw = default_template_weights(tx)
    # Force one Tip-in template to weight 5 and confirm the Tip in mode weight reflects it.
    for e in templates_for(tx):
        if e["mode"] == "Tip in":
            tw[e["template"]] = 5
    mw = mode_weights_from_templates(tx, tw)
    assert mw["Tip in"] == 5
    # Modes without templates keep a sensible default.
    assert mw["Acceleration"] >= 1


def test_parse_ect_filename():
    assert parse_ect_filename("AT_V46_TITO_AD.ect") == ("AT", "TITO_AD")
    assert parse_ect_filename("MT_V46_Manual_Power_On_Up.ect") == ("MT", "Manual_Power_On_Up")
    assert parse_ect_filename("not-a-template.txt") is None
