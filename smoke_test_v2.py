"""Smoke test V2 — validates metric math AND V2 coaching prompt routing.

Inherits all V1 profile tests unchanged.
Adds Section 2: coaching prompt state tests (no real Claude call needed).

V2 coaching states tested:
  state_first_upload     — no prior session → baseline framing
  state_improvement      — separation up ≥2° → acknowledge progress
  state_no_change        — separation flat (±1°) → honest plateau + focus
  state_regression       — separation down ≥2° → honest regression + fix

Claude is mocked: _call_claude() is patched to return a sentinel string
so tests run offline without hitting the API. What we're validating is
prompt routing and payload structure, not Claude's output quality.

Usage:
  python smoke_test_v2.py
"""

from __future__ import annotations

import json
import math
import unittest.mock as mock
from dataclasses import dataclass

import numpy as np
import pandas as pd

import snowlens_v2_analysis as sl


# ---------------------------------------------------------------------------
# Synthetic keypoint generator (unchanged from V1)
# ---------------------------------------------------------------------------

@dataclass
class SynthConfig:
    duration_s: float = 12.0
    fps: float = 30.0
    width: int = 720
    height: int = 1280
    real_separation_deg: float = 9.0
    real_banking_lead: float = 0.0
    real_shoulder_bounce_px: float = 5.0
    real_shoulder_jitter_deg: float = 3.0
    detector_noise_deg: float = 1.0
    detector_noise_px: float = 1.0
    rider_offset_norm: float = 0.0
    turn_period_s: float = 2.0
    rider_motion_amplitude_px: float = 90.0
    seed: int = 7


def synth_keypoints(cfg: SynthConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    n = int(cfg.duration_s * cfg.fps)
    t = np.arange(n) / cfg.fps
    w, h = cfg.width, cfg.height

    base_phase = np.sin(2 * np.pi * (t / cfg.turn_period_s))
    hip_center_x = w / 2 + cfg.rider_offset_norm * w + cfg.rider_motion_amplitude_px * base_phase
    hip_center_y = h * 0.62 + 4 * np.sin(2 * np.pi * (t / 0.5))

    future_phase = np.sin(2 * np.pi * ((t + cfg.real_banking_lead) / cfg.turn_period_s))
    shoulder_center_x = w / 2 + cfg.rider_offset_norm * w + cfg.rider_motion_amplitude_px * future_phase
    shoulder_center_y = (h * 0.42
                          + cfg.real_shoulder_bounce_px * rng.standard_normal(n)
                          + cfg.detector_noise_px * rng.standard_normal(n))

    hip_tilt = (8.0 * np.cos(2 * np.pi * (t / cfg.turn_period_s))
                + cfg.detector_noise_deg * rng.standard_normal(n))
    shoulder_tilt = (hip_tilt
                     - cfg.real_separation_deg * np.cos(2 * np.pi * (t / cfg.turn_period_s))
                     + cfg.real_shoulder_jitter_deg * rng.standard_normal(n)
                     + cfg.detector_noise_deg * rng.standard_normal(n))

    SH_HALF, HIP_HALF = 95, 70

    def pair(cx, cy, half, tilt):
        a = np.radians(tilt)
        return (cx - half * np.cos(a), cy - half * np.sin(a),
                cx + half * np.cos(a), cy + half * np.sin(a))

    l_sx, l_sy, r_sx, r_sy = pair(shoulder_center_x, shoulder_center_y, SH_HALF, shoulder_tilt)
    l_hx, l_hy, r_hx, r_hy = pair(hip_center_x, hip_center_y, HIP_HALF, hip_tilt)

    rows = []
    for i in range(n):
        row = {"frame": i, "time_s": i / cfg.fps, "person_conf": 0.85}
        for name in sl.COCO_KEYPOINTS:
            row[f"{name}_x"] = np.nan
            row[f"{name}_y"] = np.nan
            row[f"{name}_conf"] = 0.0
        for name, x, y in [("l_shoulder", l_sx[i], l_sy[i]),
                            ("r_shoulder", r_sx[i], r_sy[i]),
                            ("l_hip", l_hx[i], l_hy[i]),
                            ("r_hip", r_hx[i], r_hy[i])]:
            row[f"{name}_x"] = float(x)
            row[f"{name}_y"] = float(y)
            row[f"{name}_conf"] = 0.9
        rows.append(row)

    df = pd.DataFrame(rows)
    df.attrs["fps"] = cfg.fps
    df.attrs["width"] = cfg.width
    df.attrs["height"] = cfg.height
    return df


# ---------------------------------------------------------------------------
# V1 profiles (unchanged)
# ---------------------------------------------------------------------------

PROFILES = {
    "solid_all_good": SynthConfig(
        real_separation_deg=17.0, real_banking_lead=-0.05,
        real_shoulder_bounce_px=2.5, real_shoulder_jitter_deg=2.0,
        detector_noise_deg=0.8,
    ),
    "novice_all_bad": SynthConfig(
        real_separation_deg=2.0, real_banking_lead=0.45,
        real_shoulder_bounce_px=18.0, real_shoulder_jitter_deg=16.0,
        detector_noise_deg=1.5,
    ),
    "pure_separation_problem": SynthConfig(
        real_separation_deg=1.5, real_banking_lead=-0.05,
        real_shoulder_bounce_px=2.5, real_shoulder_jitter_deg=2.0,
    ),
    "pure_stability_problem": SynthConfig(
        real_separation_deg=17.0, real_banking_lead=-0.05,
        real_shoulder_bounce_px=20.0, real_shoulder_jitter_deg=18.0,
    ),
    "pure_lean_problem": SynthConfig(
        real_separation_deg=17.0, real_banking_lead=0.5,
        real_shoulder_bounce_px=2.5, real_shoulder_jitter_deg=2.0,
    ),
    "offcenter": SynthConfig(
        real_separation_deg=17.0, real_banking_lead=-0.05,
        real_shoulder_bounce_px=2.5, real_shoulder_jitter_deg=2.0,
        rider_offset_norm=0.28,
    ),
    "rejected_static": SynthConfig(rider_motion_amplitude_px=0.0),
    "rejected_short": SynthConfig(duration_s=2.0),
}

EXPECTED = {
    "solid_all_good":          {"separation": "Solid",        "early_inside_lean": "Solid",         "shoulder_stability": "Solid"},
    "novice_all_bad":          {"separation": "any",          "early_inside_lean": "Needs Work",    "shoulder_stability": "Needs Work"},
    "pure_separation_problem": {"separation": "Needs Work",   "early_inside_lean": "Solid",         "shoulder_stability": "Solid"},
    "pure_stability_problem":  {"separation": "Solid",        "early_inside_lean": "Solid",         "shoulder_stability": "Needs Work"},
    "pure_lean_problem":       {"separation": "Solid",        "early_inside_lean": "Needs Work",    "shoulder_stability": "Solid"},
    "offcenter":               {"separation": "Solid",        "early_inside_lean": "Not Evaluated", "shoulder_stability": "Solid"},
    "rejected_static":         {"_global": "rejected"},
    "rejected_short":          {"_global": "rejected"},
}


def run_profile(name: str, cfg: SynthConfig) -> dict:
    df = synth_keypoints(cfg)
    rej = sl.apply_rejection_gate(df)
    if not rej.passed:
        return {"global": "rejected", "reasons": rej.reasons, "levels": {}}
    metrics = [
        sl.compute_separation(df),
        sl.compute_early_inside_lean(df),
        sl.compute_shoulder_stability(df),
    ]
    levels = {m.name: sl.SEVERITY_LABEL[m.severity] for m in metrics}
    raws = {m.name: (None if (isinstance(m.raw_value, float) and np.isnan(m.raw_value))
                     else round(m.raw_value, 3)) for m in metrics}
    return {"global": "passed", "levels": levels, "raws": raws}


# ---------------------------------------------------------------------------
# Section 1 — V1 metric tests (unchanged logic)
# ---------------------------------------------------------------------------

def run_section_1() -> bool:
    print("=" * 96)
    print("SECTION 1 — Metric math (V1 profiles, unchanged)")
    print("=" * 96)
    print(f"{'profile':<26} {'global':<10} {'separation':<16} {'lean':<16} {'stability':<16} {'pass?':<6}")
    print("-" * 96)

    all_pass = True
    for name, cfg in PROFILES.items():
        result = run_profile(name, cfg)
        if result["global"] == "rejected":
            row = f"{name:<26} {'reject':<10} {'-':<16} {'-':<16} {'-':<16}"
            ok = EXPECTED[name].get("_global") == "rejected"
        else:
            lev = result["levels"]
            row = (f"{name:<26} {'pass':<10} "
                   f"{lev.get('separation','?'):<16} "
                   f"{lev.get('early_inside_lean','?'):<16} "
                   f"{lev.get('shoulder_stability','?'):<16}")
            exp = EXPECTED[name]
            ok = all(
                exp[m] in ("any", lev[m]) for m in ("separation", "early_inside_lean", "shoulder_stability")
                if m in exp
            )
        all_pass = all_pass and ok
        print(f"{row} {'✓' if ok else '✗ FAIL':<6}")

    print()
    print("Section 1:", "ALL PASS ✓" if all_pass else "FAILURES ABOVE ✗")
    return all_pass


# ---------------------------------------------------------------------------
# Section 2 — V2 coaching prompt state tests (Claude mocked)
# ---------------------------------------------------------------------------

CLAUDE_SENTINEL = "__CLAUDE_MOCK_OUTPUT__"

# Good-technique synthetic report for coaching state tests
_GOOD_CFG = PROFILES["solid_all_good"]


def _make_report(sport: str = "ski") -> sl.AnalysisReport:
    df = synth_keypoints(_GOOD_CFG)
    rej = sl.apply_rejection_gate(df)
    metrics = [
        sl.compute_separation(df),
        sl.compute_early_inside_lean(df),
        sl.compute_shoulder_stability(df),
    ]
    return sl.AnalysisReport(video="synth:coaching_test", sport=sport, rejection=rej, metrics=metrics)


def _get_sep_angle(report: sl.AnalysisReport) -> float:
    sep = next((m for m in report.metrics if m.name == "separation"), None)
    return round(sep.raw_value, 1) if sep and not math.isnan(sep.raw_value) else 10.0


COACHING_STATES = {
    # state_name: (prior_session or None, what to assert)
    # Assertions are on payload STRUCTURE and prompt routing — not Claude output text.
}


def run_section_2() -> bool:
    print()
    print("=" * 96)
    print("SECTION 2 — V2 coaching prompt routing (Claude mocked)")
    print("=" * 96)
    print(f"{'state':<25} {'claude_called':<15} {'has_coaching':<15} {'has_sep_angle':<15} {'prior_in_internal':<20} {'pass?'}")
    print("-" * 96)

    report = _make_report()
    sep_angle = _get_sep_angle(report)

    states = {
        "first_upload":  None,
        "improvement":   {"date": "Nov 12", "separation_angle": sep_angle - 6.0},
        "no_change":     {"date": "Nov 12", "separation_angle": sep_angle - 0.5},
        "regression":    {"date": "Nov 12", "separation_angle": sep_angle + 6.0},
    }

    # Expected prompt keywords per state
    expected_keywords = {
        "first_upload": "first session",
        "improvement":  "improved",
        "no_change":    "hasn't meaningfully changed",
        "regression":   "regressed",
    }

    all_pass = True

    with mock.patch.object(sl, "_call_claude", return_value=CLAUDE_SENTINEL) as mock_claude:
        for state_name, prior in states.items():
            payload = sl.generate_coach_output(report, prior_session=prior)

            claude_called    = mock_claude.called
            has_coaching     = payload.get("coaching") == CLAUDE_SENTINEL
            has_sep_angle    = payload.get("separation_angle") is not None
            prior_in_int     = payload["_internal"].get("prior_session") == prior
            prompt_used      = payload["_internal"].get("prompt_used", "")
            keyword          = expected_keywords[state_name]
            keyword_in_prompt = keyword.lower() in prompt_used.lower()
            claude_succeeded = payload["_internal"].get("claude_succeeded") is True

            ok = all([claude_called, has_coaching, has_sep_angle,
                      prior_in_int, keyword_in_prompt, claude_succeeded])
            all_pass = all_pass and ok

            print(f"{state_name:<25} {'yes' if claude_called else 'NO':<15} "
                  f"{'yes' if has_coaching else 'NO':<15} "
                  f"{'yes' if has_sep_angle else 'NO':<15} "
                  f"{'yes' if prior_in_int else 'NO':<20} "
                  f"{'✓' if ok else '✗ FAIL (keyword: ' + keyword + ')'}")

            # Reset mock call count between iterations
            mock_claude.reset_mock()

    print()
    print("Section 2:", "ALL PASS ✓" if all_pass else "FAILURES ABOVE ✗")
    return all_pass


# ---------------------------------------------------------------------------
# Section 3 — Claude fallback: if _call_claude returns None, V1 strings used
# ---------------------------------------------------------------------------

def run_section_3() -> bool:
    print()
    print("=" * 96)
    print("SECTION 3 — Claude fallback (API failure → V1 deterministic output)")
    print("=" * 96)

    report = _make_report()
    all_pass = True

    with mock.patch.object(sl, "_call_claude", return_value=None):
        payload = sl.generate_coach_output(report, prior_session=None)

        has_coaching       = bool(payload.get("coaching"))
        claude_succeeded   = payload["_internal"].get("claude_succeeded") is False
        coaching_non_empty = len(payload.get("coaching", "")) > 10

        ok = has_coaching and claude_succeeded and coaching_non_empty
        all_pass = ok

        print(f"  coaching present:      {'yes ✓' if has_coaching else 'NO ✗'}")
        print(f"  claude_succeeded=False:{'yes ✓' if claude_succeeded else 'NO ✗'}")
        print(f"  fallback text non-empty:{'yes ✓' if coaching_non_empty else 'NO ✗'}")
        print(f"  fallback text preview: \"{payload.get('coaching','')[:80]}...\"")

    print()
    print("Section 3:", "ALL PASS ✓" if all_pass else "FAILURES ✗")
    return all_pass


# ---------------------------------------------------------------------------
# Section 4 — Snowboard sport swap still works in V2
# ---------------------------------------------------------------------------

def run_section_4() -> bool:
    print()
    print("=" * 96)
    print("SECTION 4 — Snowboard sport swap in V2 payload")
    print("=" * 96)

    report_sb = _make_report(sport="snowboard")
    all_pass = True

    with mock.patch.object(sl, "_call_claude", return_value=CLAUDE_SENTINEL):
        payload = sl.generate_coach_output(report_sb, prior_session=None)

        sport_ok     = payload.get("sport") == "snowboard"
        prompt_used  = payload["_internal"].get("prompt_used", "")
        prompt_sb    = "snowboarder" in prompt_used and "board" in prompt_used

        ok = sport_ok and prompt_sb
        all_pass = ok

        print(f"  sport='snowboard' in payload: {'yes ✓' if sport_ok else 'NO ✗'}")
        print(f"  'snowboarder'/'board' in prompt: {'yes ✓' if prompt_sb else 'NO ✗'}")

    print()
    print("Section 4:", "ALL PASS ✓" if all_pass else "FAILURES ✗")
    return all_pass


# ---------------------------------------------------------------------------
# Representative payload print
# ---------------------------------------------------------------------------

def print_representative_payload():
    print()
    print("=" * 96)
    print("REPRESENTATIVE PAYLOAD — improvement state (Claude mocked)")
    print("=" * 96)
    report = _make_report()
    sep_angle = _get_sep_angle(report)
    prior = {"date": "Nov 12", "separation_angle": sep_angle - 6.0}

    with mock.patch.object(sl, "_call_claude", return_value="[Claude coaching output would appear here]"):
        payload = sl.generate_coach_output(report, prior_session=prior)

    # Trim _internal for readability
    display = {k: v for k, v in payload.items() if k != "_internal"}
    display["_internal (truncated)"] = {
        "claude_succeeded": payload["_internal"]["claude_succeeded"],
        "prior_session": payload["_internal"]["prior_session"],
    }
    print(json.dumps(display, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    s1 = run_section_1()
    s2 = run_section_2()
    s3 = run_section_3()
    s4 = run_section_4()
    print_representative_payload()

    print()
    print("=" * 96)
    overall = all([s1, s2, s3, s4])
    print("OVERALL:", "ALL SECTIONS PASS ✓" if overall else "ONE OR MORE SECTIONS FAILED ✗")


if __name__ == "__main__":
    main()
