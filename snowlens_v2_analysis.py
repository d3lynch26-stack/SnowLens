"""
SnowLens V2 — Body Position Coaching Engine
============================================

Follow-cam video  →  YOLO11 pose  →  3 metrics  →  Claude coaching output

V2 changes vs V1:
  - generate_coach_output() now calls Claude Haiku for natural coaching language
  - Accepts optional prior_session dict for history-aware output
  - Handles 4 prompt states: first upload, improvement, no change, regression
  - Falls back to V1 deterministic strings if Claude call fails
  - Returns 'coaching' (Claude text) and 'separation_angle' (store in Supabase)

V1 scope (LOCKED):
  - Follow-cam only (camera behind rider)
  - 3 metrics: Separation, Early Inside Lean, Shoulder Stability
  - Rejection gate runs FIRST; bad input never produces coaching
  - Snowboard variant = same metrics, light wording swap
  - All thresholds are tunable constants up top

Supabase integration:
  Before calling run(), fetch prior session:
    SELECT separation_angle, created_at
    FROM analyses
    WHERE user_id = $1
    ORDER BY created_at DESC LIMIT 1

  After run(), store:
    INSERT INTO analyses (user_id, separation_angle, ...)
    VALUES ($1, coach['separation_angle'], ...)

CLI:
  python snowlens_v2_analysis.py path/to/clip.mp4 --sport ski
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants — TUNABLES (these are the values instructor validation will move)
# ---------------------------------------------------------------------------

COCO_KEYPOINTS = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]

# Per-keypoint confidence threshold to trust a detection
KP_CONF_MIN = 0.40

# Rejection gate
MIN_DURATION_SEC = 4.0
MIN_DETECTION_RATE = 0.60          # ≥60% of frames must have a confident person
MIN_MOTION_SCORE = 0.020           # std of hip-center / frame width

# Separation thresholds (degrees, average across clip)
SEP_BANDS = {
    "needs_work_below": 3.0,
    "developing_below": 8.0,
    "good_below": 15.0,             # 8–15° = good
    # >15° = over-rotated
}

# Early-inside-lean: torso lateral offset normalized to frame width
LEAN_OFFSET_SIG = 0.030             # |offset| > 3% of frame width = meaningful
LEAN_LOOKAHEAD_FRAMES = 8           # ~0.27s lead at 30fps
LEAN_LEAD_RATE_GOOD = 0.10          # <10% of frames showing lead = good
LEAN_LEAD_RATE_DEVELOPING = 0.25

# Shoulder stability
STAB_ANGLE_GOOD = 8.0               # degrees of std
STAB_ANGLE_DEVELOPING = 15.0
STAB_BOUNCE_GOOD = 3.0              # % of frame height std

# User-facing tier labels (severity → display label)
SEVERITY_LABEL = {
    "good": "Solid",
    "developing": "Developing",
    "needs_work": "Needs Work",
    "n_a": "Not Evaluated",
}

KP = {n: i for i, n in enumerate(COCO_KEYPOINTS)}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RejectionResult:
    passed: bool
    reasons: List[str]
    detection_rate: float
    motion_score: float
    duration_sec: float


@dataclass
class MetricResult:
    name: str
    score: float                    # 0–100, higher = better
    severity: str                   # "good" | "developing" | "needs_work" | "n_a"
    raw_value: float
    headline: str
    detail: str
    frame_evidence: List[int] = field(default_factory=list)


@dataclass
class AnalysisReport:
    video: str
    sport: str
    rejection: RejectionResult
    metrics: List[MetricResult]


# ---------------------------------------------------------------------------
# Pose extraction
# ---------------------------------------------------------------------------

def extract_keypoints(video_path: str, model_name: str = "yolo11l-pose.pt") -> pd.DataFrame:
    """Run YOLO11-pose on the video; return a per-frame keypoint DataFrame.

    Columns: frame, time_s, person_conf, {kp}_x, {kp}_y, {kp}_conf for each COCO keypoint.
    Follow-cam assumption: pick the person closest to horizontal frame center.
    """
    from ultralytics import YOLO  # local import: cold path

    model = YOLO(model_name)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    rows = []
    for frame_idx, result in enumerate(model(video_path, stream=True, verbose=False)):
        row = {"frame": frame_idx, "time_s": frame_idx / fps}

        kp_obj = result.keypoints
        if kp_obj is None or kp_obj.xy is None or len(kp_obj.xy) == 0:
            row["person_conf"] = 0.0
            for n in COCO_KEYPOINTS:
                row[f"{n}_x"] = np.nan
                row[f"{n}_y"] = np.nan
                row[f"{n}_conf"] = 0.0
        else:
            xys = kp_obj.xy.cpu().numpy()           # (N, 17, 2)
            confs = kp_obj.conf.cpu().numpy()       # (N, 17)

            # Score each detected person: mean conf - distance-from-center penalty
            best_score, best_i = -1e9, 0
            for i in range(xys.shape[0]):
                valid = confs[i] > 0.10
                if not valid.any():
                    continue
                mean_x = float(xys[i, valid, 0].mean())
                center_penalty = abs(mean_x - w / 2) / w
                s = float(confs[i].mean()) - 0.5 * center_penalty
                if s > best_score:
                    best_score, best_i = s, i

            kp_xy = xys[best_i]
            kp_conf = confs[best_i]
            row["person_conf"] = float(kp_conf.mean())
            for i, n in enumerate(COCO_KEYPOINTS):
                if kp_conf[i] >= KP_CONF_MIN:
                    row[f"{n}_x"] = float(kp_xy[i, 0])
                    row[f"{n}_y"] = float(kp_xy[i, 1])
                else:
                    row[f"{n}_x"] = np.nan
                    row[f"{n}_y"] = np.nan
                row[f"{n}_conf"] = float(kp_conf[i])

        rows.append(row)

    df = pd.DataFrame(rows)
    df.attrs["fps"] = fps
    df.attrs["width"] = w
    df.attrs["height"] = h
    return df


# ---------------------------------------------------------------------------
# Rejection gate
# ---------------------------------------------------------------------------

def apply_rejection_gate(df: pd.DataFrame) -> RejectionResult:
    fps = df.attrs.get("fps", 30.0)
    w = df.attrs.get("width", 720)

    duration = len(df) / fps if fps > 0 else 0.0
    detection_rate = float((df["person_conf"] > 0.30).mean())

    hip_x = ((df["l_hip_x"] + df["r_hip_x"]) / 2).dropna()
    hip_y = ((df["l_hip_y"] + df["r_hip_y"]) / 2).dropna()
    if len(hip_x) < 5:
        motion_score = 0.0
    else:
        motion_score = float(np.hypot(hip_x.std(), hip_y.std()) / w)

    reasons: List[str] = []
    if duration < MIN_DURATION_SEC:
        reasons.append(f"Clip too short ({duration:.1f}s, need ≥{MIN_DURATION_SEC:.0f}s).")
    if detection_rate < MIN_DETECTION_RATE:
        reasons.append(
            f"Rider not detected consistently ({detection_rate*100:.0f}% of frames, "
            f"need ≥{MIN_DETECTION_RATE*100:.0f}%). Try clearer follow-cam framing."
        )
    if motion_score < MIN_MOTION_SCORE:
        reasons.append("Not enough motion. Is this an actual run, or a stationary clip?")

    return RejectionResult(
        passed=not reasons,
        reasons=reasons,
        detection_rate=detection_rate,
        motion_score=motion_score,
        duration_sec=duration,
    )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _line_angle_deg(x1, y1, x2, y2):
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))


def _angular_diff(a, b):
    """Smallest unsigned angular difference between two angles in degrees."""
    return np.abs(((a - b + 180) % 360) - 180)


def _interp_score(value, breakpoints, scores):
    return float(np.clip(np.interp(value, breakpoints, scores), 0, 100))


# ---------------------------------------------------------------------------
# Metric 1 — Separation
# ---------------------------------------------------------------------------

def compute_separation(df: pd.DataFrame) -> MetricResult:
    """Angular difference between shoulder line and hip line, averaged across clip.

    Follow-cam from behind: when a rider counter-rotates (good), the projected
    angles of the shoulder line and hip line diverge. Average separation across
    a clip is a reasonable scalar proxy for upper/lower body independence.
    """
    sh_raw = _line_angle_deg(df["l_shoulder_x"], df["l_shoulder_y"],
                              df["r_shoulder_x"], df["r_shoulder_y"])
    hp_raw = _line_angle_deg(df["l_hip_x"], df["l_hip_y"],
                              df["r_hip_x"], df["r_hip_y"])
    # Smooth BEFORE differencing — otherwise detector jitter inflates "separation"
    # and noisy upper bodies score the same as real counter-rotation.
    sh = sh_raw.rolling(7, center=True, min_periods=4).mean()
    hp = hp_raw.rolling(7, center=True, min_periods=4).mean()
    sep = _angular_diff(sh, hp)
    valid = sep.dropna()

    if len(valid) < 10:
        return MetricResult(
            name="separation", score=0.0, severity="n_a", raw_value=float("nan"),
            headline="Couldn't read your shoulders and hips clearly enough.",
            detail="Pose detection on the torso was inconsistent — try a clip with the rider centered and unobstructed.",
        )

    avg = float(valid.mean())
    score = _interp_score(avg, [0, 3, 8, 12, 18, 30], [10, 35, 75, 92, 88, 65])

    if avg < SEP_BANDS["needs_work_below"]:
        severity, headline, detail = (
            "needs_work",
            "Your legs and torso are moving as a single unit.",
            "When your skis change direction, your whole body changes direction with them. "
            "The skill to build here is independence: legs steer underneath you while the "
            "torso stays oriented down the hill. That's what lets the edges work.",
        )
    elif avg < SEP_BANDS["developing_below"]:
        severity, headline, detail = (
            "developing",
            "Your legs are starting to work independently of your torso, but the two halves are still linked.",
            "There are moments where the legs steer beneath a quieter torso, and moments "
            "where the whole body rotates together. Goal: make the independent moments the "
            "default, not the exception.",
        )
    elif avg < SEP_BANDS["good_below"]:
        severity, headline, detail = (
            "good",
            "Real upper/lower body independence — legs steering, torso facing down the hill.",
            "Through the turns your legs are working as the steering mechanism while your "
            "torso holds its own line. That's the structural ingredient that lets edges "
            "carve instead of skid.",
        )
    else:
        severity, headline, detail = (
            "developing",
            "Your torso is counter-rotating very aggressively against your legs.",
            "More angular difference than is usually productive. A small amount of "
            "counter-rotation is the engine of a turn; a lot can stall edge change and "
            "make initiation feel forced.",
        )

    top = valid.nlargest(3).index.tolist()
    return MetricResult(
        name="separation", score=score, severity=severity, raw_value=avg,
        headline=headline, detail=detail, frame_evidence=[int(i) for i in top],
    )


# ---------------------------------------------------------------------------
# Metric 2 — Early Inside Lean
# ---------------------------------------------------------------------------

def _lean_signal_is_trustworthy(df: pd.DataFrame) -> tuple:
    """Metric-specific quality check for early-inside-lean.

    Even when the overall clip passes the rejection gate, this metric's lateral
    signal can be corrupted by:
      - rider sitting off-center in frame the whole clip (follow-cam not holding subject)
      - motion that isn't actually periodic (panning camera, single-direction traverse)
      - shoulders + hips not jointly detected often enough
    Returns (ok, reason).
    """
    hpc = (df["l_hip_x"] + df["r_hip_x"]) / 2
    shc = (df["l_shoulder_x"] + df["r_shoulder_x"]) / 2
    w = df.attrs.get("width", 720)

    valid_pair = (~hpc.isna()) & (~shc.isna())
    if valid_pair.sum() < 60:                       # <2s of joint detection
        return False, "Couldn't detect shoulders and hips together for enough of the clip."

    hpc_valid = hpc[valid_pair]

    # 1. Centeredness: average hip position should be within ~20% of frame center
    off_center = abs(hpc_valid.mean() - w / 2) / w
    if off_center > 0.20:
        return False, "Rider is off-center in the frame — follow-cam wasn't holding the subject."

    # 2. Periodicity: hip motion should cross its own mean at least twice
    #    (i.e., at least one full left-right-left cycle observed)
    centered = hpc_valid - hpc_valid.mean()
    sign_changes = int(np.sum(np.diff(np.sign(centered.values)) != 0))
    if sign_changes < 2:
        return False, "Not enough left-right motion in this clip to read transition timing."

    return True, ""


def compute_early_inside_lean(df: pd.DataFrame) -> MetricResult:
    """Does the torso lead the hips laterally into a new turn?

    Method:
      offset(t)         = (shoulder_center_x − hip_center_x) / frame_width
      hip_vel_future(t) = smoothed d/dt of hip_center, shifted LOOKAHEAD frames ahead
      leading(t)        = |offset| meaningful AND sign(offset) == sign(future hip motion)
      lead_rate         = fraction of valid frames flagged

    Lower lead_rate = better (body stays stacked, edges drive the turn).

    A metric-specific quality check runs first — if camera framing or motion
    isn't clean enough for this metric specifically, we return n_a rather than
    a confidently-wrong reading.
    """
    ok, reason = _lean_signal_is_trustworthy(df)
    if not ok:
        return MetricResult(
            name="early_inside_lean", score=0.0, severity="n_a", raw_value=float("nan"),
            headline="Couldn't read your transitions reliably from this clip.",
            detail=reason + " Try a clip with the rider centered in frame and at least two full turns.",
        )
    shc = (df["l_shoulder_x"] + df["r_shoulder_x"]) / 2
    hpc = (df["l_hip_x"] + df["r_hip_x"]) / 2
    w = df.attrs.get("width", 720)

    offset = (shc - hpc) / w
    hip_vel = hpc.diff().rolling(7, center=True, min_periods=3).mean()
    hip_vel_future = hip_vel.shift(-LEAN_LOOKAHEAD_FRAMES)

    sig_offset = offset.abs() > LEAN_OFFSET_SIG
    sig_motion = hip_vel_future.abs() > 0.5
    same_sign = np.sign(offset) == np.sign(hip_vel_future)
    leading = (sig_offset & sig_motion & same_sign).astype(float)
    leading[~(sig_motion & ~hip_vel_future.isna())] = np.nan  # only score where future motion is observable

    valid = leading.dropna()
    if len(valid) < 30:
        return MetricResult(
            name="early_inside_lean", score=0.0, severity="n_a", raw_value=float("nan"),
            headline="Couldn't read your turn transitions cleanly enough.",
            detail="Need more frames with clear shoulder + hip detection through transitions.",
        )

    lead_rate = float(valid.mean())
    score = _interp_score(lead_rate, [0.0, 0.10, 0.25, 0.50, 0.80], [95, 78, 50, 20, 5])

    if lead_rate < LEAN_LEAD_RATE_GOOD:
        severity, headline, detail = (
            "good",
            "Your turns are initiating from the feet, with the body following.",
            "In transitions, your edge change happens first and your body responds after. "
            "That's angulation at work — the skis are leading you into the new turn instead "
            "of the other way around.",
        )
    elif lead_rate < LEAN_LEAD_RATE_DEVELOPING:
        severity, headline, detail = (
            "developing",
            "Most turns initiate from the feet, but some start with the body tipping in first.",
            "On a portion of your transitions you're committing the body to the new turn "
            "before the edges have changed. Not catastrophic — but on those turns you're "
            "banking, not angulating. Work on feeling the edge change happen underfoot before "
            "the body responds.",
        )
    else:
        severity, headline, detail = (
            "needs_work",
            "You're starting new turns by tipping the body into them, not by changing edges.",
            "The body is committing to the new turn before the feet have changed edges. "
            "That's banking — it works on mellow groomers but loses you grip the moment "
            "terrain steepens or firms up. Drill to try: deliberate edge change with hands "
            "out in front and the line between them held parallel to the horizon.",
        )

    top = offset.abs().dropna().nlargest(3).index.tolist()
    return MetricResult(
        name="early_inside_lean", score=score, severity=severity, raw_value=lead_rate,
        headline=headline, detail=detail, frame_evidence=[int(i) for i in top],
    )


# ---------------------------------------------------------------------------
# Metric 3 — Shoulder Stability
# ---------------------------------------------------------------------------

def compute_shoulder_stability(df: pd.DataFrame) -> MetricResult:
    """Composite: std of shoulder-line angle (rotation) + std of shoulder-center y (bounce).

    Light smoothing first to remove single-frame detector jitter.
    """
    sh_angle = _line_angle_deg(df["l_shoulder_x"], df["l_shoulder_y"],
                                df["r_shoulder_x"], df["r_shoulder_y"])
    sh_y_center = (df["l_shoulder_y"] + df["r_shoulder_y"]) / 2
    h = df.attrs.get("height", 1280)

    # Stability = how much the shoulders deviate from the SLOW trend.
    # Do NOT smooth then std — that would erase the high-frequency jitter
    # we're trying to measure. Instead: subtract a slow trend, std the residual.
    angle_trend = sh_angle.rolling(15, center=True, min_periods=5).mean()
    angle_resid = (sh_angle - angle_trend).dropna()

    y_trend = sh_y_center.rolling(15, center=True, min_periods=5).mean()
    y_resid = (sh_y_center - y_trend).dropna()

    if len(angle_resid) < 10 or len(y_resid) < 10:
        return MetricResult(
            name="shoulder_stability", score=0.0, severity="n_a", raw_value=float("nan"),
            headline="Couldn't read your shoulders cleanly enough.",
            detail="Need more frames with both shoulders confidently detected.",
        )

    angle_std = float(angle_resid.std())
    bounce_std_pct = float((y_resid.std() / h) * 100)

    angle_score = _interp_score(angle_std, [3, 8, 15, 25], [95, 75, 40, 15])
    bounce_score = _interp_score(bounce_std_pct, [1, 3, 6, 10], [95, 75, 40, 15])
    score = 0.7 * angle_score + 0.3 * bounce_score

    if angle_std < STAB_ANGLE_GOOD and bounce_std_pct < STAB_BOUNCE_GOOD:
        severity, headline, detail = (
            "good",
            "Your head and shoulders are a calm platform.",
            "Frame to frame there's almost no bounce and almost no jitter at the top of "
            "your body. A quiet platform is what your balance, vision, and fine adjustments "
            "all depend on — yours is doing its job.",
        )
    elif angle_std < STAB_ANGLE_DEVELOPING:
        severity, headline, detail = (
            "developing",
            "Your platform is mostly steady, but with visible bounce.",
            "Some frame-to-frame movement at the top of your body. Not enough to break "
            "anything else, but worth tightening. A quieter platform usually comes from "
            "balance, not from forcing the upper body still — stance, hand position, and "
            "core engagement do the work.",
        )
    else:
        severity, headline, detail = (
            "needs_work",
            "Your platform is bouncy and jittery through the run.",
            "Significant frame-to-frame movement at the top of your body. This usually "
            "signals you're out of balance and using upper-body movement to recover. "
            "Drill to try: ski with hands held out in front and consciously hold the line "
            "between them parallel to the horizon — it forces a steadier core.",
        )

    return MetricResult(
        name="shoulder_stability", score=score, severity=severity, raw_value=angle_std,
        headline=headline, detail=detail,
    )


# ---------------------------------------------------------------------------
# Coach output (sport-aware)
# ---------------------------------------------------------------------------

def _snowboard_swap(s: str) -> str:
    return (s.replace("skis", "board")
             .replace("your ski", "your board")
             .replace("edge change", "edge change")  # same term works
             .replace("carve", "carve"))



# ---------------------------------------------------------------------------
# V2 — Claude coaching engine
# ---------------------------------------------------------------------------

import math
import os

import anthropic as _anthropic

_claude_client = _anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
CLAUDE_MODEL = "claude-haiku-4-5-20251001"


def _build_coach_prompt(report: "AnalysisReport", prior_session: dict | None) -> str:
    sep  = next((m for m in report.metrics if m.name == "separation"), None)
    lean = next((m for m in report.metrics if m.name == "early_inside_lean"), None)
    stab = next((m for m in report.metrics if m.name == "shoulder_stability"), None)

    sep_angle   = round(sep.raw_value, 1)  if sep  and not math.isnan(sep.raw_value)  else None
    lean_bad    = lean and lean.severity in ("developing", "needs_work")
    stab_score  = round(stab.score, 0)     if stab and not math.isnan(stab.raw_value) else None

    sport       = report.sport
    sport_label = "skier" if sport == "ski" else "snowboarder"
    equipment   = "skis"  if sport == "ski" else "board"

    lines = []
    if sep_angle  is not None: lines.append(f"- Separation angle: {sep_angle}°")
    if lean       is not None: lines.append(f"- Early inside lean detected: {'Yes' if lean_bad else 'No'}")
    if stab_score is not None: lines.append(f"- Shoulder stability score: {int(stab_score)}/100")
    measurements_block = "\n".join(lines) if lines else "- Metrics not available"

    # History block + change instruction
    if prior_session and sep_angle is not None:
        prior_sep  = prior_session.get("separation_angle")
        prior_date = prior_session.get("date", "last session")
        if prior_sep is not None:
            history_block = f"\nPrevious session data:\n- {prior_date}: separation {round(prior_sep, 1)}°\n"
            delta = sep_angle - prior_sep
            if delta >= 2.0:
                change_instruction = (
                    "Separation has improved. Acknowledge it specifically and naturally "
                    "— like a coach who was watching in that previous session and remembers."
                )
            elif delta <= -2.0:
                change_instruction = (
                    "Separation has regressed since last session. Name it honestly. "
                    "Identify the likely cause from the other metrics. Give one specific fix. "
                    "Do not catastrophize. Do not over-encourage. "
                    "Be the coach who has seen this before and knows exactly how to fix it."
                )
            else:
                change_instruction = (
                    "Separation hasn't meaningfully changed. Address it honestly. "
                    "Identify what's likely holding it back from the other metrics. "
                    "Give one concrete thing to focus on."
                )
        else:
            history_block      = ""
            change_instruction = "Establish a baseline naturally. Set the tone like a coach meeting an athlete for the first time."
    else:
        history_block      = ""
        change_instruction = "This is the first session. Establish a baseline naturally. Set the tone like a coach meeting an athlete for the first time."

    return f"""Analyze this {sport_label}'s technique based on the following measurements:
{measurements_block}
{history_block}
{change_instruction}

Deliver feedback in the voice of a great human {sport_label} coach — direct, specific, 3-4 sentences maximum.

Do not mention "previous session data", "measurements", "metrics", or "scores".
Do not sound like you are reading from a database.
Use natural {equipment} coaching language.
Sound like you were there."""


def _call_claude(prompt: str) -> str | None:
    """Call Claude Haiku. Returns text or None on any failure."""
    try:
        resp = _claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as exc:
        print(f"[coach_v2] Claude call failed ({exc}), falling back to V1 output.", flush=True)
        return None


def _v1_fallback(report: "AnalysisReport") -> str:
    loc      = _snowboard_swap if report.sport == "snowboard" else (lambda s: s)
    rankable = [m for m in report.metrics if m.severity != "n_a"]
    if not rankable:
        return "We couldn't reliably read this clip — try a follow-cam with the rider centered for the full run."
    weakest = min(rankable, key=lambda m: m.score)
    return loc(weakest.headline) + " " + loc(weakest.detail)


# ---------------------------------------------------------------------------
# generate_coach_output  (replaces V1 version)
# ---------------------------------------------------------------------------

def generate_coach_output(report: "AnalysisReport", prior_session: dict | None = None) -> dict:
    """Build the V2 coach payload.

    Args:
        report:        AnalysisReport from the V1 pipeline (unchanged).
        prior_session: Optional dict — {'date': 'Nov 12', 'separation_angle': 38.0}
                       Fetch from Supabase before calling:
                           SELECT separation_angle, created_at
                           FROM analyses
                           WHERE user_id = $1
                           ORDER BY created_at DESC LIMIT 1

    Returns:
        Coach payload dict. Backward-compatible with V1 consumers.
        New top-level keys: 'coaching' (Claude output), 'separation_angle' (store in Supabase).
    """
    loc = _snowboard_swap if report.sport == "snowboard" else (lambda s: s)

    # --- Rejected clip: same as V1 ---
    if not report.rejection.passed:
        return {
            "status":  "rejected",
            "sport":   report.sport,
            "reasons": report.rejection.reasons,
            "metrics": [],
            "_internal": {
                "diagnostics": {
                    "detection_rate": round(report.rejection.detection_rate, 3),
                    "motion_score":   round(report.rejection.motion_score,   3),
                    "duration_sec":   round(report.rejection.duration_sec,   2),
                },
            },
        }

    # --- Claude coaching call ---
    prompt        = _build_coach_prompt(report, prior_session)
    claude_output = _call_claude(prompt)
    coaching_text = claude_output if claude_output else _v1_fallback(report)

    # --- Metric blocks (same structure as V1) ---
    metrics_user     = []
    metrics_internal = []
    for m in report.metrics:
        metrics_user.append({
            "name":           m.name,
            "level":          SEVERITY_LABEL[m.severity],
            "headline":       loc(m.headline),
            "detail":         loc(m.detail),
            "frame_evidence": m.frame_evidence,
        })
        metrics_internal.append({
            "name":      m.name,
            "severity":  m.severity,
            "score":     round(m.score, 1),
            "raw_value": (None if (isinstance(m.raw_value, float) and math.isnan(m.raw_value))
                          else round(m.raw_value, 3)),
        })

    rankable     = [m for m in report.metrics if m.severity != "n_a"]
    focus_metric = min(rankable, key=lambda m: m.score).name if rankable else None

    sep       = next((m for m in report.metrics if m.name == "separation"), None)
    sep_angle = (round(sep.raw_value, 1)
                 if sep and not math.isnan(sep.raw_value) else None)

    return {
        "status":           "ok",
        "sport":            report.sport,
        "coaching":         coaching_text,   # what the user reads — Claude output
        "separation_angle": sep_angle,       # store this in Supabase for next session
        "focus_metric":     focus_metric,
        "metrics":          metrics_user,
        "_internal": {
            "diagnostics": {
                "detection_rate": round(report.rejection.detection_rate, 3),
                "motion_score":   round(report.rejection.motion_score,   3),
                "duration_sec":   round(report.rejection.duration_sec,   2),
            },
            "metrics":          metrics_internal,
            "prompt_used":      prompt,
            "claude_succeeded": claude_output is not None,
            "prior_session":    prior_session,
        },
    }


# ---------------------------------------------------------------------------
# Annotated video
# ---------------------------------------------------------------------------

SKELETON_PAIRS = [
    ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
    ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
    ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
    ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"),
]

SEVERITY_COLOR_BGR = {
    "good": (90, 200, 90),
    "developing": (60, 180, 255),
    "needs_work": (60, 60, 240),
    "n_a": (160, 160, 160),
}


def render_annotated_video(video_path: str, df: pd.DataFrame, report: AnalysisReport,
                            output_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx < len(df):
            row = df.iloc[idx]
            # Skeleton (limbs)
            for a, b in SKELETON_PAIRS:
                ax, ay = row.get(f"{a}_x"), row.get(f"{a}_y")
                bx, by = row.get(f"{b}_x"), row.get(f"{b}_y")
                if not (np.isnan(ax) or np.isnan(bx)):
                    cv2.line(frame, (int(ax), int(ay)), (int(bx), int(by)),
                             (235, 235, 235), 2, cv2.LINE_AA)
            # Highlight the two metric-defining lines
            for a, b, color in [
                ("l_shoulder", "r_shoulder", (255, 180, 60)),  # shoulders — orange/blue
                ("l_hip", "r_hip", (255, 120, 200)),           # hips — pink/magenta
            ]:
                ax, ay = row.get(f"{a}_x"), row.get(f"{a}_y")
                bx, by = row.get(f"{b}_x"), row.get(f"{b}_y")
                if not (np.isnan(ax) or np.isnan(bx)):
                    cv2.line(frame, (int(ax), int(ay)), (int(bx), int(by)),
                             color, 4, cv2.LINE_AA)

        # Metrics panel (bottom-left) — TIER LABELS ONLY, no numeric scores
        if report.rejection.passed and report.metrics:
            n = len(report.metrics)
            panel_h = 26 * (n + 1) + 16
            cv2.rectangle(frame, (12, h - panel_h - 12), (470, h - 12), (0, 0, 0), -1)
            cv2.putText(frame, "SnowLens V1", (22, h - panel_h + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
            for i, m in enumerate(report.metrics):
                y = h - panel_h + 6 + 24 * (i + 1)
                pretty_name = m.name.replace("_", " ").title()
                label = f"{pretty_name}: {SEVERITY_LABEL[m.severity]}"
                cv2.putText(frame, label, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                            SEVERITY_COLOR_BGR[m.severity], 1, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (12, h - 60), (540, h - 12), (0, 0, 80), -1)
            cv2.putText(frame, "Clip rejected by quality gate",
                        (22, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (240, 240, 240), 1, cv2.LINE_AA)

        out.write(frame)
        idx += 1

    cap.release()
    out.release()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(video_path: str, output_dir: str, sport: str, model_name: str,
        prior_session: dict | None = None) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(video_path).stem

    print(f"[1/4] Extracting keypoints with {model_name} …", flush=True)
    df = extract_keypoints(video_path, model_name=model_name)
    kp_csv = out_dir / f"{base}_keypoints.csv"
    df.to_csv(kp_csv, index=False)
    print(f"      → {kp_csv}  ({len(df)} frames, {df.attrs['fps']:.1f} fps)", flush=True)

    print("[2/4] Rejection gate …", flush=True)
    rej = apply_rejection_gate(df)
    print(f"      passed={rej.passed}  detection={rej.detection_rate:.2f}  "
          f"motion={rej.motion_score:.3f}  duration={rej.duration_sec:.1f}s", flush=True)
    if not rej.passed:
        for r in rej.reasons:
            print(f"      reject: {r}", flush=True)

    if rej.passed:
        print("[3/4] Computing metrics …", flush=True)
        metrics = [
            compute_separation(df),
            compute_early_inside_lean(df),
            compute_shoulder_stability(df),
        ]
        for m in metrics:
            raw = "nan" if (isinstance(m.raw_value, float) and np.isnan(m.raw_value)) else f"{m.raw_value:.3f}"
            print(f"      {m.name:<22} score={m.score:5.1f}  sev={m.severity:<11}  raw={raw}",
                  flush=True)
    else:
        metrics = []
        print("[3/4] Skipped (rejected).", flush=True)

    report = AnalysisReport(video=video_path, sport=sport, rejection=rej, metrics=metrics)
    coach = generate_coach_output(report, prior_session=prior_session)
    coach_path = out_dir / f"{base}_coach.json"
    with open(coach_path, "w") as f:
        json.dump(coach, f, indent=2)
    print(f"      → {coach_path}", flush=True)

    print("[4/4] Rendering annotated video …", flush=True)
    ann_path = out_dir / f"{base}_annotated.mp4"
    render_annotated_video(video_path, df, report, str(ann_path))
    print(f"      → {ann_path}", flush=True)

    return {
        "keypoints_csv": str(kp_csv),
        "coach_json": str(coach_path),
        "annotated_mp4": str(ann_path),
        "coach": coach,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="SnowLens V2 — follow-cam body position coaching from a single clip."
    )
    ap.add_argument("video", help="Path to input video (.mp4)")
    ap.add_argument("--output-dir", default="/mnt/user-data/outputs",
                    help="Where to write artifacts")
    ap.add_argument("--sport", choices=["ski", "snowboard"], default="ski")
    ap.add_argument("--model", default="yolo11l-pose.pt",
                    help="Ultralytics YOLO11 pose model (default: yolo11l-pose.pt)")
    args = ap.parse_args(argv)

    result = run(args.video, args.output_dir, args.sport, args.model)
    print("\n=== COACH OUTPUT ===")
    print(json.dumps(result["coach"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
