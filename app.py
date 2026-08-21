"""
ChurnScope — Customer Retention Risk Engine
==========================================
A single-file Flask app that serves a pre-trained Keras ANN
(customer-churn binary classifier) through a modern risk-dashboard UI.

FOLDER LAYOUT (only 2 files you create/manage):
    app.py
    requirements.txt

Plus, in the SAME folder as app.py, place:
    model_ann.pkl                (required — your trained Keras model)
    scaler_ann.pkl                (OPTIONAL — your training StandardScaler)

--------------------------------------------------------------------------
IMPORTANT — ABOUT SCALING
--------------------------------------------------------------------------
This ANN was almost certainly trained on STANDARDIZED inputs (mean 0,
std 1), which is standard practice for this exact churn-prediction
architecture. No scaler.pkl was provided, so this app ships with a
*fallback* scaler built from published reference statistics for the
well-known 10,000-row bank-churn dataset this architecture is commonly
trained on. This will give reasonable, but NOT guaranteed exact,
results.

For production-accurate predictions:
  1. In your original training notebook, save the scaler you fit:
         import pickle
         with open("scaler_ann.pkl", "wb") as f:
             pickle.dump(sc, f)                 # sc = your fitted StandardScaler
  2. Drop scaler_ann.pkl into this same folder.
  3. Restart the app — it will auto-detect and use it instead of the
     fallback, and the "approximate scaling" banner will disappear.

--------------------------------------------------------------------------
IMPORTANT — ABOUT FEATURE ORDER / ENCODING
--------------------------------------------------------------------------
The model expects exactly 10 numeric inputs, in this order:

    [CreditScore, Geography, Gender, Age, Tenure, Balance,
     NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]

Categorical encoding assumed (standard LabelEncoder, alphabetical):
    Geography : France = 0, Germany = 1, Spain = 2
    Gender    : Female = 0, Male = 1

If your training notebook encoded these differently (e.g. one-hot, or
a different alphabetical order), edit GEOGRAPHY_MAP / GENDER_MAP below
and/or FEATURE_ORDER to match exactly — a mismatch here will silently
produce wrong predictions.

Run locally:
    pip install -r requirements.txt
    python app.py
    -> open http://localhost:5000

Deploy on Render:
    New Web Service -> connect repo
    Build Command : pip install -r requirements.txt
    Start Command : gunicorn app:app
    NOTE: TensorFlow is a large dependency. Render's free tier (512MB
    RAM) can be tight — first build/boot may be slow, and you may need
    a paid instance for reliable performance.
"""

import os
import io
import csv
import pickle
import logging
from datetime import datetime

import numpy as np
from flask import Flask, render_template_string, request, jsonify

# --------------------------------------------------------------------------
# App setup
# --------------------------------------------------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("churnscope")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_ann.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler_ann.pkl")

model = None
scaler = None
MODEL_READY = False
USING_FALLBACK_SCALER = True

# --------------------------------------------------------------------------
# Feature configuration — see docstring above before changing this
# --------------------------------------------------------------------------
FEATURE_ORDER = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary",
]
GEOGRAPHY_MAP = {"France": 0, "Germany": 1, "Spain": 2}
GENDER_MAP = {"Female": 0, "Male": 1}

# Fallback reference stats (mean, std) per feature, in FEATURE_ORDER —
# approximate values from the standard 10k-row bank-churn dataset this
# architecture is commonly trained on. Used ONLY if scaler_ann.pkl is
# not found alongside this file.
FALLBACK_MEAN = np.array([650.5, 0.75, 0.545, 38.9, 5.01, 76485.9, 1.53, 0.7055, 0.5151, 100090.2])
FALLBACK_STD = np.array([96.65, 0.83, 0.498, 10.49, 2.89, 62397.4, 0.5817, 0.4558, 0.4998, 57510.5])


class FallbackScaler:
    """Minimal drop-in replacement mimicking sklearn's StandardScaler.transform()."""

    def transform(self, X):
        X = np.asarray(X, dtype="float64")
        return (X - FALLBACK_MEAN) / FALLBACK_STD


try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    MODEL_READY = True
    log.info("ANN model loaded successfully.")
except Exception as exc:  # noqa: BLE001
    log.error("Failed to load model: %s", exc)

if os.path.exists(SCALER_PATH):
    try:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        USING_FALLBACK_SCALER = False
        log.info("Custom scaler_ann.pkl loaded — using exact training scaling.")
    except Exception as exc:  # noqa: BLE001
        log.error("Found scaler_ann.pkl but failed to load it: %s", exc)
        scaler = FallbackScaler()
else:
    scaler = FallbackScaler()
    log.warning("No scaler_ann.pkl found — using approximate fallback scaling.")


def build_row(payload: dict):
    return [
        float(payload["CreditScore"]),
        GEOGRAPHY_MAP[payload["Geography"]],
        GENDER_MAP[payload["Gender"]],
        float(payload["Age"]),
        float(payload["Tenure"]),
        float(payload["Balance"]),
        float(payload["NumOfProducts"]),
        1.0 if payload["HasCrCard"] else 0.0,
        1.0 if payload["IsActiveMember"] else 0.0,
        float(payload["EstimatedSalary"]),
    ]


def predict_churn(payload: dict):
    X = np.array([build_row(payload)], dtype="float64")
    X_scaled = scaler.transform(X)
    prob = float(model.predict(X_scaled, verbose=0)[0][0])
    return prob


def get_feature_std():
    """Return a {feature: std} lookup, preferring the real fitted scaler
    (scale_ attribute) over the fallback reference stats."""
    if scaler is not None and hasattr(scaler, "scale_"):
        return dict(zip(FEATURE_ORDER, scaler.scale_))
    return dict(zip(FEATURE_ORDER, FALLBACK_STD))


# Clamp bounds for perturbed numeric features during sensitivity analysis
NUMERIC_BOUNDS = {
    "CreditScore": (300, 900),
    "Age": (18, 100),
    "Tenure": (0, 15),
    "Balance": (0, None),
    "NumOfProducts": (1, 4),
    "EstimatedSalary": (0, None),
}

FEATURE_DISPLAY_NAMES = {
    "CreditScore": "Credit Score",
    "Age": "Age",
    "Tenure": "Tenure",
    "Balance": "Account Balance",
    "NumOfProducts": "Number of Products",
    "EstimatedSalary": "Estimated Salary",
    "Geography": "Geography",
    "Gender": "Gender",
    "HasCrCard": "Has Credit Card",
    "IsActiveMember": "Active Member Status",
}


def _clip(value, bounds):
    lo, hi = bounds
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


def compute_sensitivity(payload: dict):
    """Local sensitivity analysis: perturb each input against the REAL
    loaded model (finite differences, batched into a single forward pass)
    to see which features move this specific customer's churn probability
    the most. This is genuine model introspection, not a heuristic."""
    std = get_feature_std()
    numeric_fields = list(NUMERIC_BOUNDS.keys())

    rows = [build_row(payload)]          # index 0 = baseline
    plan = []                             # (feature, kind, extra)

    for f in numeric_fields:
        base_val = float(payload[f])
        step = std.get(f, 1.0) or 1.0
        plus_val = _clip(base_val + step, NUMERIC_BOUNDS[f])
        minus_val = _clip(base_val - step, NUMERIC_BOUNDS[f])

        p_plus = dict(payload); p_plus[f] = plus_val
        p_minus = dict(payload); p_minus[f] = minus_val
        rows.append(build_row(p_plus)); plan.append((f, "plus", None))
        rows.append(build_row(p_minus)); plan.append((f, "minus", None))

    geo_alts = [g for g in GEOGRAPHY_MAP if g != payload["Geography"]]
    for g in geo_alts:
        p = dict(payload); p["Geography"] = g
        rows.append(build_row(p)); plan.append(("Geography", "alt", g))

    gender_alt = next(g for g in GENDER_MAP if g != payload["Gender"])
    p = dict(payload); p["Gender"] = gender_alt
    rows.append(build_row(p)); plan.append(("Gender", "flip", gender_alt))

    p = dict(payload); p["HasCrCard"] = not payload["HasCrCard"]
    rows.append(build_row(p)); plan.append(("HasCrCard", "flip", None))

    p = dict(payload); p["IsActiveMember"] = not payload["IsActiveMember"]
    rows.append(build_row(p)); plan.append(("IsActiveMember", "flip", None))

    X = np.array(rows, dtype="float64")
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled, verbose=0).flatten()

    baseline = float(preds[0])
    impacts = {}
    idx = 1
    for f in numeric_fields:
        prob_plus = float(preds[idx]); idx += 1
        prob_minus = float(preds[idx]); idx += 1
        impacts[f] = (prob_plus - prob_minus) / 2.0

    for _ in geo_alts:
        prob_g = float(preds[idx]); idx += 1
        diff = prob_g - baseline
        if "Geography" not in impacts or abs(diff) > abs(impacts["Geography"]):
            impacts["Geography"] = diff

    prob_gender = float(preds[idx]); idx += 1
    impacts["Gender"] = prob_gender - baseline

    prob_cc = float(preds[idx]); idx += 1
    impacts["HasCrCard"] = prob_cc - baseline

    prob_active = float(preds[idx]); idx += 1
    impacts["IsActiveMember"] = prob_active - baseline

    result = [
        {"feature": FEATURE_DISPLAY_NAMES.get(k, k), "impact": round(v, 4)}
        for k, v in impacts.items()
    ]
    result.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return result[:8]


def score_batch(rows_raw):
    """rows_raw: list of dicts (as parsed from CSV). Returns (results, errors)."""
    row_vectors = []
    kept_raw = []
    errors = []

    for i, r in enumerate(rows_raw):
        line_no = i + 2  # header is line 1
        try:
            geography = (r.get("Geography") or "").strip()
            gender = (r.get("Gender") or "").strip()
            if geography not in GEOGRAPHY_MAP:
                errors.append(f"Row {line_no}: invalid Geography '{geography}'")
                continue
            if gender not in GENDER_MAP:
                errors.append(f"Row {line_no}: invalid Gender '{gender}'")
                continue

            has_cc = str(r.get("HasCrCard", "")).strip().lower() in ("1", "true", "yes", "y")
            is_active = str(r.get("IsActiveMember", "")).strip().lower() in ("1", "true", "yes", "y")

            row_vectors.append([
                float(r["CreditScore"]), GEOGRAPHY_MAP[geography], GENDER_MAP[gender],
                float(r["Age"]), float(r["Tenure"]), float(r["Balance"]),
                float(r["NumOfProducts"]), 1.0 if has_cc else 0.0,
                1.0 if is_active else 0.0, float(r["EstimatedSalary"]),
            ])
            kept_raw.append(r)
        except (KeyError, ValueError, TypeError) as exc:
            errors.append(f"Row {line_no}: {exc}")

    if not row_vectors:
        return [], errors

    X = np.array(row_vectors, dtype="float64")
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled, verbose=0).flatten()

    results = []
    for raw, prob in zip(kept_raw, preds):
        prob = float(max(0.0, min(1.0, prob)))
        risk = "High" if prob >= 0.7 else ("Medium" if prob >= 0.4 else "Low")
        row_out = dict(raw)
        row_out["churn_probability"] = round(prob, 4)
        row_out["risk_level"] = risk
        results.append(row_out)

    return results, errors


# --------------------------------------------------------------------------
# UI — embedded template (single file, no /templates folder needed)
# --------------------------------------------------------------------------
INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ChurnScope — Customer Retention Risk Engine</title>
<meta name="description" content="Predict customer churn risk with a trained neural network.">

<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">

<style>
  :root{
    --bg:            #0a0d12;
    --panel:         #12161d;
    --panel-raised:  #171c24;
    --stroke:        #262e3a;
    --text:          #e9edf3;
    --text-muted:    #8792a1;
    --text-faint:    #545f6e;

    --risk-safe:     #22d3a6;   /* teal — retained */
    --risk-safe-dim: #22d3a622;
    --risk-danger:   #ff5d5d;   /* red — churn */
    --risk-danger-dim:#ff5d5d22;
    --risk-amber:    #ffb020;   /* amber — medium risk / focus */
    --risk-amber-dim:#ffb02022;

    --font-display: 'Space Grotesk', sans-serif;
    --font-body:    'Inter', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;

    --radius: 14px;
    --radius-sm: 8px;
  }

  *{ box-sizing:border-box; margin:0; padding:0; }
  html{ scroll-behavior:smooth; }

  body{
    background:
      radial-gradient(circle at 12% 0%, #121a24 0%, transparent 45%),
      radial-gradient(circle at 88% 15%, #1a1218 0%, transparent 40%),
      var(--bg);
    color:var(--text);
    font-family:var(--font-body);
    min-height:100vh;
    line-height:1.5;
  }
  body::before{
    content:""; position:fixed; inset:0;
    background-image:repeating-linear-gradient(
      to bottom, rgba(255,255,255,0.012) 0px, rgba(255,255,255,0.012) 1px,
      transparent 1px, transparent 3px
    );
    pointer-events:none; z-index:0;
  }
  a{ color:inherit; }
  :focus-visible{ outline:2px solid var(--risk-amber); outline-offset:3px; border-radius:4px; }

  .wrap{ max-width:1180px; margin:0 auto; padding:0 28px; position:relative; z-index:1; }

  header{
    padding:26px 0 10px; display:flex; align-items:center; justify-content:space-between;
    border-bottom:1px solid var(--stroke);
  }
  .brand{ display:flex; align-items:center; gap:12px; }
  .brand-mark{
    width:38px; height:38px; border-radius:10px;
    background:linear-gradient(145deg, var(--risk-amber), #b5760d);
    display:flex; align-items:center; justify-content:center;
    box-shadow:0 0 22px var(--risk-amber-dim); flex-shrink:0;
  }
  .brand-mark svg{ width:20px; height:20px; }
  .brand-name{ font-family:var(--font-display); font-size:20px; font-weight:700; letter-spacing:0.3px; }
  .brand-tag{
    font-family:var(--font-mono); font-size:10.5px; letter-spacing:1.5px;
    color:var(--text-muted); text-transform:uppercase; margin-top:1px;
  }
  .status-chip{
    display:flex; align-items:center; gap:8px;
    font-family:var(--font-mono); font-size:11.5px; letter-spacing:0.5px;
    color:var(--text-muted); border:1px solid var(--stroke);
    padding:7px 13px; border-radius:100px; text-transform:uppercase;
  }
  .status-dot{
    width:7px; height:7px; border-radius:50%;
    background:var(--risk-safe); box-shadow:0 0 8px var(--risk-safe);
    animation:pulse 2.4s ease-in-out infinite;
  }
  .status-dot.down{ background:var(--risk-danger); box-shadow:0 0 8px var(--risk-danger); animation:none; }
  @keyframes pulse{ 0%,100%{opacity:1;} 50%{opacity:0.35;} }

  .scaler-banner{
    display:{{ 'flex' if using_fallback_scaler else 'none' }};
    align-items:center; gap:10px;
    background:var(--risk-amber-dim); border:1px solid #ffb02055;
    color:var(--risk-amber); font-family:var(--font-mono); font-size:12px;
    padding:11px 16px; border-radius:var(--radius-sm); margin-top:20px;
  }

  .hero{
    display:grid; grid-template-columns: 1.05fr 0.95fr; gap:56px;
    padding:44px 0 44px; align-items:center;
  }
  .eyebrow{
    font-family:var(--font-mono); font-size:11.5px; letter-spacing:2px;
    color:var(--risk-amber); text-transform:uppercase; margin-bottom:18px;
    display:flex; align-items:center; gap:10px;
  }
  .eyebrow::before{ content:""; width:22px; height:1px; background:var(--risk-amber); display:inline-block; }
  .hero h1{
    font-family:var(--font-display); font-weight:700;
    font-size:clamp(32px, 4.2vw, 50px); line-height:1.08; letter-spacing:-0.5px;
    margin-bottom:20px;
  }
  .hero h1 span{ color:var(--risk-amber); }
  .hero p{ color:var(--text-muted); font-size:16px; max-width:480px; margin-bottom:32px; }

  .spec-row{ display:flex; gap:30px; flex-wrap:wrap; }
  .spec{ font-family:var(--font-mono); }
  .spec .val{ font-size:22px; font-weight:700; color:var(--text); }
  .spec .lbl{ font-size:10.5px; color:var(--text-faint); text-transform:uppercase; letter-spacing:1px; margin-top:2px; }

  .hero-visual{
    background:var(--panel); border:1px solid var(--stroke); border-radius:var(--radius);
    padding:26px; display:flex; flex-direction:column; align-items:center; justify-content:center;
  }
  .hero-visual-head{
    width:100%; display:flex; justify-content:space-between; align-items:center;
    font-family:var(--font-mono); font-size:11px; color:var(--text-faint);
    text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;
  }
  .idle-net{ width:100%; height:150px; }
  .net-shape{
    font-family:var(--font-mono); font-size:11px; color:var(--text-faint);
    text-align:center; margin-top:8px; letter-spacing:0.5px;
  }

  .panel-title{
    font-family:var(--font-mono); font-size:11px; letter-spacing:1.5px;
    text-transform:uppercase; color:var(--text-faint); margin-bottom:14px;
    display:flex; align-items:center; gap:8px;
  }
  .panel-title .n{ color:var(--risk-amber); }

  .form-panel{
    background:var(--panel); border:1px solid var(--stroke); border-radius:var(--radius);
    padding:30px; margin-bottom:26px;
  }
  .form-panel h2{ font-family:var(--font-display); font-size:20px; font-weight:600; margin-bottom:22px; }

  .field-grid{ display:grid; grid-template-columns:repeat(2, 1fr); gap:18px 22px; }
  @media (max-width:700px){ .field-grid{ grid-template-columns:1fr; } }

  .field label{
    display:block; font-family:var(--font-mono); font-size:11px; letter-spacing:0.5px;
    color:var(--text-faint); text-transform:uppercase; margin-bottom:7px;
  }
  .field input[type=number], .field select{
    width:100%; background:var(--bg); border:1px solid var(--stroke); border-radius:var(--radius-sm);
    color:var(--text); font-family:var(--font-body); font-size:14.5px;
    padding:11px 13px; transition:border-color .15s ease;
  }
  .field input:focus, .field select:focus{ outline:none; border-color:var(--risk-amber); }

  .toggle-field{
    display:flex; align-items:center; justify-content:space-between;
    background:var(--bg); border:1px solid var(--stroke); border-radius:var(--radius-sm);
    padding:11px 14px;
  }
  .toggle-field span{ font-family:var(--font-body); font-size:14px; color:var(--text-muted); }
  .switch{ position:relative; width:42px; height:24px; flex-shrink:0; }
  .switch input{ opacity:0; width:0; height:0; }
  .slider{
    position:absolute; cursor:pointer; inset:0; background:var(--stroke);
    border-radius:100px; transition:.2s;
  }
  .slider::before{
    content:""; position:absolute; height:18px; width:18px; left:3px; bottom:3px;
    background:var(--text-faint); border-radius:50%; transition:.2s;
  }
  .switch input:checked + .slider{ background:var(--risk-safe-dim); border:1px solid var(--risk-safe); }
  .switch input:checked + .slider::before{ transform:translateX(18px); background:var(--risk-safe); }

  .form-actions{ display:flex; align-items:center; gap:16px; margin-top:26px; }
  .btn-predict{
    font-family:var(--font-display); font-weight:600; font-size:15px;
    background:var(--risk-amber); color:#241a02; border:none;
    padding:14px 28px; border-radius:var(--radius-sm); cursor:pointer;
    transition:transform .12s ease, box-shadow .12s ease;
    display:flex; align-items:center; gap:10px;
  }
  .btn-predict:hover{ transform:translateY(-1px); box-shadow:0 8px 24px var(--risk-amber-dim); }
  .btn-predict:disabled{ opacity:0.5; cursor:not-allowed; transform:none; box-shadow:none; }
  .btn-reset{
    font-family:var(--font-mono); font-size:12.5px; color:var(--text-faint);
    background:none; border:none; cursor:pointer; text-decoration:underline; text-underline-offset:3px;
  }
  .btn-reset:hover{ color:var(--text-muted); }

  #errorBox{
    display:none; margin-top:16px; padding:12px 16px;
    background:var(--risk-danger-dim); border:1px solid #ff5d5d55;
    border-radius:var(--radius-sm); color:var(--risk-danger);
    font-family:var(--font-mono); font-size:13px;
  }

  #readout{
    display:none; background:var(--panel); border:1px solid var(--stroke); border-radius:var(--radius);
    padding:30px; margin-bottom:26px; animation:rise .35s ease;
  }
  @keyframes rise{ from{ opacity:0; transform:translateY(10px);} to{opacity:1; transform:translateY(0);} }

  .readout-grid{ display:grid; grid-template-columns: 0.9fr 1.1fr; gap:36px; align-items:center; }
  @media (max-width:760px){ .readout-grid{ grid-template-columns:1fr; } }

  .gauge-wrap{ display:flex; flex-direction:column; align-items:center; }
  #gaugeSvg{ width:100%; max-width:280px; }
  .gauge-value{ font-family:var(--font-display); font-size:38px; font-weight:700; margin-top:-38px; }
  .gauge-label{ font-family:var(--font-mono); font-size:11px; color:var(--text-faint); text-transform:uppercase; letter-spacing:1px; margin-top:4px; }

  .verdict-badge{
    display:inline-flex; font-family:var(--font-display); font-weight:700; font-size:20px;
    padding:10px 20px; border-radius:var(--radius-sm); letter-spacing:0.3px; margin-bottom:18px;
  }
  .verdict-badge.safe{ background:var(--risk-safe-dim); color:var(--risk-safe); border:1px solid #22d3a655; }
  .verdict-badge.danger{ background:var(--risk-danger-dim); color:var(--risk-danger); border:1px solid #ff5d5d55; }

  .risk-meta{ font-family:var(--font-mono); font-size:12px; color:var(--text-faint); margin-bottom:20px; }
  .risk-meta div{ margin-top:4px; }

  .meter-row{ margin-bottom:16px; }
  .meter-label{
    display:flex; justify-content:space-between; font-family:var(--font-mono); font-size:12px;
    color:var(--text-muted); margin-bottom:6px; text-transform:uppercase; letter-spacing:0.5px;
  }
  .meter-track{ height:10px; background:var(--panel-raised); border-radius:100px; overflow:hidden; border:1px solid var(--stroke); }
  .meter-fill{ height:100%; border-radius:100px; width:0%; transition:width .7s cubic-bezier(.2,.8,.2,1); }
  .meter-fill.danger{ background:linear-gradient(90deg, #b12e2e, var(--risk-danger)); }
  .meter-fill.safe{ background:linear-gradient(90deg, #12a37a, var(--risk-safe)); }

  .recap{ margin-top:20px; display:flex; flex-wrap:wrap; gap:8px; }
  .recap-tag{
    font-family:var(--font-mono); font-size:11px; background:var(--panel-raised);
    border:1px solid var(--stroke); padding:5px 11px; border-radius:6px; color:var(--text-muted);
  }

  .readout-actions{ display:flex; gap:14px; margin-top:22px; }
  .btn-ghost{
    font-family:var(--font-mono); font-size:12px; color:var(--text-muted);
    background:var(--panel-raised); border:1px solid var(--stroke);
    padding:9px 15px; border-radius:var(--radius-sm); cursor:pointer;
    display:flex; align-items:center; gap:7px; transition:border-color .15s ease, color .15s ease;
  }
  .btn-ghost:hover{ border-color:var(--risk-amber); color:var(--text); }

  /* ---------------- Feature impact (sensitivity) ---------------- */
  #impactPanel{ margin-top:28px; padding-top:24px; border-top:1px solid var(--stroke); }
  .impact-row{ display:grid; grid-template-columns: 150px 1fr 60px; align-items:center; gap:12px; margin-bottom:10px; }
  .impact-label{ font-family:var(--font-mono); font-size:11.5px; color:var(--text-muted); text-align:right; }
  .impact-track{ position:relative; height:16px; background:var(--panel-raised); border-radius:4px; overflow:hidden; }
  .impact-track::before{
    content:""; position:absolute; left:50%; top:0; bottom:0; width:1px; background:var(--stroke); z-index:1;
  }
  .impact-bar{
    position:absolute; top:0; bottom:0; width:0%;
    transition:width .6s cubic-bezier(.2,.8,.2,1), left .6s cubic-bezier(.2,.8,.2,1);
  }
  .impact-bar.up{ background:linear-gradient(90deg, #b12e2e, var(--risk-danger)); left:50%; border-radius:0 3px 3px 0; }
  .impact-bar.down{ background:linear-gradient(90deg, var(--risk-safe), #12a37a); right:50%; border-radius:3px 0 0 3px; }
  .impact-val{ font-family:var(--font-mono); font-size:11px; color:var(--text-faint); }
  .impact-note{ font-family:var(--font-mono); font-size:11px; color:var(--text-faint); margin-top:14px; line-height:1.6; }

  /* ---------------- Batch scoring ---------------- */
  .batch-panel{
    background:var(--panel); border:1px solid var(--stroke); border-radius:var(--radius);
    padding:30px; margin-bottom:26px;
  }
  .batch-panel h2{ font-family:var(--font-display); font-size:20px; font-weight:600; margin-bottom:6px; }
  .batch-panel .sub{ font-family:var(--font-body); font-size:13.5px; color:var(--text-muted); margin-bottom:20px; }

  .dropzone{
    border:1.5px dashed var(--stroke); border-radius:var(--radius-sm);
    padding:32px 20px; text-align:center; cursor:pointer; transition:border-color .15s ease, background .15s ease;
  }
  .dropzone:hover, .dropzone.drag{ border-color:var(--risk-amber); background:var(--risk-amber-dim); }
  .dropzone svg{ width:28px; height:28px; margin-bottom:10px; color:var(--text-faint); }
  .dropzone p{ font-family:var(--font-mono); font-size:12.5px; color:var(--text-muted); }
  .dropzone .fname{ color:var(--risk-amber); margin-top:6px; }
  #csvFileInput{ display:none; }

  .batch-actions{ display:flex; align-items:center; gap:16px; margin-top:20px; flex-wrap:wrap; }

  #batchResults{ margin-top:24px; display:none; }
  .batch-summary{
    display:flex; gap:22px; flex-wrap:wrap; margin-bottom:16px;
    font-family:var(--font-mono); font-size:12px; color:var(--text-muted);
  }
  .batch-summary b{ color:var(--text); }
  .batch-table-wrap{ overflow-x:auto; border:1px solid var(--stroke); border-radius:var(--radius-sm); max-height:360px; overflow-y:auto; }
  table.batch-table{ width:100%; border-collapse:collapse; font-family:var(--font-mono); font-size:12px; }
  table.batch-table th{
    text-align:left; padding:10px 12px; background:var(--panel-raised); color:var(--text-faint);
    text-transform:uppercase; letter-spacing:0.5px; font-size:10.5px; position:sticky; top:0;
  }
  table.batch-table td{ padding:9px 12px; border-top:1px solid var(--stroke); color:var(--text-muted); white-space:nowrap; }
  .risk-pill{
    display:inline-block; padding:2px 9px; border-radius:100px; font-size:10.5px; font-weight:600;
  }
  .risk-pill.High{ background:var(--risk-danger-dim); color:var(--risk-danger); }
  .risk-pill.Medium{ background:var(--risk-amber-dim); color:var(--risk-amber); }
  .risk-pill.Low{ background:var(--risk-safe-dim); color:var(--risk-safe); }
  .batch-errors{
    margin-top:14px; font-family:var(--font-mono); font-size:11px; color:var(--risk-amber);
    background:var(--risk-amber-dim); border:1px solid #ffb02055; border-radius:var(--radius-sm); padding:12px 14px;
  }

  /* ---------------- API reference ---------------- */
  .api-panel{
    background:var(--panel); border:1px solid var(--stroke); border-radius:var(--radius);
    padding:30px; margin-bottom:50px;
  }
  .api-panel h2{ font-family:var(--font-display); font-size:20px; font-weight:600; margin-bottom:18px; }
  .api-endpoint{ margin-bottom:18px; }
  .api-endpoint:last-child{ margin-bottom:0; }
  .api-method{
    display:inline-block; font-family:var(--font-mono); font-size:11px; font-weight:700;
    padding:3px 9px; border-radius:5px; margin-right:8px;
  }
  .api-method.get{ background:var(--risk-safe-dim); color:var(--risk-safe); }
  .api-method.post{ background:var(--risk-amber-dim); color:var(--risk-amber); }
  .api-path{ font-family:var(--font-mono); font-size:13px; color:var(--text); }
  .api-desc{ font-family:var(--font-body); font-size:13px; color:var(--text-muted); margin:6px 0 8px; }
  .api-code{
    background:var(--bg); border:1px solid var(--stroke); border-radius:var(--radius-sm);
    padding:12px 14px; font-family:var(--font-mono); font-size:11.5px; color:var(--text-muted);
    overflow-x:auto; white-space:pre;
  }

  #history{ margin-bottom:60px; }
  #historyList{ display:flex; flex-direction:column; gap:10px; }
  .empty-state{
    font-family:var(--font-mono); font-size:12.5px; color:var(--text-faint);
    border:1px dashed var(--stroke); border-radius:var(--radius-sm); padding:22px; text-align:center;
  }
  .hist-item{
    display:flex; align-items:center; gap:14px; background:var(--panel);
    border:1px solid var(--stroke); border-radius:var(--radius-sm); padding:13px 16px;
  }
  .hist-dot{ width:9px; height:9px; border-radius:50%; flex-shrink:0; }
  .hist-dot.safe{ background:var(--risk-safe); box-shadow:0 0 8px var(--risk-safe); }
  .hist-dot.danger{ background:var(--risk-danger); box-shadow:0 0 8px var(--risk-danger); }
  .hist-text{ flex:1; font-size:13.5px; color:var(--text-muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .hist-conf{ font-family:var(--font-mono); font-size:11.5px; color:var(--text-faint); flex-shrink:0; }
  .hist-time{ font-family:var(--font-mono); font-size:11px; color:var(--text-faint); flex-shrink:0; }

  .stack{
    display:flex; align-items:center; justify-content:center; gap:12px; flex-wrap:wrap;
    padding:26px 0; border-top:1px solid var(--stroke); border-bottom:1px solid var(--stroke); margin-bottom:40px;
  }
  .stack-item{
    font-family:var(--font-mono); font-size:11px; letter-spacing:0.5px; color:var(--text-faint);
    border:1px solid var(--stroke); padding:7px 14px; border-radius:100px; text-transform:uppercase;
  }

  footer{ padding:30px 0 50px; text-align:center; }
  footer p{ font-family:var(--font-mono); font-size:11.5px; color:var(--text-faint); }

  @media (max-width:900px){
    .hero{ grid-template-columns:1fr; }
    .hero p{ max-width:none; }
  }
  @media (prefers-reduced-motion: reduce){
    *{ animation-duration:0.001ms !important; transition-duration:0.001ms !important; }
  }
</style>
</head>
<body>

<div class="wrap">

  <header>
    <div class="brand">
      <div class="brand-mark">
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <circle cx="12" cy="12" r="8" stroke="#241a02" stroke-width="2"/>
          <path d="M12 8v4l3 2" stroke="#241a02" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <div>
        <div class="brand-name">ChurnScope</div>
        <div class="brand-tag">Customer Retention Risk Engine</div>
      </div>
    </div>
    <div class="status-chip">
      <span class="status-dot {{ 'down' if not model_ready else '' }}"></span>
      {{ 'model online' if model_ready else 'model unavailable' }}
    </div>
  </header>

  <div class="scaler-banner">
    ⚠ No custom scaler_ann.pkl found — using approximate reference-dataset scaling. Add your training scaler for exact results (see app.py header comment).
  </div>

  <section class="hero">
    <div>
      <div class="eyebrow">Artificial Neural Network · 10 → 8 → 7 → 1</div>
      <h1>Predict which customers<br>are about to <span>walk away.</span></h1>
      <p>ChurnScope runs customer profiles through a trained neural network and reports the probability they'll churn — so retention teams know exactly who to focus on.</p>
      <div class="spec-row">
        <div class="spec"><div class="val" id="specInputDim">—</div><div class="lbl">Input Features</div></div>
        <div class="spec"><div class="val" id="specLayers">—</div><div class="lbl">Dense Layers</div></div>
        <div class="spec"><div class="val" id="specParams">—</div><div class="lbl">Trainable Params</div></div>
        <div class="spec"><div class="val">Binary</div><div class="lbl">Churn / Retained</div></div>
      </div>
    </div>

    <div class="hero-visual">
      <div class="hero-visual-head">
        <span>network topology</span>
        <span>live from model</span>
      </div>
      <svg class="idle-net" viewBox="0 0 320 150" xmlns="http://www.w3.org/2000/svg" id="idleNet"></svg>
      <div class="net-shape" id="netShape">—</div>
    </div>
  </section>

  <div class="stack">
    <span class="stack-item">Flask</span>
    <span class="stack-item">TensorFlow / Keras</span>
    <span class="stack-item">Dense Neural Network</span>
    <span class="stack-item">Gunicorn</span>
  </div>

  <section class="form-panel">
    <h2>Customer profile</h2>
    <div class="field-grid">
      <div class="field">
        <label for="creditScore">Credit Score</label>
        <input type="number" id="creditScore" value="650" min="300" max="900">
      </div>
      <div class="field">
        <label for="geography">Geography</label>
        <select id="geography">
          <option value="France">France</option>
          <option value="Germany">Germany</option>
          <option value="Spain">Spain</option>
        </select>
      </div>
      <div class="field">
        <label for="gender">Gender</label>
        <select id="gender">
          <option value="Female">Female</option>
          <option value="Male">Male</option>
        </select>
      </div>
      <div class="field">
        <label for="age">Age</label>
        <input type="number" id="age" value="35" min="18" max="100">
      </div>
      <div class="field">
        <label for="tenure">Tenure (years with bank)</label>
        <input type="number" id="tenure" value="5" min="0" max="15">
      </div>
      <div class="field">
        <label for="balance">Account Balance</label>
        <input type="number" id="balance" value="75000" min="0" step="100">
      </div>
      <div class="field">
        <label for="numProducts">Number of Products</label>
        <input type="number" id="numProducts" value="2" min="1" max="4">
      </div>
      <div class="field">
        <label for="estimatedSalary">Estimated Salary</label>
        <input type="number" id="estimatedSalary" value="100000" min="0" step="100">
      </div>
      <div class="field">
        <label>Has Credit Card</label>
        <div class="toggle-field">
          <span>Customer owns a credit card</span>
          <label class="switch">
            <input type="checkbox" id="hasCrCard" checked>
            <span class="slider"></span>
          </label>
        </div>
      </div>
      <div class="field">
        <label>Active Member</label>
        <div class="toggle-field">
          <span>Customer is currently active</span>
          <label class="switch">
            <input type="checkbox" id="isActiveMember" checked>
            <span class="slider"></span>
          </label>
        </div>
      </div>
    </div>

    <div class="form-actions">
      <button class="btn-predict" id="predictBtn" {{ 'disabled' if not model_ready else '' }}>
        <span id="predictLabel">Run Prediction</span>
      </button>
      <button class="btn-reset" id="resetBtn">Reset to defaults</button>
    </div>

    <div id="errorBox"></div>
  </section>

  <section id="readout">
    <div class="panel-title"><span class="n">//</span> Risk Readout</div>

    <div class="readout-grid">
      <div class="gauge-wrap">
        <svg id="gaugeSvg" viewBox="0 0 200 120">
          <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#171c24" stroke-width="16" stroke-linecap="round"/>
          <path id="gaugeArc" d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#22d3a6" stroke-width="16" stroke-linecap="round"
                stroke-dasharray="251.2" stroke-dashoffset="251.2"/>
          <line id="gaugeNeedle" x1="100" y1="100" x2="100" y2="35" stroke="#e9edf3" stroke-width="3" stroke-linecap="round" transform="rotate(-90 100 100)"/>
          <circle cx="100" cy="100" r="6" fill="#e9edf3"/>
        </svg>
        <div class="gauge-value" id="gaugeValue">0%</div>
        <div class="gauge-label">Churn Probability</div>
      </div>

      <div>
        <div class="verdict-badge" id="verdictBadge">—</div>
        <div class="risk-meta">
          <div>Predicted at <span id="metaTime">—</span></div>
          <div id="metaRisk">Risk level: —</div>
        </div>

        <div class="meter-row">
          <div class="meter-label"><span>Churn Risk</span><span id="churnVal">0%</span></div>
          <div class="meter-track"><div class="meter-fill danger" id="churnFill"></div></div>
        </div>
        <div class="meter-row">
          <div class="meter-label"><span>Retention Likelihood</span><span id="retainVal">0%</span></div>
          <div class="meter-track"><div class="meter-fill safe" id="retainFill"></div></div>
        </div>

        <div class="recap" id="recapTags"></div>

        <div class="readout-actions">
          <button class="btn-ghost" id="downloadReportBtn">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3v12m0 0l-4-4m4 4l4-4M4 19h16"/></svg>
            Download Report (JSON)
          </button>
        </div>
      </div>
    </div>

    <div id="impactPanel">
      <div class="panel-title"><span class="n">//</span> Feature Impact — Local Sensitivity</div>
      <div id="impactBars"></div>
      <div class="impact-note">Each bar shows how much this customer's churn probability shifts when that input is nudged up/down (numeric) or switched (categorical), holding everything else fixed — computed directly from the loaded model, not a static rule.</div>
    </div>
  </section>

  <section class="batch-panel">
    <h2>Batch scoring</h2>
    <div class="sub">Upload a CSV with columns: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary — score up to 500 customers at once.</div>

    <div class="dropzone" id="dropzone">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M12 3v12m0 0l-4-4m4 4l4-4M4 19h16"/></svg>
      <p>Drop a CSV file here, or click to browse</p>
      <div class="fname" id="fileName"></div>
    </div>
    <input type="file" id="csvFileInput" accept=".csv">

    <div class="batch-actions">
      <button class="btn-predict" id="batchBtn" {{ 'disabled' if not model_ready else '' }}>
        <span id="batchLabel">Run Batch Scoring</span>
      </button>
      <button class="btn-ghost" id="downloadTemplateBtn">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3v12m0 0l-4-4m4 4l4-4M4 19h16"/></svg>
        Download CSV Template
      </button>
      <button class="btn-ghost" id="downloadResultsBtn" style="display:none;">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3v12m0 0l-4-4m4 4l4-4M4 19h16"/></svg>
        Download Results CSV
      </button>
    </div>

    <div id="batchError" style="display:none;"></div>

    <div id="batchResults">
      <div class="batch-summary">
        <span><b id="batchCount">0</b> scored</span>
        <span><b id="batchHigh">0</b> high risk</span>
        <span><b id="batchMed">0</b> medium risk</span>
        <span><b id="batchLow">0</b> low risk</span>
      </div>
      <div class="batch-table-wrap">
        <table class="batch-table" id="batchTable">
          <thead>
            <tr>
              <th>Geography</th><th>Gender</th><th>Age</th><th>Tenure</th>
              <th>Balance</th><th>Products</th><th>Churn Prob.</th><th>Risk</th>
            </tr>
          </thead>
          <tbody id="batchTableBody"></tbody>
        </table>
      </div>
      <div class="batch-errors" id="batchErrorsList" style="display:none;"></div>
    </div>
  </section>

  <section class="api-panel">
    <h2>API reference</h2>

    <div class="api-endpoint">
      <span class="api-method get">GET</span><span class="api-path">/model_info</span>
      <div class="api-desc">Returns the real architecture of the loaded model (layers, units, activations, param count).</div>
    </div>

    <div class="api-endpoint">
      <span class="api-method post">POST</span><span class="api-path">/predict</span>
      <div class="api-desc">Score a single customer. Body: JSON with the 10 feature fields. Returns churn_probability, risk_level, and sensitivity.</div>
      <div class="api-code">curl -X POST /predict -H "Content-Type: application/json" -d '{"CreditScore":650,"Geography":"France","Gender":"Female","Age":35,"Tenure":5,"Balance":75000,"NumOfProducts":2,"HasCrCard":true,"IsActiveMember":true,"EstimatedSalary":100000}'</div>
    </div>

    <div class="api-endpoint">
      <span class="api-method post">POST</span><span class="api-path">/predict_batch</span>
      <div class="api-desc">Score up to 500 customers from a CSV file. Multipart form field name: file.</div>
      <div class="api-code">curl -X POST /predict_batch -F "file=@customers.csv"</div>
    </div>
  </section>

  <section id="history">
    <div class="panel-title"><span class="n">//</span> Recent Predictions</div>
    <div id="historyList">
      <div class="empty-state">No predictions yet — run one above to see it here.</div>
    </div>
  </section>

  <footer>
    <p>CHURNSCOPE · KERAS ANN CUSTOMER CHURN ENGINE · BUILT WITH FLASK</p>
  </footer>

</div>

<script>
  // ---------- idle network topology visual ----------
  const svgNS = "http://www.w3.org/2000/svg";
  const idleNet = document.getElementById('idleNet');

  function drawNetwork(layerSizes){
    idleNet.innerHTML = '';
    // cap displayed nodes per layer for legibility; real count still shown in the label below
    const displaySizes = layerSizes.map(n => Math.max(1, Math.min(n, 8)));
    const layerX = displaySizes.map((_, i) => 30 + i * (260 / (displaySizes.length - 1 || 1)));
    const nodesByLayer = [];

    displaySizes.forEach((count, li) => {
      const nodes = [];
      const spacing = 150 / (count + 1);
      for (let i = 0; i < count; i++){
        nodes.push({ x: layerX[li], y: spacing * (i + 1) + 5 });
      }
      nodesByLayer.push(nodes);
    });

    for (let li = 0; li < nodesByLayer.length - 1; li++){
      nodesByLayer[li].forEach(a => {
        nodesByLayer[li + 1].forEach(b => {
          const line = document.createElementNS(svgNS, 'line');
          line.setAttribute('x1', a.x); line.setAttribute('y1', a.y);
          line.setAttribute('x2', b.x); line.setAttribute('y2', b.y);
          line.setAttribute('stroke', '#262e3a');
          line.setAttribute('stroke-width', '1');
          idleNet.appendChild(line);
        });
      });
    }
    nodesByLayer.forEach((nodes, li) => {
      nodes.forEach(n => {
        const circle = document.createElementNS(svgNS, 'circle');
        circle.setAttribute('cx', n.x); circle.setAttribute('cy', n.y);
        circle.setAttribute('r', 5);
        circle.setAttribute('fill', li === nodesByLayer.length - 1 ? '#ffb020' : '#171c24');
        circle.setAttribute('stroke', '#ffb020');
        circle.setAttribute('stroke-width', '1.5');
        idleNet.appendChild(circle);
      });
    });
  }

  // fallback shape while /model_info loads
  drawNetwork([4, 8, 7, 1]);

  fetch('/model_info').then(r => r.json()).then(info => {
    if (info.error) return;
    const layerSizes = [info.input_dim, ...info.layers.map(l => l.units)];
    drawNetwork(layerSizes);
    document.getElementById('netShape').textContent = layerSizes.join(' → ');

    document.getElementById('specInputDim').textContent = info.input_dim;
    document.getElementById('specLayers').textContent = info.layers.length;
    document.getElementById('specParams').textContent = info.total_params.toLocaleString();
  }).catch(() => {
    document.getElementById('netShape').textContent = '10 → 8 → 7 → 1';
  });

  // ---------- elements ----------
  const predictBtn   = document.getElementById('predictBtn');
  const predictLabel = document.getElementById('predictLabel');
  const resetBtn      = document.getElementById('resetBtn');
  const errorBox      = document.getElementById('errorBox');
  const readout        = document.getElementById('readout');
  const historyList     = document.getElementById('historyList');

  const DEFAULTS = {
    creditScore: 650, geography: 'France', gender: 'Female', age: 35,
    tenure: 5, balance: 75000, numProducts: 2, estimatedSalary: 100000,
    hasCrCard: true, isActiveMember: true
  };

  let history = [];

  resetBtn.addEventListener('click', () => {
    document.getElementById('creditScore').value = DEFAULTS.creditScore;
    document.getElementById('geography').value = DEFAULTS.geography;
    document.getElementById('gender').value = DEFAULTS.gender;
    document.getElementById('age').value = DEFAULTS.age;
    document.getElementById('tenure').value = DEFAULTS.tenure;
    document.getElementById('balance').value = DEFAULTS.balance;
    document.getElementById('numProducts').value = DEFAULTS.numProducts;
    document.getElementById('estimatedSalary').value = DEFAULTS.estimatedSalary;
    document.getElementById('hasCrCard').checked = DEFAULTS.hasCrCard;
    document.getElementById('isActiveMember').checked = DEFAULTS.isActiveMember;
    readout.style.display = 'none';
    errorBox.style.display = 'none';
  });

  function showError(msg){
    errorBox.textContent = msg;
    errorBox.style.display = 'block';
  }

  function setGauge(prob){
    // prob is 0..1. Gauge arc: full circumference of the half-circle path ~251.2
    const circumference = 251.2;
    const offset = circumference - (circumference * prob);
    const arc = document.getElementById('gaugeArc');
    arc.style.transition = 'stroke-dashoffset 0.8s cubic-bezier(.2,.8,.2,1), stroke 0.4s ease';
    arc.setAttribute('stroke-dashoffset', offset);
    arc.setAttribute('stroke', prob >= 0.5 ? '#ff5d5d' : '#22d3a6');

    const needle = document.getElementById('gaugeNeedle');
    const angle = -90 + (180 * prob);
    needle.style.transition = 'transform 0.8s cubic-bezier(.2,.8,.2,1)';
    needle.setAttribute('transform', `rotate(${angle} 100 100)`);

    document.getElementById('gaugeValue').textContent = Math.round(prob * 100) + '%';
  }

  function addHistory(summary, isChurn, prob, time){
    history.unshift({ summary, isChurn, prob, time });
    history = history.slice(0, 6);
    historyList.innerHTML = '';
    if (history.length === 0){
      historyList.innerHTML = '<div class="empty-state">No predictions yet — run one above to see it here.</div>';
      return;
    }
    history.forEach(item => {
      const row = document.createElement('div');
      row.className = 'hist-item';
      row.innerHTML = `
        <span class="hist-dot ${item.isChurn ? 'danger' : 'safe'}"></span>
        <span class="hist-text">${item.summary}</span>
        <span class="hist-conf">${(item.prob * 100).toFixed(1)}%</span>
        <span class="hist-time">${item.time}</span>
      `;
      historyList.appendChild(row);
    });
  }

  async function predict(){
    const payload = {
      CreditScore: Number(document.getElementById('creditScore').value),
      Geography: document.getElementById('geography').value,
      Gender: document.getElementById('gender').value,
      Age: Number(document.getElementById('age').value),
      Tenure: Number(document.getElementById('tenure').value),
      Balance: Number(document.getElementById('balance').value),
      NumOfProducts: Number(document.getElementById('numProducts').value),
      HasCrCard: document.getElementById('hasCrCard').checked,
      IsActiveMember: document.getElementById('isActiveMember').checked,
      EstimatedSalary: Number(document.getElementById('estimatedSalary').value),
    };

    errorBox.style.display = 'none';
    predictBtn.disabled = true;
    predictLabel.textContent = 'Running…';

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();

      if (!res.ok){
        showError(data.error || 'Something went wrong. Please try again.');
        return;
      }

      readout.style.display = 'block';
      readout.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

      const isChurn = data.churn_probability >= 0.5;
      const badge = document.getElementById('verdictBadge');
      badge.textContent = isChurn ? 'LIKELY TO CHURN' : 'LIKELY TO STAY';
      badge.className = 'verdict-badge ' + (isChurn ? 'danger' : 'safe');

      document.getElementById('metaTime').textContent = data.timestamp;
      document.getElementById('metaRisk').textContent = 'Risk level: ' + data.risk_level;

      const churnPct = data.churn_probability * 100;
      const retainPct = 100 - churnPct;
      document.getElementById('churnVal').textContent = churnPct.toFixed(1) + '%';
      document.getElementById('retainVal').textContent = retainPct.toFixed(1) + '%';
      document.getElementById('churnFill').style.width = churnPct + '%';
      document.getElementById('retainFill').style.width = retainPct + '%';

      setGauge(data.churn_probability);

      const recap = document.getElementById('recapTags');
      recap.innerHTML = '';
      const tags = [
        `${payload.Geography}`, `${payload.Gender}`, `Age ${payload.Age}`,
        `${payload.NumOfProducts} product(s)`, `${payload.Tenure}y tenure`,
        payload.IsActiveMember ? 'active member' : 'inactive member'
      ];
      tags.forEach(t => {
        const tag = document.createElement('span');
        tag.className = 'recap-tag';
        tag.textContent = t;
        recap.appendChild(tag);
      });

      const summary = `${payload.Gender}, ${payload.Age}y, ${payload.Geography}`;
      addHistory(summary, isChurn, data.churn_probability, data.timestamp);

      renderImpactBars(data.sensitivity || []);
      lastResult = { profile: payload, ...data };

    } catch (err) {
      showError('Could not reach the prediction server. Please try again.');
    } finally {
      predictBtn.disabled = false;
      predictLabel.textContent = 'Run Prediction';
    }
  }

  predictBtn.addEventListener('click', predict);

  // ---------- feature impact bars ----------
  let lastResult = null;

  function renderImpactBars(sensitivity){
    const box = document.getElementById('impactBars');
    box.innerHTML = '';
    if (!sensitivity.length){
      box.innerHTML = '<div class="empty-state">No sensitivity data returned.</div>';
      return;
    }
    const maxAbs = Math.max(...sensitivity.map(s => Math.abs(s.impact)), 0.001);
    sensitivity.forEach(s => {
      const pct = Math.min(100, (Math.abs(s.impact) / maxAbs) * 48); // half-width max
      const row = document.createElement('div');
      row.className = 'impact-row';
      const sign = s.impact >= 0 ? '+' : '';
      row.innerHTML = `
        <div class="impact-label">${s.feature}</div>
        <div class="impact-track">
          <div class="impact-bar ${s.impact >= 0 ? 'up' : 'down'}" style="width:${pct}%"></div>
        </div>
        <div class="impact-val">${sign}${(s.impact * 100).toFixed(1)}%</div>
      `;
      box.appendChild(row);
    });
  }

  document.getElementById('downloadReportBtn').addEventListener('click', () => {
    if (!lastResult) return;
    const blob = new Blob([JSON.stringify(lastResult, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `churn-report-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });

  // ---------- batch scoring ----------
  const dropzone = document.getElementById('dropzone');
  const csvFileInput = document.getElementById('csvFileInput');
  const fileNameEl = document.getElementById('fileName');
  const batchBtn = document.getElementById('batchBtn');
  const batchLabel = document.getElementById('batchLabel');
  const batchError = document.getElementById('batchError');
  const batchResults = document.getElementById('batchResults');
  const batchTableBody = document.getElementById('batchTableBody');
  const downloadResultsBtn = document.getElementById('downloadResultsBtn');

  let selectedFile = null;
  let lastBatchResults = null;

  dropzone.addEventListener('click', () => csvFileInput.click());
  dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('drag'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag');
    if (e.dataTransfer.files.length){
      selectedFile = e.dataTransfer.files[0];
      fileNameEl.textContent = selectedFile.name;
    }
  });
  csvFileInput.addEventListener('change', () => {
    if (csvFileInput.files.length){
      selectedFile = csvFileInput.files[0];
      fileNameEl.textContent = selectedFile.name;
    }
  });

  document.getElementById('downloadTemplateBtn').addEventListener('click', () => {
    const header = 'CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary';
    const example = '650,France,Female,35,5,75000,2,1,1,100000';
    const blob = new Blob([header + '\n' + example + '\n'], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'churnscope_template.csv'; a.click();
    URL.revokeObjectURL(url);
  });

  batchBtn.addEventListener('click', async () => {
    batchError.style.display = 'none';
    if (!selectedFile){
      batchError.textContent = 'Please choose a CSV file first.';
      batchError.style.display = 'block';
      return;
    }

    batchBtn.disabled = true;
    batchLabel.textContent = 'Scoring…';

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      const res = await fetch('/predict_batch', { method: 'POST', body: formData });
      const data = await res.json();

      if (!res.ok){
        batchError.textContent = data.error || 'Batch scoring failed.';
        batchError.style.display = 'block';
        return;
      }

      lastBatchResults = data.results;
      batchResults.style.display = 'block';
      downloadResultsBtn.style.display = 'inline-flex';

      const high = data.results.filter(r => r.risk_level === 'High').length;
      const med = data.results.filter(r => r.risk_level === 'Medium').length;
      const low = data.results.filter(r => r.risk_level === 'Low').length;
      document.getElementById('batchCount').textContent = data.count;
      document.getElementById('batchHigh').textContent = high;
      document.getElementById('batchMed').textContent = med;
      document.getElementById('batchLow').textContent = low;

      batchTableBody.innerHTML = '';
      data.results.forEach(r => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${r.Geography}</td><td>${r.Gender}</td><td>${r.Age}</td><td>${r.Tenure}</td>
          <td>${Number(r.Balance).toLocaleString()}</td><td>${r.NumOfProducts}</td>
          <td>${(r.churn_probability * 100).toFixed(1)}%</td>
          <td><span class="risk-pill ${r.risk_level}">${r.risk_level}</span></td>
        `;
        batchTableBody.appendChild(tr);
      });

      const errList = document.getElementById('batchErrorsList');
      if (data.skipped > 0){
        errList.style.display = 'block';
        errList.textContent = `${data.skipped} row(s) skipped: ` + (data.errors || []).join(' · ');
      } else {
        errList.style.display = 'none';
      }

    } catch (err) {
      batchError.textContent = 'Could not reach the batch scoring server.';
      batchError.style.display = 'block';
    } finally {
      batchBtn.disabled = false;
      batchLabel.textContent = 'Run Batch Scoring';
    }
  });

  downloadResultsBtn.addEventListener('click', () => {
    if (!lastBatchResults || !lastBatchResults.length) return;
    const cols = Object.keys(lastBatchResults[0]);
    const lines = [cols.join(',')];
    lastBatchResults.forEach(r => {
      lines.push(cols.map(c => r[c]).join(','));
    });
    const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `churnscope_results_${Date.now()}.csv`; a.click();
    URL.revokeObjectURL(url);
  });
</script>

</body>
</html>
"""


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template_string(
        INDEX_HTML,
        model_ready=MODEL_READY,
        using_fallback_scaler=USING_FALLBACK_SCALER,
    )


@app.route("/health")
def health():
    return jsonify(
        status="ok" if MODEL_READY else "degraded",
        model_ready=MODEL_READY,
        using_fallback_scaler=USING_FALLBACK_SCALER,
    )


@app.route("/model_info")
def model_info():
    """Introspects the REAL loaded Keras model so the UI never shows
    stale/hardcoded architecture numbers."""
    if not MODEL_READY:
        return jsonify(error="Model is not loaded on the server."), 503

    layers = []
    for layer in model.layers:
        cfg = layer.get_config()
        layers.append({
            "name": layer.name,
            "units": cfg.get("units"),
            "activation": cfg.get("activation"),
        })

    try:
        input_dim = model.input_shape[-1]
        output_dim = model.output_shape[-1]
    except Exception:  # noqa: BLE001
        input_dim = output_dim = None

    return jsonify(
        layers=layers,
        total_params=int(model.count_params()),
        input_dim=input_dim,
        output_dim=output_dim,
        using_fallback_scaler=USING_FALLBACK_SCALER,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_READY:
        return jsonify(error="Model is not loaded on the server."), 503

    payload = request.get_json(silent=True) or {}

    required = ["CreditScore", "Geography", "Gender", "Age", "Tenure",
                "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
                "EstimatedSalary"]
    missing = [k for k in required if k not in payload]
    if missing:
        return jsonify(error=f"Missing fields: {', '.join(missing)}"), 400

    if payload["Geography"] not in GEOGRAPHY_MAP:
        return jsonify(error="Geography must be one of: France, Germany, Spain."), 400
    if payload["Gender"] not in GENDER_MAP:
        return jsonify(error="Gender must be one of: Female, Male."), 400

    try:
        prob = predict_churn(payload)
        sensitivity = compute_sensitivity(payload)
    except (TypeError, ValueError) as exc:
        return jsonify(error=f"Invalid input values: {exc}"), 400
    except Exception as exc:  # noqa: BLE001
        log.error("Prediction failed: %s", exc)
        return jsonify(error="Prediction failed on the server."), 500

    prob = max(0.0, min(1.0, prob))
    if prob >= 0.7:
        risk_level = "High"
    elif prob >= 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return jsonify(
        churn_probability=round(prob, 4),
        risk_level=risk_level,
        sensitivity=sensitivity,
        using_fallback_scaler=USING_FALLBACK_SCALER,
        timestamp=datetime.utcnow().strftime("%H:%M:%S"),
    )


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    if not MODEL_READY:
        return jsonify(error="Model is not loaded on the server."), 503

    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify(error="Please attach a CSV file under the 'file' field."), 400

    file = request.files["file"]

    try:
        raw = file.read().decode("utf-8-sig")
    except UnicodeDecodeError:
        return jsonify(error="Could not read the file as UTF-8 text. Please upload a CSV."), 400

    reader = csv.DictReader(io.StringIO(raw))
    if not reader.fieldnames:
        return jsonify(error="CSV appears to be empty."), 400

    missing_cols = [c for c in FEATURE_ORDER if c not in reader.fieldnames]
    if missing_cols:
        return jsonify(error=f"CSV is missing required columns: {', '.join(missing_cols)}"), 400

    all_rows = list(reader)
    if len(all_rows) > 500:
        return jsonify(error="For this demo, batch scoring is capped at 500 rows."), 400
    if not all_rows:
        return jsonify(error="CSV has a header but no data rows."), 400

    try:
        results, errors = score_batch(all_rows)
    except Exception as exc:  # noqa: BLE001
        log.error("Batch scoring failed: %s", exc)
        return jsonify(error="Batch scoring failed on the server."), 500

    if not results:
        return jsonify(error="No valid rows could be scored.", details=errors[:10]), 400

    return jsonify(
        results=results,
        count=len(results),
        skipped=len(errors),
        errors=errors[:10],
        timestamp=datetime.utcnow().strftime("%H:%M:%S"),
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
