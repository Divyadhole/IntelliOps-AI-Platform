"""Render the executive dashboard as one self-contained HTML file.

Why a static build alongside the Streamlit app: a link that opens instantly with no
Python, no install and no server is the version a hiring manager will actually look
at. The Streamlit app is the interactive product; this is its shareable snapshot,
generated from the same warehouse tables so the two can never disagree.

Outputs
    artifacts/reports/dashboard.html       standalone page (open with a double-click)
    artifacts/reports/dashboard.body.html  same content, no <html>/<head> wrapper
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from html import escape

import pandas as pd

from ..config import Config, load_config
from ..data_pipeline import warehouse
from ..logging_utils import get_logger, stage
from .svg import columns, donut, hbars, legend, line_area, meter

logger = get_logger(__name__)

# viewBox width matched to a single-column card's rendered width (~340px at 1440)
NARROW = 340

RISK_ORDER = ["Low", "Medium", "High", "Critical"]
RISK_TOKENS = {
    "Low": "var(--status-good)",
    "Medium": "var(--status-warning)",
    "High": "var(--status-serious)",
    "Critical": "var(--status-critical)",
}
CATEGORICAL = ["var(--series-1)", "var(--series-2)", "var(--series-3)"]


# ------------------------------------------------------------------ data assembly
def collect(cfg: Config) -> dict:
    tables = cfg["warehouse.schema_tables"]
    features = warehouse.read_table(tables["features"], cfg)
    predictions = warehouse.read_table(tables["predictions"], cfg)
    quality = warehouse.read_table(tables["quality"], cfg)

    segments = warehouse.read_table("fct_customer_segments", cfg) \
        if warehouse.table_exists("fct_customer_segments", cfg) else pd.DataFrame()
    topics = warehouse.read_table("agg_topic_summary", cfg) \
        if warehouse.table_exists("agg_topic_summary", cfg) else pd.DataFrame()
    reviews = warehouse.read_table("fct_review_nlp", cfg) \
        if warehouse.table_exists("fct_review_nlp", cfg) else pd.DataFrame()

    report_path = cfg.path("paths.reports") / "model_report.json"
    model = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}

    merged = predictions.merge(
        features[["customerID", "tenure", "MonthlyCharges", "Contract", "annual_margin_at_risk"]],
        on="customerID", how="left",
    )
    econ = cfg["churn_model.economics"]

    # Probability-weighted exposure, not a worst case: Σ P(churn) × annual margin.
    margin_at_risk = float((merged["churn_probability"] * merged["annual_margin_at_risk"]).sum())
    targeted = merged[merged["targeted_by_policy"] == 1]

    hazard = (
        features.assign(bucket=pd.cut(features["tenure"], bins=range(0, 79, 6), right=True))
        .groupby("bucket", observed=True)["Churn"].mean()
    )
    hazard_points = [(f"{int(iv.right)}", float(v)) for iv, v in hazard.items() if pd.notna(v)]

    band_counts = predictions["risk_band"].value_counts()
    by_contract = (
        features.groupby("Contract")
        .agg(customers=("customerID", "count"), churn_rate=("Churn", "mean"),
             margin=("annual_margin_at_risk", "sum"))
        .sort_values("churn_rate", ascending=False)
    )

    seg_summary = pd.DataFrame()
    if not segments.empty:
        joined = features.merge(segments, on="customerID", how="left")
        seg_summary = (
            joined.groupby("segment_name")
            .agg(customers=("customerID", "count"), churn_rate=("Churn", "mean"),
                 avg_monthly=("MonthlyCharges", "mean"), avg_tenure=("tenure", "mean"))
            .sort_values("customers", ascending=False)
        )

    call_list = (
        merged[merged["targeted_by_policy"] == 1]
        .sort_values("expected_value_of_offer", ascending=False)
        .head(12)
    )

    verbatims = []
    if not reviews.empty and "sentiment_score" in reviews.columns:
        worst = reviews.sort_values("sentiment_score").head(60)
        seen: set[str] = set()
        for _, row in worst.iterrows():
            text = str(row["review_text"])
            key = text[:40]
            if key in seen:
                continue
            seen.add(key)
            verbatims.append({"text": text, "rating": int(row.get("rating", 0)),
                              "channel": str(row.get("channel", "feedback"))})
            if len(verbatims) == 3:
                break

    return {
        "generated": datetime.now(timezone.utc),
        "n_customers": len(features),
        "churn_rate": float(features["Churn"].mean()),
        "mrr": float(features["MonthlyCharges"].sum()),
        "arpu": float(features["MonthlyCharges"].mean()),
        "avg_tenure": float(features["tenure"].mean()),
        "margin_at_risk": margin_at_risk,
        "hazard": hazard_points,
        "bands": [(b, int(band_counts.get(b, 0))) for b in RISK_ORDER],
        "by_contract": by_contract,
        "segments": seg_summary,
        "topics": topics,
        "verbatims": verbatims,
        "call_list": call_list,
        "targeted_n": int(len(targeted)),
        "campaign_cost": float(len(targeted) * float(econ["retention_offer_cost"])),
        "campaign_net": float(targeted["expected_value_of_offer"].sum()),
        "quality": quality,
        "model": model,
        "econ": econ,
    }


# ------------------------------------------------------------------------ helpers
def money(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:,.0f}"


def tile(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="tile-sub">{escape(sub)}</div>' if sub else ""
    return (f'<div class="tile"><div class="tile-label">{escape(label)}</div>'
            f'<div class="tile-value">{escape(value)}</div>{sub_html}</div>')


def table(headers: list[str], rows: list[list[str]], numeric_from: int = 1) -> str:
    head = "".join(
        f'<th class="{"num" if i >= numeric_from else ""}">{escape(h)}</th>'
        for i, h in enumerate(headers)
    )
    body = "".join(
        "<tr>" + "".join(
            f'<td class="{"num" if i >= numeric_from else ""}">{cell}</td>'
            for i, cell in enumerate(row)
        ) + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def data_table(headers: list[str], rows: list[list[str]]) -> str:
    """Every chart carries a table view — identity and value never gated behind colour."""
    return (f'<details class="datatable"><summary>Data table</summary>'
            f'{table(headers, rows)}</details>')


def card(title: str, subtitle: str, body: str, span: str = "", extra: str = "",
         anchor: str = "") -> str:
    anchor_attr = f' id="{anchor}"' if anchor else ""
    return f"""<section class="card {span}"{anchor_attr}>
  <header class="card-head">
    <div><h3>{escape(title)}</h3><p>{escape(subtitle)}</p></div>{extra}
  </header>
  {body}
</section>"""


# ------------------------------------------------------------------------ page body
def render_body(d: dict, cfg: Config) -> str:
    model = d["model"]
    metrics = model.get("final_metrics", {})
    policy = model.get("decision_policy", {})
    # ROI recomputed over the whole scored base, not the stored test-set figure —
    # otherwise this tile would quietly contradict the hero, which is a full-base number.
    roi = ((d["campaign_net"] + d["campaign_cost"]) / d["campaign_cost"]) if d["campaign_cost"] else 0.0
    gen = d["generated"].strftime("%d %b %Y, %H:%M UTC")

    # ---- overview ---------------------------------------------------------
    tiles = "".join([
        tile("Customers", f"{d['n_customers']:,}", "in the warehouse"),
        tile("Churn rate", f"{d['churn_rate']:.1%}", "observed, all cohorts"),
        tile("Monthly recurring revenue", money(d["mrr"]), f"ARPU {money(d['arpu'])}"),
        tile("Average tenure", f"{d['avg_tenure']:.0f} mo", "across the base"),
        tile("Model ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}",
             f"{model.get('selected_model', 'n/a').replace('_', ' ')}, {model.get('calibration', 'n/a')}-calibrated"),
        tile("Campaign ROI", f"{roi:.2f}×", "expected, at the chosen threshold"),
    ])

    hero = f"""<section class="hero">
  <div>
    <div class="hero-label">Expected gross margin at risk · next 12 months</div>
    <div class="hero-value">{money(d['margin_at_risk'])}</div>
    <p class="hero-note">Probability-weighted exposure — Σ P(churn) × annual gross margin — not a
    worst case. Of that, the model judges <strong>{money(d['campaign_net'])}</strong> recoverable at
    a campaign cost of {money(d['campaign_cost'])} across {d['targeted_n']:,} targeted customers.</p>
  </div>
  <div class="hero-side">
    <div class="hero-side-row"><span>Targeted by policy</span><strong>{d['targeted_n']:,}</strong></div>
    {meter(d['targeted_n'], d['n_customers'])}
    <div class="hero-side-row"><span>Offer cost</span><strong>{money(float(d['econ']['retention_offer_cost']))}</strong></div>
    <div class="hero-side-row"><span>Assumed save rate</span><strong>{float(d['econ']['offer_success_rate']):.0%}</strong></div>
    <div class="hero-side-row"><span>Decision threshold</span><strong>{policy.get('threshold', 0):.2f}</strong></div>
  </div>
</section>"""

    # ---- risk -------------------------------------------------------------
    hazard_card = card(
        "Churn rate by tenure",
        "Early-life hazard: risk is highest in the first six months and decays with tenure",
        line_area(d["hazard"], value_kind="pct0")
        + '<p class="caption">Horizontal axis: months since acquisition (bucket upper bound).</p>'
        + data_table(["Tenure (months)", "Churn rate"],
                     [[f"≤ {lbl}", f"{v:.1%}"] for lbl, v in d["hazard"]]),
        span="span-2",
    )

    band_card = card(
        "Customers by risk band",
        "Calibrated probability, banded for routing",
        columns([(b, n) for b, n in d["bands"]],
                colors=[RISK_TOKENS[b] for b, _ in d["bands"]], width=NARROW)
        + data_table(["Risk band", "Customers"], [[b, f"{n:,}"] for b, n in d["bands"]]),
    )

    contract = d["by_contract"]
    contract_card = card(
        "Churn rate by contract",
        "The single largest structural driver in the book",
        columns([(idx, float(row.churn_rate)) for idx, row in contract.iterrows()],
                value_kind="pct0", width=NARROW)
        + data_table(["Contract", "Customers", "Churn rate", "Margin at risk"],
                     [[idx, f"{int(row.customers):,}", f"{row.churn_rate:.1%}", money(float(row.margin))]
                      for idx, row in contract.iterrows()]),
    )

    seg = d["segments"]
    if not seg.empty:
        # anchor target for the sidebar's "Segments" link
        seg_items = [(idx, float(row.customers)) for idx, row in seg.iterrows()]
        total_customers = f"{d['n_customers']:,}"
        segment_card = card(
            "Behavioural segments",
            "K-Means, k chosen by silhouette, named from their own centroids",
            '<div class="donut-wrap">' + donut(seg_items, CATEGORICAL, "customers", total_customers) + "</div>"
            + legend([(idx, CATEGORICAL[i % 3]) for i, idx in enumerate(seg.index)])
            + data_table(["Segment", "Customers", "Churn rate", "Avg monthly", "Avg tenure"],
                         [[idx, f"{int(row.customers):,}", f"{row.churn_rate:.1%}",
                           money(float(row.avg_monthly)), f"{row.avg_tenure:.0f} mo"]
                          for idx, row in seg.iterrows()]),
            anchor="segments",
        )
    else:
        segment_card = ""

    call_rows = [
        [escape(str(row.customerID)),
         f'<span class="pill" style="--pill:{RISK_TOKENS.get(row.risk_band, "var(--series-1)")}">'
         f'{escape(str(row.risk_band))}</span>',
         f"{row.churn_probability:.0%}",
         f"{int(row.tenure)} mo",
         money(float(row.MonthlyCharges)),
         escape(str(row.Contract)),
         f"<strong>{money(float(row.expected_value_of_offer))}</strong>"]
        for row in d["call_list"].itertuples()
    ]
    call_card = card(
        "Retention call list",
        "Ranked by expected value of the offer, not by risk alone — a high-risk, low-margin "
        "customer is not worth contacting",
        table(["Customer", "Band", "P(churn)", "Tenure", "Monthly", "Contract", "Expected value"],
              call_rows, numeric_from=2),
        span="span-3",
    )

    # ---- voice of customer ------------------------------------------------
    topics = d["topics"]
    if not topics.empty:
        ranked = topics.sort_values("negative_share", ascending=False).head(6)
        plotted = ranked[ranked["negative_share"] > 0]
        voice_card = card(
            "Feedback themes by negative share",
            "NMF topics over support tickets, reviews and survey text",
            hbars([(str(r.topic_label), float(r.negative_share)) for r in plotted.itertuples()],
                  value_kind="pct0")
            + '<p class="caption">Themes with no negative messages are omitted from the bars and '
              'kept in the data table below.</p>' 
            + data_table(["Theme", "Messages", "Negative share", "Avg rating", "Customers"],
                         [[str(r.topic_label), f"{int(r.documents):,}", f"{r.negative_share:.0%}",
                           f"{r.avg_rating:.2f}", f"{int(r.customers_affected):,}"]
                          for r in ranked.itertuples()]),
            span="span-2",
        )
    else:
        voice_card = ""

    quotes = "".join(
        f'<blockquote>{escape(v["text"])}'
        f'<cite>{escape(v["channel"])} · {v["rating"]}/5</cite></blockquote>'
        for v in d["verbatims"]
    )
    quotes_card = card("What they actually said",
                       "Lowest-sentiment verbatims behind the themes",
                       f'<div class="quotes">{quotes}</div>') if quotes else ""

    # ---- model ops --------------------------------------------------------
    candidates = model.get("candidates", [])
    selected = model.get("selected_model")
    cand_card = ""
    if candidates:
        cand_rows = [
            [c["name"].replace("_", " ") + (" ✓" if c["name"] == selected else ""),
             f"{c['cv_roc_auc_mean']:.3f} ± {c['cv_roc_auc_std']:.3f}",
             f"{c['test_metrics']['roc_auc']:.3f}",
             f"{c['test_metrics']['pr_auc']:.3f}",
             f"{c['test_metrics']['brier']:.3f}"]
            for c in sorted(candidates, key=lambda c: -c["cv_roc_auc_mean"])
        ]
        cand_card = card(
            "Model selection",
            "Five-fold stratified cross-validation; the winner is calibrated before it ships",
            table(["Model", "CV ROC-AUC", "Test ROC-AUC", "PR-AUC", "Brier"], cand_rows)
            + '<p class="caption">The regularised baseline won here — the engineered features '
              'already capture the non-linearity gradient boosting would have found. Shipping the '
              'more complex model would cost interpretability and latency for nothing.</p>',
            span="span-2",
        )

    lift = model.get("lift_table", [])
    lift_card = ""
    if lift:
        lift_card = card(
            "Decile lift",
            "Churn concentration versus the base rate, best-scored decile first",
            columns([(str(r["decile"]), float(r["lift"])) for r in lift], value_kind="x",
                    label_values=False, width=NARROW)
            + f'<p class="caption">The top decile carries {lift[0]["lift"]:.2f}× the base churn rate '
              f'and contains {lift[0]["cumulative_capture"]:.0%} of all churners.</p>'
            + data_table(["Decile", "Customers", "Churn rate", "Lift", "Cumulative capture"],
                         [[str(r["decile"]), f"{r['customers']:,}", f"{r['churn_rate']:.1%}",
                           f"{r['lift']:.2f}×", f"{r['cumulative_capture']:.0%}"] for r in lift]),
        )

    quality = d["quality"]
    counts = quality["status"].value_counts().to_dict() if not quality.empty else {}
    quality_card = card(
        "Data quality gate",
        "Contract checks enforced before anything reaches the warehouse",
        f"""<div class="qgrid">
          <div class="qstat"><span class="dot-good"></span><strong>{counts.get('PASS', 0)}</strong> passed</div>
          <div class="qstat"><span class="dot-warn"></span><strong>{counts.get('WARN', 0)}</strong> warnings</div>
          <div class="qstat"><span class="dot-crit"></span><strong>{counts.get('FAIL', 0)}</strong> failures</div>
        </div>
        <p class="caption">Schema presence, per-column null budgets, duplicate budgets, numeric range
        assertions and a target base-rate sanity check. Results are persisted to
        <code>ops_data_quality</code> so quality is a time series, not a console message.</p>""",
    )

    nav_items = [("overview", "Overview"), ("risk", "Customer risk"), ("segments", "Segments"),
                 ("voice", "Voice of customer"), ("model", "Model ops")]
    nav = "".join(f'<a href="#{slug}"><span class="nav-dot"></span>{escape(name)}</a>'
                  for slug, name in nav_items)

    return f"""<div class="app" data-palette="#2a78d6,#eb6834,#1baf7a">
<aside class="sidebar">
  <div class="brand"><span class="brand-mark">IO</span><span class="brand-text">IntelliOps</span></div>
  <nav>{nav}</nav>
  <div class="sidebar-foot">
    <div class="badge-model">{escape(str(model.get('selected_model', 'model')).replace('_', ' '))}</div>
    <div class="sidebar-meta">ROC-AUC {metrics.get('roc_auc', 0):.3f}</div>
    <div class="sidebar-meta">Brier {metrics.get('brier', 0):.3f}</div>
    <div class="sidebar-meta">Threshold {policy.get('threshold', 0):.2f}</div>
  </div>
</aside>

<main class="main">
  <header class="topbar">
    <div>
      <h1>Customer Intelligence</h1>
      <p>Churn risk, retention economics and voice-of-customer across {d['n_customers']:,} accounts</p>
    </div>
    <div class="topbar-meta">
      <span class="chip chip-warn">Synthetic demo data</span>
      <span class="chip">Generated {escape(gen)}</span>
    </div>
  </header>

  <h2 id="overview" class="section-title">Overview</h2>
  {hero}
  <div class="tiles">{tiles}</div>

  <h2 id="risk" class="section-title">Customer risk</h2>
  <div class="grid">{hazard_card}{band_card}{contract_card}{segment_card}{quality_card}{call_card}</div>

  <h2 id="voice" class="section-title">Voice of customer</h2>
  <div class="grid">{voice_card}{quotes_card}</div>

  <h2 id="model" class="section-title">Model operations</h2>
  <div class="grid">{cand_card}{lift_card}</div>

  <footer class="foot">
    <p><strong>IntelliOps AI Platform</strong> — generated by
    <code>python -m intelliops.reporting.build_dashboard</code> from the same warehouse tables the
    API and Streamlit app read. Figures describe a synthetic dataset with a known generating
    process: they demonstrate the pipeline end to end, they are not evidence about real customers.</p>
    <p>Offer cost, save rate and margin multiplier are assumptions held in
    <code>configs/config.yaml</code>; every dollar figure is only as good as those three numbers.</p>
  </footer>
</main>
<div class="tip" id="tip" role="status" aria-live="polite"></div>
</div>"""


# ----------------------------------------------------------------------------- CSS
FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    'family=IBM+Plex+Mono:wght@400;500;600&'
    'family=IBM+Plex+Sans:wght@400;450;500;600;700&display=swap">'
)

# IBM Plex, not a neutral UI default: the subject is instrumentation — a warehouse,
# a scored base, a decision threshold — and Plex is a typeface drawn for engineering
# systems. Sans carries the prose; Mono carries anything the reader scans as a
# reading off an instrument: eyebrows, axis ticks, chips, numeric columns.
CSS = """
:root{
  color-scheme: light;
  --font-ui:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --font-mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  /* Neutrals carry a slight violet bias toward the accent — a pure grey reads unconsidered */
  --surface-1:#fbfbfd; --surface-2:#ffffff; --page:#f2f2f6;
  --text-primary:#0d0c14; --text-secondary:#4f4e5c; --text-muted:#85849a;
  --grid:#e3e2ec; --axis:#c4c3d2; --border:rgba(13,12,20,0.10);
  --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a;
  --series-1-soft:#cde2fb;
  --status-good:#0ca30c; --status-warning:#fab219; --status-serious:#ec835a; --status-critical:#d03b3b;
  --accent:#4a3aa7; --accent-soft:#ecebf9;
  --shadow:0 1px 2px rgba(13,12,20,.05), 0 8px 24px -12px rgba(13,12,20,.16);
  --radius:14px;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme: dark;
    --surface-1:#16161d; --surface-2:#1c1c25; --page:#0b0b10;
    --text-primary:#f7f7fb; --text-secondary:#c2c1cf; --text-muted:#8a89a0;
    --grid:#2a2a35; --axis:#3a3a48; --border:rgba(247,247,251,0.11);
    --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
    --series-1-soft:#184f95;
    --accent:#9085e9; --accent-soft:#221f38;
    --shadow:0 1px 2px rgba(0,0,0,.45), 0 8px 24px -12px rgba(0,0,0,.65);
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --surface-1:#16161d; --surface-2:#1c1c25; --page:#0b0b10;
  --text-primary:#f7f7fb; --text-secondary:#c2c1cf; --text-muted:#8a89a0;
  --grid:#2a2a35; --axis:#3a3a48; --border:rgba(247,247,251,0.11);
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
  --series-1-soft:#184f95;
  --accent:#9085e9; --accent-soft:#221f38;
  --shadow:0 1px 2px rgba(0,0,0,.45), 0 8px 24px -12px rgba(0,0,0,.65);
}

*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--text-primary);font-family:var(--font-ui);
  -webkit-font-smoothing:antialiased;line-height:1.5}
h1,h2,h3{text-wrap:balance}
a{color:inherit}
code{font-family:var(--font-mono);font-size:.9em;background:var(--accent-soft);
  padding:.1em .35em;border-radius:5px}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:4px}

.app{display:grid;grid-template-columns:236px minmax(0,1fr);min-height:100vh}

/* ---------- sidebar ---------- */
.sidebar{background:var(--surface-1);border-right:1px solid var(--border);
  padding:22px 16px;position:sticky;top:0;height:100vh;display:flex;flex-direction:column;gap:26px}
.brand{display:flex;align-items:center;gap:10px}
.brand-mark{width:32px;height:32px;border-radius:9px;background:var(--accent);color:#fff;
  display:grid;place-items:center;font-family:var(--font-mono);font-weight:600;font-size:13px}
.brand-text{font-weight:600;font-size:16px;letter-spacing:-.015em}
.sidebar nav{display:flex;flex-direction:column;gap:2px}
.sidebar nav a{display:flex;align-items:center;gap:10px;padding:9px 11px;border-radius:9px;
  text-decoration:none;color:var(--text-secondary);font-size:13.5px;font-weight:450}
.sidebar nav a:hover{background:var(--accent-soft);color:var(--text-primary)}
.nav-dot{width:6px;height:6px;border-radius:50%;background:var(--axis);flex:none}
.sidebar nav a:hover .nav-dot{background:var(--accent)}
.sidebar-foot{margin-top:auto;border-top:1px solid var(--border);padding-top:16px}
.badge-model{display:inline-block;background:var(--accent-soft);color:var(--accent);
  font-family:var(--font-mono);font-size:11.5px;font-weight:500;padding:4px 9px;border-radius:7px;
  margin-bottom:9px;text-transform:capitalize}
.sidebar-meta{font-family:var(--font-mono);font-size:11.5px;color:var(--text-muted)}

/* ---------- main ---------- */
.main{padding:26px 30px 56px;min-width:0}
.topbar{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;flex-wrap:wrap;
  margin-bottom:28px}
.topbar h1{margin:0;font-size:26px;font-weight:600;letter-spacing:-.025em}
.topbar p{margin:5px 0 0;color:var(--text-secondary);font-size:14px;max-width:64ch}
.topbar-meta{display:flex;gap:8px;flex-wrap:wrap}
.chip{font-family:var(--font-mono);font-size:11.5px;color:var(--text-secondary);
  background:var(--surface-1);border:1px solid var(--border);padding:5px 10px;border-radius:999px;
  white-space:nowrap}
.chip-warn{color:#8a5a00;background:#fff6e2;border-color:#f4dfae}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]) .chip-warn{color:#fab219;background:#2a2313;border-color:#4a3d1c}
}
:root[data-theme="dark"] .chip-warn{color:#fab219;background:#2a2313;border-color:#4a3d1c}

.section-title{font-family:var(--font-mono);font-size:11.5px;text-transform:uppercase;
  letter-spacing:.1em;color:var(--text-muted);font-weight:500;margin:34px 0 14px}
.section-title:first-of-type{margin-top:0}

/* ---------- hero ---------- */
.hero{background:var(--surface-1);border:1px solid var(--border);border-radius:var(--radius);
  box-shadow:var(--shadow);padding:24px 26px;display:grid;
  grid-template-columns:minmax(0,1.65fr) minmax(230px,.85fr);gap:26px;margin-bottom:14px}
.hero-label{font-family:var(--font-mono);font-size:11.5px;color:var(--text-secondary);
  text-transform:uppercase;letter-spacing:.07em}
.hero-value{font-size:54px;font-weight:600;letter-spacing:-.035em;line-height:1.02;margin:8px 0 10px}
.hero-note{margin:0;color:var(--text-secondary);font-size:13.5px;max-width:62ch}
.hero-side{border-left:1px solid var(--border);padding-left:22px;display:flex;flex-direction:column;gap:9px}
.hero-side-row{display:flex;justify-content:space-between;align-items:baseline;gap:12px;font-size:13px;
  color:var(--text-secondary)}
.hero-side-row strong{color:var(--text-primary);font-family:var(--font-mono);font-weight:500}
.meter{height:7px;border-radius:999px;background:var(--series-1-soft);overflow:hidden;margin:2px 0 8px}
.meter-fill{height:100%;border-radius:999px}

/* ---------- tiles ---------- */
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:12px}
.tile{background:var(--surface-1);border:1px solid var(--border);border-radius:var(--radius);
  padding:15px 16px;box-shadow:var(--shadow)}
.tile-label{font-size:12px;color:var(--text-secondary);font-weight:450}
.tile-value{font-size:26px;font-weight:600;letter-spacing:-.025em;margin-top:5px}
.tile-sub{font-family:var(--font-mono);font-size:11px;color:var(--text-muted);margin-top:4px}

/* ---------- cards & grid ---------- */
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;align-items:start}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:var(--radius);
  padding:18px 20px 20px;box-shadow:var(--shadow);min-width:0}
.card.span-2{grid-column:span 2}
.card.span-3{grid-column:span 3}
.card-head{display:flex;justify-content:space-between;align-items:flex-start;gap:14px;margin-bottom:12px}
.card-head h3{margin:0;font-size:15px;font-weight:600;letter-spacing:-.015em}
.card-head p{margin:4px 0 0;font-size:12.5px;color:var(--text-secondary);max-width:62ch}
.caption{font-size:12px;color:var(--text-muted);margin:10px 0 0;max-width:68ch}

/* ---------- charts ---------- */
.chart{width:100%;height:auto;display:block;overflow:visible}
.chart .grid{stroke:var(--grid);stroke-width:1}
.chart .tick{fill:var(--text-muted);font-family:var(--font-mono);font-size:10.5px}
.chart .rowlabel{fill:var(--text-secondary);font-size:12px}
.chart .pointlabel{fill:var(--text-primary);font-family:var(--font-mono);font-size:11.5px;font-weight:500}
.chart .dot{stroke:var(--surface-1);stroke-width:2}
.chart .bar{transition:opacity .12s ease}
.chart .bar:hover,.chart .dot:hover{opacity:.82;cursor:default}
.donut-wrap{display:grid;place-items:center;padding:4px 0 2px}
.donut{max-width:250px}
.chart .donut-value{fill:var(--text-primary);font-size:22px;font-weight:600}
.chart .donut-label{fill:var(--text-muted);font-family:var(--font-mono);font-size:10.5px}
.legend{display:flex;flex-wrap:wrap;gap:10px 16px;justify-content:center;margin-top:8px}
.legend-item{display:inline-flex;align-items:center;gap:7px;font-size:12px;color:var(--text-secondary)}
.legend-item i{width:10px;height:10px;border-radius:3px;flex:none}

/* ---------- tables ---------- */
.table-wrap{overflow-x:auto;margin-top:4px}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{text-align:left;font-family:var(--font-mono);font-weight:500;font-size:11px;
  text-transform:uppercase;letter-spacing:.05em;color:var(--text-muted);padding:8px 10px;
  border-bottom:1px solid var(--border);white-space:nowrap}
td{padding:9px 10px;border-bottom:1px solid var(--grid);color:var(--text-primary)}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover td{background:var(--accent-soft)}
th.num,td.num{text-align:right}
td.num{font-family:var(--font-mono);font-variant-numeric:tabular-nums}
.pill{display:inline-block;font-family:var(--font-mono);font-size:10.5px;font-weight:500;
  padding:2.5px 8px;border-radius:999px;color:var(--pill);border:1px solid var(--pill);
  background:transparent;white-space:nowrap;text-transform:uppercase;letter-spacing:.04em}
.datatable{margin-top:12px}
.datatable summary{cursor:pointer;font-family:var(--font-mono);font-size:11.5px;
  color:var(--text-muted);padding:5px 0;user-select:none}
.datatable summary:hover{color:var(--text-primary)}

/* ---------- misc blocks ---------- */
.quotes{display:flex;flex-direction:column;gap:12px}
blockquote{margin:0;padding:12px 14px;background:var(--surface-2);border:1px solid var(--border);
  border-left:3px solid var(--status-critical);border-radius:9px;font-size:13px;color:var(--text-secondary)}
blockquote cite{display:block;margin-top:7px;font-style:normal;font-family:var(--font-mono);
  font-size:11px;color:var(--text-muted)}
.qgrid{display:flex;gap:18px;flex-wrap:wrap;margin:2px 0 4px}
.qstat{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--text-secondary)}
.qstat strong{font-size:19px;color:var(--text-primary);font-weight:600;
  font-family:var(--font-mono)}
.dot-good,.dot-warn,.dot-crit{width:9px;height:9px;border-radius:50%;flex:none}
.dot-good{background:var(--status-good)} .dot-warn{background:var(--status-warning)}
.dot-crit{background:var(--status-critical)}
.foot{margin-top:34px;padding-top:18px;border-top:1px solid var(--border);
  color:var(--text-muted);font-size:12px}
.foot p{margin:0 0 8px;max-width:88ch}

/* ---------- tooltip ---------- */
.tip{position:fixed;pointer-events:none;opacity:0;transform:translate(-50%,-130%);
  background:var(--text-primary);color:var(--surface-1);font-family:var(--font-mono);font-size:11.5px;
  padding:6px 9px;border-radius:7px;white-space:nowrap;z-index:50;transition:opacity .1s ease}
.tip strong{font-weight:600}

@media (prefers-reduced-motion: reduce){
  *{transition-duration:.001ms !important;animation-duration:.001ms !important}
}
@media (max-width:1180px){ .grid{grid-template-columns:repeat(2,minmax(0,1fr))}
  .card.span-3,.card.span-2{grid-column:span 2} }
@media (max-width:860px){
  .app{grid-template-columns:1fr}
  .sidebar{position:static;height:auto;flex-direction:row;align-items:center;gap:14px;
    overflow-x:auto;border-right:none;border-bottom:1px solid var(--border)}
  .sidebar nav{flex-direction:row}
  .sidebar-foot{margin:0;border:none;padding:0;display:flex;gap:10px;align-items:center}
  .sidebar-meta{white-space:nowrap}
  .main{padding:18px 16px 40px}
  .grid{grid-template-columns:1fr} .card.span-2,.card.span-3{grid-column:span 1}
  .hero{grid-template-columns:1fr} .hero-side{border-left:none;padding-left:0;
    border-top:1px solid var(--border);padding-top:16px}
  .hero-value{font-size:40px}
}
"""

JS = """
(function(){
  var tip = document.getElementById('tip');
  if(!tip) return;
  function show(e){
    var t = e.target.closest('[data-tip]');
    if(!t){ tip.style.opacity = 0; return; }
    tip.innerHTML = t.getAttribute('data-tip') + ' &middot; <strong>' +
                    t.getAttribute('data-tipval') + '</strong>';
    tip.style.opacity = 1;
    tip.style.left = e.clientX + 'px';
    tip.style.top  = e.clientY + 'px';
  }
  document.addEventListener('mousemove', show, {passive:true});
  document.addEventListener('mouseleave', function(){ tip.style.opacity = 0; }, true);
})();
"""


# ------------------------------------------------------------------------- assembly
def build(cfg: Config | None = None) -> dict[str, str]:
    cfg = cfg or load_config()
    cfg.ensure_dirs()

    with stage(logger, "collect dashboard data"):
        data = collect(cfg)

    with stage(logger, "render HTML"):
        body = render_body(data, cfg)
        title = "IntelliOps Customer Intelligence"
        fragment = (f"<title>{title}</title>\n{FONT_LINK}\n<style>{CSS}</style>\n"
                    f"{body}\n<script>{JS}</script>")
        standalone = (
            f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"<title>{title}</title>\n{FONT_LINK}\n<style>{CSS}</style>\n</head>\n<body>\n{body}\n"
            f"<script>{JS}</script>\n</body>\n</html>\n"
        )

    out_dir = cfg.path("paths.reports")
    full_path = out_dir / "dashboard.html"
    frag_path = out_dir / "dashboard.body.html"
    full_path.write_text(standalone, encoding="utf-8")
    frag_path.write_text(fragment, encoding="utf-8")
    logger.info("Wrote %s (%.0f KB) and %s", full_path, len(standalone) / 1024, frag_path.name)
    return {"standalone": str(full_path), "fragment": str(frag_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the static executive dashboard")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    build(load_config(args.config) if args.config else None)


if __name__ == "__main__":
    main()
