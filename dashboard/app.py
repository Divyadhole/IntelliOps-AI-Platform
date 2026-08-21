"""IntelliOps executive dashboard (Streamlit).

Single scrolling page, deliberately: an executive view that hides its numbers behind
tabs gets read once. Everything a retention decision needs is on one screen, with the
interactive parts — campaign economics, live scoring, the RAG analyst — inline where
the number they change appears.

Two design decisions worth naming:

* **It shares its chart library and stylesheet with the static report.** Both render
  the same ``intelliops.reporting.svg`` primitives under the same CSS custom
  properties, so the app and the shareable HTML snapshot cannot drift apart — and the
  app needs no plotting dependency at all.
* **It reads the warehouse directly.** The API is called only for live single-customer
  scoring and the RAG analyst, so the dashboard still works with the API stopped.

Run: ``streamlit run dashboard/app.py``  (or ``make dashboard``)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from intelliops.config import load_config  # noqa: E402
from intelliops.data_pipeline import warehouse  # noqa: E402
from intelliops.reporting.build_dashboard import CSS, RISK_TOKENS, collect  # noqa: E402
from intelliops.reporting.svg import columns, donut, hbars, legend, line_area  # noqa: E402

st.set_page_config(page_title="IntelliOps · Customer Intelligence", page_icon="📊", layout="wide")

CFG = load_config()
API_URL = CFG.get("dashboard.api_url", "http://localhost:8000")
CATEGORICAL = ["var(--series-1)", "var(--series-2)", "var(--series-3)"]

# Streamlit paints its own chrome; these overrides hand the page back to our tokens.
STREAMLIT_CSS = """
.block-container{padding-top:1.6rem;padding-bottom:3rem;max-width:1500px}
#MainMenu, footer, header [data-testid="stDecoration"]{visibility:hidden}
[data-testid="stSidebar"]{background:var(--surface-1);border-right:1px solid var(--border)}
[data-testid="stMetricValue"]{font-size:26px;font-weight:650;letter-spacing:-.02em}
[data-testid="stMetricLabel"]{color:var(--text-secondary)}
div[data-testid="stVerticalBlockBorderWrapper"]{background:var(--surface-1);border-radius:14px}
.stApp{background:var(--page)}
h1,h2,h3{letter-spacing:-.02em}
.sec{font-size:12px;text-transform:uppercase;letter-spacing:.09em;color:var(--text-muted);
  font-weight:650;margin:26px 0 10px}
"""


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


st.markdown(f"<style>{CSS}{STREAMLIT_CSS}</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- data
@st.cache_data(ttl=300, show_spinner="Reading the warehouse…")
def load_all() -> dict:
    return collect(CFG)


@st.cache_data(ttl=300, show_spinner=False)
def load_table(name: str) -> pd.DataFrame:
    try:
        return warehouse.read_table(name, CFG)
    except Exception:
        return pd.DataFrame()


def money(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:,.0f}"


@st.cache_resource(show_spinner=False)
def bootstrap() -> str:
    """Build anything missing before the first render.

    A deployed demo has no `make all` step — the host clones the repo and runs this
    file into an empty container. Showing a stack trace to the first visitor is not
    an acceptable cold start, so the app builds its own warehouse, model and index.
    """
    from intelliops.bootstrap import ensure_platform_ready

    return ensure_platform_ready(CFG).summary()


with st.spinner("First run here — building the warehouse, training the model and indexing "
                "feedback. About a minute, then it is cached."):
    BOOT = bootstrap()

try:
    D = load_all()
except Exception as exc:  # pragma: no cover - only reachable if bootstrap itself failed
    st.error(f"Could not read the warehouse: {exc}\n\nTry `make all` from the repo root.")
    st.stop()


# ------------------------------------------------ inference: API, else in-process
@st.cache_resource(show_spinner=False)
def local_scorer():
    from intelliops.churn_model.predict import ChurnScorer

    return ChurnScorer(cfg=CFG)


@st.cache_resource(show_spinner=False)
def local_assistant():
    from intelliops.rag_assistant.assistant import BusinessAnalystAssistant

    return BusinessAnalystAssistant(cfg=CFG)


def score_customer(payload: dict) -> tuple[dict, str]:
    """Prefer the API; fall back to the same objects in-process.

    The API is the production path and the dashboard should exercise it. But a
    single-container deployment has no API to call, and a demo whose two most
    interesting buttons return connection errors is worse than one that quietly
    loads the model itself — it is the identical ChurnScorer either way.
    """
    try:
        r = requests.post(f"{API_URL}/predict_churn", json=payload, timeout=10)
        r.raise_for_status()
        return r.json(), "API"
    except Exception:
        return local_scorer().explain_one(payload), "in-process"


def ask_analyst(question: str) -> tuple[dict, str]:
    try:
        r = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=90)
        r.raise_for_status()
        return r.json(), "API"
    except Exception:
        return local_assistant().ask(question).to_dict(), "in-process"

MODEL = D["model"]
METRICS = MODEL.get("final_metrics", {})
POLICY = MODEL.get("decision_policy", {})
predictions = load_table(CFG["warehouse.schema_tables.predictions"])
features = load_table(CFG["warehouse.schema_tables.features"])

# ------------------------------------------------------------------------ sidebar
with st.sidebar:
    html('<div class="brand"><span class="brand-mark">IO</span>'
         '<span class="brand-text">IntelliOps</span></div>')
    st.caption("Enterprise customer intelligence")

    st.divider()
    st.markdown("**Campaign economics**")
    st.caption("Every dollar figure below is only as good as these three assumptions.")
    econ = D["econ"]
    offer_cost = st.number_input("Retention offer cost ($)", 5.0, 500.0,
                                 float(econ["retention_offer_cost"]), 5.0)
    save_rate = st.slider("Assumed save rate", 0.05, 0.80,
                          float(econ["offer_success_rate"]), 0.05)
    margin_mult = st.slider("Gross margin on revenue", 0.05, 0.90,
                            float(econ["monthly_margin_multiplier"]), 0.05)
    horizon = int(econ["horizon_months"])

    st.divider()
    health = None
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
    except Exception:
        pass
    st.caption(f"API: {'🟢 ' + health['status'] if health else '⚪ offline — scoring runs in-process'}")
    st.caption(f"Model: {MODEL.get('selected_model', 'n/a').replace('_', ' ')}")
    st.caption(f"ROC-AUC {METRICS.get('roc_auc', 0):.3f} · Brier {METRICS.get('brier', 0):.3f}")

# ------------------------------------------------------------------------- header
left, right = st.columns([3, 1])
with left:
    st.title("Customer Intelligence")
    st.caption(f"Churn risk, retention economics and voice-of-customer across "
               f"{D['n_customers']:,} accounts")
with right:
    html('<div style="text-align:right;padding-top:12px">'
         '<span class="chip chip-warn">Synthetic demo data</span></div>')

# ------------------------------------------------------------------- recomputation
# The sidebar assumptions re-price the campaign live. Recomputing here rather than
# reading the stored column is the point: the user can see how sensitive the business
# case is to a save rate nobody has actually measured yet.
scored = predictions.merge(
    features[["customerID", "tenure", "MonthlyCharges", "Contract", "annual_margin_at_risk"]],
    on="customerID", how="left",
)
margin_at_risk_row = scored["MonthlyCharges"] * margin_mult * horizon
scored["ev"] = scored["churn_probability"] * save_rate * margin_at_risk_row - offer_cost
scored["margin_at_risk_row"] = margin_at_risk_row

threshold = st.slider(
    "Churn probability threshold for outreach", 0.05, 0.95,
    float(POLICY.get("threshold", 0.5)), 0.01,
    help="The model's own expected-value optimum is preselected. Customers below the "
         "line are not contacted; customers above it are contacted only when the offer "
         "still has positive expected value.",
)
targeted = scored[(scored["churn_probability"] >= threshold) & (scored["ev"] > 0)]
campaign_cost = len(targeted) * offer_cost
net_saving = float(targeted["ev"].sum())
roi = (net_saving + campaign_cost) / campaign_cost if campaign_cost else 0.0
exposure = float((scored["churn_probability"] * margin_at_risk_row).sum())

# ---------------------------------------------------------------------- overview
html('<div class="sec">Overview</div>')
html(f"""<div class="hero">
  <div>
    <div class="hero-label">Expected gross margin at risk · next {horizon} months</div>
    <div class="hero-value">{money(exposure)}</div>
    <p class="hero-note">Probability-weighted exposure — Σ P(churn) × annual gross margin — not a
    worst case. At the current threshold the model judges <strong>{money(net_saving)}</strong>
    recoverable for {money(campaign_cost)} spent on {len(targeted):,} customers.</p>
  </div>
  <div class="hero-side">
    <div class="hero-side-row"><span>Targeted</span><strong>{len(targeted):,}</strong></div>
    <div class="meter"><div class="meter-fill" style="width:{len(targeted) / max(1, len(scored)) * 100:.1f}%;
      background:var(--series-1)"></div></div>
    <div class="hero-side-row"><span>Campaign cost</span><strong>{money(campaign_cost)}</strong></div>
    <div class="hero-side-row"><span>Expected net saving</span><strong>{money(net_saving)}</strong></div>
    <div class="hero-side-row"><span>Campaign ROI</span><strong>{roi:.2f}×</strong></div>
  </div>
</div>""")

k = st.columns(6)
for col, (label, value, sub) in zip(k, [
    ("Customers", f"{D['n_customers']:,}", "in the warehouse"),
    ("Churn rate", f"{D['churn_rate']:.1%}", "observed"),
    ("Monthly revenue", money(D["mrr"]), f"ARPU {money(D['arpu'])}"),
    ("Average tenure", f"{D['avg_tenure']:.0f} mo", "across the base"),
    ("Model ROC-AUC", f"{METRICS.get('roc_auc', 0):.3f}",
     MODEL.get("selected_model", "").replace("_", " ")),
    ("Campaign ROI", f"{roi:.2f}×", "at this threshold"),
], strict=False):
    with col:
        html(f'<div class="tile"><div class="tile-label">{label}</div>'
             f'<div class="tile-value">{value}</div><div class="tile-sub">{sub}</div></div>')

# ------------------------------------------------------------------- customer risk
html('<div class="sec">Customer risk</div>')
c1, c2 = st.columns([2, 1])
with c1:
    with st.container(border=True):
        st.markdown("**Churn rate by tenure**")
        st.caption("Early-life hazard: risk peaks in the first six months and decays with tenure")
        html(line_area(D["hazard"], value_kind="pct0", width=700))
        st.caption("Horizontal axis: months since acquisition (bucket upper bound).")
with c2:
    with st.container(border=True):
        st.markdown("**Customers by risk band**")
        st.caption("Calibrated probability, banded for routing")
        bands = predictions["risk_band"].value_counts()
        items = [(b, int(bands.get(b, 0))) for b in ["Low", "Medium", "High", "Critical"]]
        html(columns(items, colors=[RISK_TOKENS[b] for b, _ in items], width=330))

c3, c4, c5 = st.columns(3)
with c3:
    with st.container(border=True):
        st.markdown("**Churn rate by contract**")
        st.caption("The largest structural driver in the book")
        contract = D["by_contract"]
        html(columns([(i, float(r.churn_rate)) for i, r in contract.iterrows()],
                     value_kind="pct0", width=330))
with c4:
    with st.container(border=True):
        st.markdown("**Behavioural segments**")
        st.caption("K-Means, k chosen by silhouette")
        seg = D["segments"]
        if seg.empty:
            st.info("Run `make segment` to populate segments.")
        else:
            html('<div class="donut-wrap">'
                 + donut([(i, float(r.customers)) for i, r in seg.iterrows()],
                         CATEGORICAL, "customers", f"{D['n_customers']:,}")
                 + "</div>"
                 + legend([(i, CATEGORICAL[n % 3]) for n, i in enumerate(seg.index)]))
with c5:
    with st.container(border=True):
        st.markdown("**Data quality gate**")
        st.caption("Contract checks enforced before anything reaches the warehouse")
        counts = D["quality"]["status"].value_counts().to_dict() if not D["quality"].empty else {}
        html(f"""<div class="qgrid">
          <div class="qstat"><span class="dot-good"></span><strong>{counts.get('PASS', 0)}</strong> passed</div>
          <div class="qstat"><span class="dot-warn"></span><strong>{counts.get('WARN', 0)}</strong> warnings</div>
          <div class="qstat"><span class="dot-crit"></span><strong>{counts.get('FAIL', 0)}</strong> failures</div>
        </div>""")
        with st.expander("All checks"):
            st.dataframe(D["quality"].sort_values("status"), use_container_width=True, height=240)

# ---------------------------------------------------------------------- call list
with st.container(border=True):
    st.markdown("**Retention call list**")
    st.caption("Ranked by expected value of the offer, not by risk alone — a high-risk, "
               "low-margin customer is not worth contacting")
    call_list = targeted.sort_values("ev", ascending=False)[
        ["customerID", "risk_band", "churn_probability", "tenure", "MonthlyCharges",
         "Contract", "ev", "recommended_action"]
    ].rename(columns={"ev": "expected_value", "churn_probability": "p_churn"})
    st.dataframe(
        call_list.head(200),
        use_container_width=True, height=380, hide_index=True,
        column_config={
            "p_churn": st.column_config.ProgressColumn("P(churn)", min_value=0.0, max_value=1.0,
                                                       format="%.0f%%"),
            "expected_value": st.column_config.NumberColumn("Expected value", format="$%.2f"),
            "MonthlyCharges": st.column_config.NumberColumn("Monthly", format="$%.2f"),
            "tenure": st.column_config.NumberColumn("Tenure", format="%d mo"),
        },
    )
    st.download_button("Download call list (CSV)", call_list.to_csv(index=False),
                       "retention_call_list.csv", "text/csv")

# ------------------------------------------------------------------ live scoring
html('<div class="sec">Score a customer</div>')
with st.container(border=True):
    st.caption(f"Calls POST {API_URL}/predict_churn when the API is up, and loads the same "
               "model in-process when it is not — identical result either way, with SHAP "
               "drivers rendered as retention actions.")
    f1, f2, f3, f4 = st.columns(4)
    payload = {
        "customerID": "AD-HOC",
        "tenure": f1.number_input("Tenure (months)", 0, 120, 3),
        "MonthlyCharges": f2.number_input("Monthly charges ($)", 0.0, 300.0, 95.0),
        "Contract": f3.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
        "InternetService": f4.selectbox("Internet", ["Fiber optic", "DSL", "No"]),
        "PaymentMethod": f1.selectbox("Payment", ["Electronic check", "Mailed check",
                                                  "Bank transfer (automatic)",
                                                  "Credit card (automatic)"]),
        "TechSupport": f2.selectbox("Tech support", ["No", "Yes"]),
        "OnlineSecurity": f3.selectbox("Online security", ["No", "Yes"]),
    }
    payload["TotalCharges"] = payload["tenure"] * payload["MonthlyCharges"]

    if st.button("Predict", type="primary"):
        with st.spinner("Scoring…"):
            res, via = score_customer(payload)
        m1, m2, m3 = st.columns(3)
        m1.metric("Churn probability", f"{res['churn_probability']:.1%}")
        m2.metric("Risk band", res["risk_band"])
        m3.metric("Expected value of offer", money(res["expected_value_of_offer"]))
        st.info(f"**Recommended action:** {res['recommended_action']}")
        st.dataframe(pd.DataFrame(res["top_drivers"]), use_container_width=True,
                     hide_index=True)
        st.caption(f"Scored via {via}.")

# ------------------------------------------------------------------ voice of customer
html('<div class="sec">Voice of customer</div>')
v1, v2 = st.columns([2, 1])
topics = D["topics"]
with v1:
    with st.container(border=True):
        st.markdown("**Feedback themes by negative share**")
        st.caption("NMF topics over support tickets, reviews and survey text")
        if topics.empty:
            st.info("Run `make nlp` to populate themes.")
        else:
            ranked = topics.sort_values("negative_share", ascending=False)
            plotted = ranked[ranked["negative_share"] > 0].head(6)
            html(hbars([(str(r.topic_label), float(r.negative_share)) for r in plotted.itertuples()],
                       value_kind="pct0", width=700))
            with st.expander("All themes"):
                st.dataframe(ranked, use_container_width=True, hide_index=True)
with v2:
    with st.container(border=True):
        st.markdown("**What they actually said**")
        st.caption("Lowest-sentiment verbatims behind the themes")
        for v in D["verbatims"]:
            html(f'<blockquote>{v["text"]}<cite>{v["channel"]} · {v["rating"]}/5</cite></blockquote>')

with st.container(border=True):
    st.markdown("**Ask the AI analyst**")
    st.caption("Retrieval-augmented over KPIs, model drivers, feedback themes and verbatims. "
               "Every claim carries a citation back to a warehouse row or a customer message.")
    question = st.text_input("Question", "Why are customers leaving?", label_visibility="collapsed")
    if st.button("Ask"):
        with st.spinner("Retrieving evidence…"):
            ans, via = ask_analyst(question)
        st.markdown(ans["answer"])
        st.caption(f"Answered via {via} · generated by {ans['generated_by']} "
                   f"{ans.get('model', '')}")
        with st.expander(f"Evidence ({len(ans['citations'])} items)"):
            for c in ans["citations"]:
                st.markdown(f"**[{c['id']}]** `{c['source']}` — {c['text']}")

# --------------------------------------------------------------------- model ops
html('<div class="sec">Model operations</div>')
m1, m2 = st.columns([2, 1])
with m1:
    with st.container(border=True):
        st.markdown("**Model selection**")
        st.caption("Five-fold stratified cross-validation; the winner is calibrated before it ships")
        candidates = MODEL.get("candidates", [])
        if candidates:
            flat = pd.DataFrame([{
                "model": c["name"].replace("_", " ") + (" ✓" if c["name"] == MODEL.get("selected_model") else ""),
                "cv_roc_auc": c["cv_roc_auc_mean"],
                "cv_std": c["cv_roc_auc_std"],
                **{k: v for k, v in c["test_metrics"].items() if k in ("roc_auc", "pr_auc", "brier")},
            } for c in sorted(candidates, key=lambda c: -c["cv_roc_auc_mean"])])
            st.dataframe(flat.style.format(precision=4), use_container_width=True, hide_index=True)
            st.caption("The regularised baseline won here — the engineered features already capture "
                       "the non-linearity gradient boosting would have found. Shipping the more "
                       "complex model would cost interpretability and latency for nothing.")
with m2:
    with st.container(border=True):
        st.markdown("**Decile lift**")
        st.caption("Churn concentration versus the base rate")
        lift = MODEL.get("lift_table", [])
        if lift:
            html(columns([(str(r["decile"]), float(r["lift"])) for r in lift],
                         value_kind="x", label_values=False, width=330))
            st.caption(f"Top decile carries {lift[0]['lift']:.2f}× the base rate and contains "
                       f"{lift[0]['cumulative_capture']:.0%} of all churners.")

st.divider()
st.caption("Figures describe a synthetic dataset with a known generating process: they demonstrate "
           "the pipeline end to end, they are not evidence about real customers. Offer cost, save "
           "rate and margin multiplier are assumptions — adjust them in the sidebar to see how "
           "sensitive the business case is.")
