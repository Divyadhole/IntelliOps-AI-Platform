"""IntelliOps executive dashboard (Streamlit).

Reads the warehouse directly so it works whether or not the API is running, and
calls the API only for live single-customer scoring. Four views:

  Executive Overview — the numbers a VP asks for first
  Customer Risk      — the ranked call list, sorted by expected value, exportable
  AI Insights        — feedback themes and the RAG analyst
  Data & Model Ops   — data-quality checks and model evaluation, kept visible on purpose

Run: ``streamlit run dashboard/app.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from intelliops.config import load_config  # noqa: E402
from intelliops.data_pipeline import warehouse  # noqa: E402

st.set_page_config(page_title="IntelliOps AI Platform", page_icon="📊", layout="wide")

CFG = load_config()
API_URL = CFG.get("dashboard.api_url", "http://localhost:8000")
TABLES = CFG["warehouse.schema_tables"]


# ---------------------------------------------------------------- data access
@st.cache_data(ttl=300, show_spinner=False)
def load_table(name: str) -> pd.DataFrame:
    try:
        return warehouse.read_table(name, CFG)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_report(filename: str) -> dict:
    path = CFG.path("paths.reports") / filename
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def api_get(path: str, **params):
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def money(value: float) -> str:
    return f"${value:,.0f}"


features = load_table(TABLES["features"])
predictions = load_table(TABLES["predictions"])
quality = load_table(TABLES["quality"])
topics = load_table("agg_topic_summary")
reviews = load_table("fct_review_nlp")
segments = load_table("fct_customer_segments")
model_report = load_report("model_report.json")

if features.empty:
    st.error("The warehouse is empty. Run `make pipeline` and `make train` first.")
    st.stop()

# ------------------------------------------------------------------- sidebar
st.sidebar.title("IntelliOps AI Platform")
view = st.sidebar.radio(
    "View", ["Executive Overview", "Customer Risk", "AI Insights", "Data & Model Ops"]
)
health = api_get("/health")
st.sidebar.caption(
    f"API: {'🟢 ' + health['status'] if health else '⚪ offline (dashboard reads the warehouse directly)'}"
)
if model_report:
    st.sidebar.caption(
        f"Model: {model_report.get('selected_model')} · "
        f"ROC-AUC {model_report.get('final_metrics', {}).get('roc_auc', 0):.3f}"
    )

# -------------------------------------------------------- executive overview
if view == "Executive Overview":
    st.title("Executive Overview")

    churn_rate = features["Churn"].mean() if "Churn" in features else float("nan")
    mrr = features["MonthlyCharges"].sum()
    margin_at_risk = 0.0
    if not predictions.empty:
        merged = predictions.merge(features[["customerID", "annual_margin_at_risk"]], on="customerID", how="left")
        margin_at_risk = float((merged["churn_probability"] * merged["annual_margin_at_risk"]).sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Customers", f"{len(features):,}")
    c2.metric("Churn rate", f"{churn_rate:.1%}")
    c3.metric("Monthly recurring revenue", money(mrr))
    c4.metric("ARPU", money(features["MonthlyCharges"].mean()))
    c5.metric("Expected margin at risk (12m)", money(margin_at_risk),
              help="Σ P(churn) × annual gross margin — the probability-weighted exposure, not a worst case")

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Churn rate by contract")
        by_contract = (
            features.groupby("Contract")
            .agg(customers=("customerID", "count"), churn_rate=("Churn", "mean"),
                 margin_at_risk=("annual_margin_at_risk", "sum"))
            .sort_values("churn_rate", ascending=False)
        )
        st.bar_chart(by_contract["churn_rate"])
        st.dataframe(
            by_contract.style.format({"churn_rate": "{:.1%}", "margin_at_risk": "${:,.0f}"}),
            use_container_width=True,
        )

    with right:
        st.subheader("Churn rate by tenure bucket")
        order = ["0-6m", "6-12m", "1-2y", "2-4y", "4y+"]
        by_tenure = (
            features.groupby("tenure_bucket")["Churn"].mean().reindex(order).dropna()
        )
        st.bar_chart(by_tenure)
        st.caption("Early-life churn is the largest single lever — the first six months dominate.")

    if not segments.empty:
        st.divider()
        st.subheader("Behavioural segments")
        seg = features.merge(segments, on="customerID", how="left")
        seg_summary = (
            seg.groupby("segment_name")
            .agg(customers=("customerID", "count"), churn_rate=("Churn", "mean"),
                 avg_monthly=("MonthlyCharges", "mean"), avg_tenure=("tenure", "mean"))
            .sort_values("churn_rate", ascending=False)
        )
        st.dataframe(
            seg_summary.style.format(
                {"churn_rate": "{:.1%}", "avg_monthly": "${:,.2f}", "avg_tenure": "{:.1f}"}
            ),
            use_container_width=True,
        )

# ------------------------------------------------------------- customer risk
elif view == "Customer Risk":
    st.title("Customer Risk & Retention Targeting")
    if predictions.empty:
        st.warning("No predictions yet. Run `make train` to score the customer base.")
        st.stop()

    threshold = st.slider("Churn probability threshold", 0.0, 1.0,
                          float(CFG.get("dashboard.high_risk_threshold", 0.55)), 0.01)
    only_economic = st.checkbox(
        "Only customers where a retention offer has positive expected value", value=True,
        help="Expected value = P(churn) × save rate × margin at risk − offer cost",
    )

    view_df = predictions.merge(
        features[["customerID", "tenure", "MonthlyCharges", "Contract", "annual_margin_at_risk"]],
        on="customerID", how="left",
    )
    view_df = view_df[view_df["churn_probability"] >= threshold]
    if only_economic:
        view_df = view_df[view_df["expected_value_of_offer"] > 0]
    view_df = view_df.sort_values("expected_value_of_offer", ascending=False)

    econ = CFG["churn_model.economics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers in scope", f"{len(view_df):,}")
    c2.metric("Campaign cost", money(len(view_df) * float(econ["retention_offer_cost"])))
    c3.metric("Expected net saving", money(view_df["expected_value_of_offer"].sum()))

    st.subheader("Call list (ranked by expected value, not by risk alone)")
    st.dataframe(
        view_df.head(300).style.format(
            {"churn_probability": "{:.1%}", "expected_value_of_offer": "${:,.2f}",
             "MonthlyCharges": "${:,.2f}", "annual_margin_at_risk": "${:,.0f}"}
        ),
        use_container_width=True, height=430,
    )
    st.download_button("Download call list (CSV)", view_df.to_csv(index=False),
                       "retention_call_list.csv", "text/csv")

    st.divider()
    st.subheader("Score a customer live")
    st.caption(f"Calls POST {API_URL}/predict_churn — start the API with `make api`.")
    f1, f2, f3 = st.columns(3)
    payload = {
        "customerID": "AD-HOC",
        "tenure": f1.number_input("Tenure (months)", 0, 120, 3),
        "MonthlyCharges": f2.number_input("Monthly charges", 0.0, 300.0, 95.0),
        "Contract": f3.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
        "InternetService": f1.selectbox("Internet service", ["Fiber optic", "DSL", "No"]),
        "PaymentMethod": f2.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        ),
        "TechSupport": f3.selectbox("Tech support", ["No", "Yes"]),
    }
    payload["TotalCharges"] = payload["tenure"] * payload["MonthlyCharges"]

    if st.button("Predict", type="primary"):
        try:
            resp = requests.post(f"{API_URL}/predict_churn", json=payload, timeout=15)
            resp.raise_for_status()
            result = resp.json()
            m1, m2, m3 = st.columns(3)
            m1.metric("Churn probability", f"{result['churn_probability']:.1%}")
            m2.metric("Risk band", result["risk_band"])
            m3.metric("Expected value of offer", money(result["expected_value_of_offer"]))
            st.info(f"**Recommended action:** {result['recommended_action']}")
            st.subheader("Why (SHAP)")
            st.dataframe(pd.DataFrame(result["top_drivers"]), use_container_width=True)
        except Exception as exc:
            st.error(f"API call failed: {exc}. Is the API running on {API_URL}?")

# --------------------------------------------------------------- AI insights
elif view == "AI Insights":
    st.title("Customer Intelligence")

    if topics.empty:
        st.warning("No NLP output yet. Run `make nlp`.")
    else:
        st.subheader("Feedback themes ranked by negative share")
        show = topics.sort_values("negative_share", ascending=False)
        st.dataframe(
            show.style.format(
                {"negative_share": "{:.0%}", "avg_sentiment": "{:.2f}", "avg_rating": "{:.2f}"}
            ),
            use_container_width=True,
        )
        st.bar_chart(show.set_index("topic_label")["documents"])

        if not reviews.empty:
            theme = st.selectbox("Read verbatims for a theme", show["topic_label"].tolist())
            sample = reviews[reviews["topic_label"] == theme].head(8)
            for _, row in sample.iterrows():
                st.markdown(
                    f"> {row['review_text']}  \n"
                    f"<small>{row.get('channel', '')} · rating {row.get('rating', '?')}/5 · "
                    f"sentiment {row.get('sentiment_label', '')}</small>",
                    unsafe_allow_html=True,
                )

    st.divider()
    st.subheader("Ask the AI analyst")
    st.caption("Retrieval-augmented over KPIs, model drivers, feedback themes and verbatims.")
    question = st.text_input("Question", "Why are customers leaving?")
    if st.button("Ask", type="primary"):
        with st.spinner("Retrieving evidence…"):
            try:
                resp = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=90)
                resp.raise_for_status()
                answer = resp.json()
                st.markdown(answer["answer"])
                st.caption(f"Generated by: {answer['generated_by']} {answer.get('model', '')}")
                with st.expander(f"Evidence ({len(answer['citations'])} items)"):
                    for c in answer["citations"]:
                        st.markdown(f"**[{c['id']}]** `{c['source']}` — {c['text']}")
            except Exception as exc:
                st.error(f"API call failed: {exc}. Is the API running on {API_URL}?")

# ---------------------------------------------------------- data & model ops
else:
    st.title("Data & Model Ops")

    st.subheader("Data-quality checks (last pipeline run)")
    if quality.empty:
        st.warning("No quality report yet. Run `make pipeline`.")
    else:
        counts = quality["status"].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("Passed", int(counts.get("PASS", 0)))
        c2.metric("Warnings", int(counts.get("WARN", 0)))
        c3.metric("Failures", int(counts.get("FAIL", 0)))
        st.dataframe(quality.sort_values("status"), use_container_width=True, height=300)

    st.divider()
    st.subheader("Model evaluation")
    if not model_report:
        st.warning("No model report yet. Run `make train`.")
    else:
        st.caption(
            f"Selected **{model_report['selected_model']}**, calibration "
            f"`{model_report['calibration']}`, trained {model_report['trained_at']}"
        )
        candidates = pd.DataFrame(model_report["candidates"])
        flat = pd.concat(
            [candidates[["name", "cv_roc_auc_mean", "cv_roc_auc_std"]],
             pd.json_normalize(candidates["test_metrics"])], axis=1
        )
        st.dataframe(flat.style.format(precision=4), use_container_width=True)

        policy = model_report["decision_policy"]
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Decision threshold", f"{policy['threshold']:.2f}")
        p2.metric("Targeted (test set)", f"{policy['n_targeted']:,}")
        p3.metric("Expected net saving", money(policy["expected_net_saving"]))
        p4.metric("Campaign ROI", f"{policy['roi']:.2f}×")

        if model_report.get("lift_table"):
            st.subheader("Decile lift")
            lift = pd.DataFrame(model_report["lift_table"])
            st.bar_chart(lift.set_index("decile")["lift"])
            st.dataframe(lift, use_container_width=True)
