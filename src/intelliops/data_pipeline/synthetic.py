"""Synthetic data generator (schema-identical to IBM Telco Churn + review corpora).

Why this exists: a portfolio repo that only runs after someone manually downloads
three Kaggle datasets does not get run by recruiters. The generator produces data
with the *same schema, same dtypes and same messiness* as the real files — including
the notorious blank ``TotalCharges`` strings — so every downstream stage is exercised
end to end. Drop the real CSVs into ``data/raw/`` and ingestion prefers them
automatically; no code changes needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_YES_NO = ["Yes", "No"]
_INTERNET = ["DSL", "Fiber optic", "No"]
_CONTRACT = ["Month-to-month", "One year", "Two year"]
_PAYMENT = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
_ADDON_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

_COMPLAINT_TEMPLATES = [
    "The {topic} has been {bad_adj} for weeks and support keeps closing my ticket.",
    "I was charged twice this month. {topic} is {bad_adj} and nobody explains the bill.",
    "Delivery of my replacement router was late again. {topic} is {bad_adj}.",
    "Called three times about {topic}. Each agent gives a different answer. {bad_adj}.",
    "Price went up {pct}% with no warning. For this money the {topic} should not be {bad_adj}.",
    "Outages every evening. The {topic} is {bad_adj} during peak hours.",
]
_PRAISE_TEMPLATES = [
    "Honestly the {topic} has been {good_adj} since I switched to the annual plan.",
    "Support fixed my issue in one call. {topic} is {good_adj}.",
    "No complaints — {topic} is {good_adj} and the bill matches what I was quoted.",
    "Installation was quick and the {topic} has been {good_adj} ever since.",
]
_NEUTRAL_TEMPLATES = [
    "The {topic} is fine. Nothing special, nothing broken.",
    "Average experience overall. {topic} does what it says.",
    "Switched plans last month; {topic} works about the same as before.",
]
_TOPICS = [
    "internet speed", "billing process", "customer support", "streaming quality",
    "technical support", "device protection plan", "mobile app", "delivery time",
]
_BAD = ["unusable", "unacceptable", "constantly dropping", "painfully slow", "a mess"]
_GOOD = ["rock solid", "excellent", "very reliable", "better than expected", "great value"]


def generate_customers(n: int = 7500, seed: int = 42) -> pd.DataFrame:
    """Generate Telco-schema customer records with a learnable churn signal."""
    rng = np.random.default_rng(seed)

    tenure = rng.integers(0, 73, size=n)
    contract = rng.choice(_CONTRACT, size=n, p=[0.55, 0.21, 0.24])
    internet = rng.choice(_INTERNET, size=n, p=[0.34, 0.44, 0.22])
    payment = rng.choice(_PAYMENT, size=n, p=[0.34, 0.23, 0.22, 0.21])

    base = np.where(internet == "Fiber optic", 70.0, np.where(internet == "DSL", 45.0, 20.0))
    addons = {col: rng.choice(_YES_NO, size=n, p=[0.42, 0.58]) for col in _ADDON_COLS}
    addon_count = np.sum([(addons[c] == "Yes").astype(int) for c in _ADDON_COLS], axis=0)
    monthly = np.round(base + addon_count * rng.normal(6.5, 1.6, size=n) + rng.normal(0, 4, size=n), 2)
    monthly = np.clip(monthly, 18.0, 130.0)
    total = np.round(monthly * tenure * rng.normal(1.0, 0.04, size=n), 2)
    total = np.clip(total, 0, None)

    # ---- churn generating process ----------------------------------------
    # Deliberately NOT a clean logit: it contains threshold effects, an interaction
    # and a non-monotonic price response, so gradient boosting has something to find
    # that logistic regression cannot. Otherwise the model comparison is theatre.
    early_life = np.exp(-tenure / 9.0)                        # steep early-life hazard
    price_shock = np.clip((monthly - 78.0) / 20.0, 0, None) ** 1.6   # kicks in past a price point
    fiber_mtm = (internet == "Fiber optic") & (contract == "Month-to-month")

    logit = (
        -1.35
        + 1.55 * early_life
        - 0.018 * tenure
        + 0.85 * price_shock
        + 0.010 * monthly
        + 0.95 * (contract == "Month-to-month")
        - 0.85 * (contract == "Two year")
        + 0.35 * (internet == "Fiber optic")
        + 1.05 * fiber_mtm                                    # interaction: the churn hotspot
        + 0.45 * (payment == "Electronic check")
        - 0.45 * (addons["TechSupport"] == "Yes")
        - 0.35 * (addons["OnlineSecurity"] == "Yes")
        + 0.40 * (addon_count >= 4) * (contract == "Month-to-month")
        - 0.55 * (tenure > 48)                                # loyalty plateau
        + rng.normal(0, 0.45, size=n)
    )
    churn_prob = 1.0 / (1.0 + np.exp(-logit))
    churn = rng.binomial(1, churn_prob)

    df = pd.DataFrame(
        {
            "customerID": [f"{i:04d}-{''.join(rng.choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ'), 5))}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], size=n),
            "SeniorCitizen": rng.binomial(1, 0.16, size=n),
            "Partner": rng.choice(_YES_NO, size=n, p=[0.48, 0.52]),
            "Dependents": rng.choice(_YES_NO, size=n, p=[0.30, 0.70]),
            "tenure": tenure,
            "PhoneService": rng.choice(_YES_NO, size=n, p=[0.90, 0.10]),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], size=n, p=[0.42, 0.48, 0.10]),
            "InternetService": internet,
            **addons,
            "Contract": contract,
            "PaperlessBilling": rng.choice(_YES_NO, size=n, p=[0.59, 0.41]),
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )

    # ---- inject the real dataset's messiness ------------------------------
    # 1. TotalCharges arrives as a string column with blanks for tenure-0 customers
    df["TotalCharges"] = df["TotalCharges"].astype(str)
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "
    # 2. a sprinkle of genuine nulls in optional demographics
    null_idx = rng.choice(n, size=max(1, int(0.012 * n)), replace=False)
    df.loc[null_idx, "Partner"] = np.nan
    # 3. duplicate rows, which the validator must catch
    dupes = df.sample(n=max(1, int(0.004 * n)), random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def generate_reviews(customer_ids: list[str], n: int = 4000, seed: int = 42,
                     churn_flags: list[int] | None = None) -> pd.DataFrame:
    """Generate an unstructured feedback corpus keyed to customer IDs.

    When churn labels are supplied, sentiment is correlated with churn — churners
    skew negative. Without that correlation the NLP and RAG modules would be
    answering "why are customers leaving?" from pure noise.
    """
    rng = np.random.default_rng(seed + 1)
    ids = rng.choice(customer_ids, size=n, replace=True)

    if churn_flags is not None:
        churn_map = dict(zip(customer_ids, churn_flags, strict=False))
        churned = np.array([churn_map.get(cid, 0) for cid in ids])
        neg_p = [0.34, 0.22, 0.18, 0.15, 0.11]      # churners complain
        pos_p = [0.07, 0.09, 0.16, 0.30, 0.38]      # retained customers do not
        rating = np.where(
            churned == 1,
            rng.choice([1, 2, 3, 4, 5], size=n, p=neg_p),
            rng.choice([1, 2, 3, 4, 5], size=n, p=pos_p),
        )
    else:
        rating = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.19, 0.14, 0.17, 0.23, 0.27])

    texts, channels = [], []
    for r in rating:
        topic = rng.choice(_TOPICS)
        if r <= 2:
            tpl = rng.choice(_COMPLAINT_TEMPLATES)
        elif r == 3:
            tpl = rng.choice(_NEUTRAL_TEMPLATES)
        else:
            tpl = rng.choice(_PRAISE_TEMPLATES)
        texts.append(
            tpl.format(topic=topic, bad_adj=rng.choice(_BAD), good_adj=rng.choice(_GOOD), pct=rng.integers(5, 25))
        )
        channels.append(rng.choice(["support_ticket", "app_review", "survey", "email"], p=[0.35, 0.25, 0.25, 0.15]))

    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 730, size=n), unit="D")
    return pd.DataFrame(
        {
            "review_id": [f"R{i:06d}" for i in range(n)],
            "customerID": ids,
            "review_date": dates,
            "channel": channels,
            "rating": rating,
            "review_text": texts,
        }
    )
