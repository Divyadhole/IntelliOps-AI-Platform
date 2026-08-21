"""Customer segmentation — unsupervised structure, named for business use.

K is chosen by silhouette score rather than by eyeballing an elbow plot, and every
cluster is auto-labelled from its own centroid (value tier × loyalty tier) so the
segments arrive with names a marketing team can use rather than "Cluster 3".
"""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from ..config import Config, load_config
from ..data_pipeline import warehouse
from ..logging_utils import get_logger, stage

logger = get_logger(__name__)


def _label_cluster(profile: pd.Series, medians: pd.Series) -> str:
    """Turn a centroid into a business name."""
    high_value = profile["MonthlyCharges"] >= medians["MonthlyCharges"]
    loyal = profile["tenure"] >= medians["tenure"]
    engaged = profile.get("services_count", 0) >= medians.get("services_count", 0)

    if high_value and loyal:
        return "Premium Loyal"
    if high_value and not loyal:
        return "High-Value At-Risk"
    if not high_value and loyal:
        return "Steady Low-Spend"
    return "Price-Sensitive New"  if not engaged else "Emerging Multi-Product"


def segment(cfg: Config | None = None, features: pd.DataFrame | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    if features is None:
        features = warehouse.read_table(cfg["warehouse.schema_tables.features"], cfg)

    cols = [c for c in cfg["segmentation.features"] if c in features.columns]
    X = features[cols].astype(float).fillna(features[cols].median(numeric_only=True))

    # Cluster quality must be measured in the same space the clustering happened in;
    # scoring scaled labels against unscaled features is a classic silent bug.
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    best = None
    with stage(logger, "select k by silhouette"):
        for k in cfg["segmentation.k_range"]:
            km = KMeans(n_clusters=int(k), n_init=10, random_state=cfg.seed)
            labels = km.fit_predict(Xs)
            sil = float(silhouette_score(Xs, labels, sample_size=min(3000, len(Xs)), random_state=cfg.seed))
            db = float(davies_bouldin_score(Xs, labels))
            ch = float(calinski_harabasz_score(Xs, labels))
            logger.info("k=%d | silhouette %.4f | davies-bouldin %.4f | calinski-harabasz %.0f", k, sil, db, ch)
            if best is None or sil > best["silhouette"]:
                best = {"k": int(k), "silhouette": sil, "davies_bouldin": db,
                        "calinski_harabasz": ch, "labels": labels}

    labels = best["labels"]
    logger.info("Selected k=%d (silhouette %.4f)", best["k"], best["silhouette"])

    out = features[["customerID"]].copy()
    out["segment_id"] = labels

    profiles = features.assign(segment_id=labels).groupby("segment_id")[cols].mean()
    medians = features[cols].median()
    names = {sid: _label_cluster(profiles.loc[sid], medians) for sid in profiles.index}
    # Disambiguate collisions so every segment has a unique name.
    seen: dict[str, int] = {}
    for sid, name in list(names.items()):
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            names[sid] = f"{name} {seen[name]}"
    out["segment_name"] = out["segment_id"].map(names)

    summary = (
        features.assign(segment_id=labels, segment_name=out["segment_name"])
        .groupby("segment_name")
        .agg(customers=("customerID", "count"),
             avg_tenure=("tenure", "mean"),
             avg_monthly=("MonthlyCharges", "mean"),
             churn_rate=("Churn", "mean"))
        .round(3)
        .sort_values("churn_rate", ascending=False)
    )
    logger.info("Segment profile:\n%s", summary.to_string())

    warehouse.write_table(out, "fct_customer_segments", cfg)
    (cfg.path("paths.reports") / "segments.csv").write_text(summary.to_csv(), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run customer segmentation")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    segment(load_config(args.config) if args.config else None)


if __name__ == "__main__":
    main()
