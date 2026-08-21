# Notebooks

Notebooks here are for **exploration and communication only**. Nothing in the
platform imports from them — every transformation, model and metric lives in
`src/intelliops/`, is unit-tested and runs in CI.

That split is deliberate. A portfolio project whose logic lives in a notebook
cannot be scheduled, tested or deployed; one whose logic lives in a package can
be all three, and the notebook becomes a thin narrative layer on top of it.

## Suggested notebooks

| Notebook | Purpose | Imports from the package |
|---|---|---|
| `01_exploratory_analysis.ipynb` | Distributions, churn base rates, cohort curves, correlation structure | `data_pipeline.run_pipeline.run` |
| `02_model_development.ipynb` | Candidate comparison, learning curves, calibration plots, error analysis | `churn_model.train`, `churn_model.evaluate` |
| `03_explainability.ipynb` | SHAP summary/beeswarm/dependence plots, per-customer waterfalls | `churn_model.explain` |
| `04_nlp_deep_dive.ipynb` | Topic coherence, sentiment vs rating, theme–churn association | `nlp_engine` |
| `05_business_case.ipynb` | Threshold sensitivity, campaign ROI under different offer economics | `churn_model.evaluate` |

## Starter cell

```python
import sys; sys.path.insert(0, "../src")

from intelliops.config import load_config
from intelliops.data_pipeline import warehouse

cfg = load_config()
features = warehouse.read_table(cfg["warehouse.schema_tables.features"], cfg)
features.head()
```
