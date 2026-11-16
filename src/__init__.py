"""
src — Groundwater Stress ML source package.

Sub-modules
-----------
data_loader          : Load raw and processed CGWB datasets.
preprocessing        : Clean, validate and standardise observations.
feature_engineering  : Derive lag features, rolling statistics, season encodings.
model_training       : Train and serialise ML models.
evaluation           : Compute regression / classification metrics.
geographic_validation: Geographic hold-out experiment utilities.
explainability       : SHAP-based model interpretation helpers.
prediction           : Inference helpers for the dashboard.
"""
