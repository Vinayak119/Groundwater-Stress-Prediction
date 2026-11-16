# Tamil Nadu Groundwater Level Prediction

A research-oriented machine-learning project for exploring
groundwater-level observations and predicting the next available
groundwater-level measurement at monitoring stations across Tamil Nadu,
India.

The project combines CGWB manual groundwater-level observations, a
documented data-cleaning workflow, temporal and geographic evaluation,
trained regression models, explainability outputs, and a local Streamlit
dashboard.

> **Research scope:** This project predicts groundwater level (metres
> below ground level, m bgl). It does **not** establish an official
> groundwater-stress classification or replace CGWB resource
> assessments.

## Project highlights

-   77,024 cleaned groundwater-level observations.
-   2,318 provisional station identifiers in the cleaned data.
-   33 districts represented.
-   63,393 temporally eligible next-observation pairs after boundary
    purging.
-   Six saved regression models.
-   Temporal evaluation and a district-based geographic holdout.
-   SHAP-based global feature-importance outputs.
-   Interactive local Streamlit dashboard with six research pages.

## Data sources

The raw input files are stored in `data/raw/`:

-   `gwl_manual_quarterly_cgwb_tn_1991_2020.csv`
-   `gwl_manual_quarterly_cgwb_tn_2021_2025.csv`

The files are based on Central Ground Water Board (CGWB) manual
groundwater-level data for Tamil Nadu. Preserve the original source
attribution when redistributing or reusing the data, and check the
applicable CGWB terms and conditions.

**Date coverage note:** The first file's name suggests 1991--2020, but
the historical observations used in this project begin in January 1996
and extend through November 2020. The later file covers January 2021
through January 2025.

### Data quality and interpretation

The cleaned dataset contains 77,024 observations. The station
identifiers used in this project are provisional identifiers created for
analysis; they should not be interpreted as verified official well
identifiers.

Important data-handling decisions include:

-   Records with unresolved repeated station/date conflicts are
    quarantined rather than automatically averaged.
-   Anomaly flags are retained for investigation; unusual readings are
    not automatically deleted.
-   Negative groundwater-level values are not automatically removed
    because they may represent artesian or surface-saturation
    conditions.
-   Reduced-level-to-mean-sea-level values are too incomplete to serve
    as a general predictor.
-   Missing coordinates and uncertain station identity limit some
    geographic interpretations.

See `reports/` for cleaning summaries, anomaly registers, station
profiles, conflict reviews, and split summaries.

## Prediction task

The target is the **next available groundwater-level observation within
150 days** for a station, rather than a fixed 90-day or quarterly
forecast.

The feature set contains 12 variables:

1.  `GWL_m_bgl`
2.  `Previous_GWL_m_bgl`
3.  `Two_Observations_Back_GWL_m_bgl`
4.  `Historical_Mean_GWL_m_bgl`
5.  `Historical_Std_GWL_m_bgl`
6.  `History_Count`
7.  `Days_Since_Previous_Observation`
8.  `Current_Month`
9.  `Month_Sin`
10. `Month_Cos`
11. `Latitude_num`
12. `Longitude_num`

The saved model pipelines include median imputation. The dashboard
reconstructs the feature row and passes it to the selected saved
pipeline.

## Evaluation design

### Temporal evaluation

The temporal split uses earlier observations for training and later
periods for validation and testing. Boundary pairs are purged to reduce
leakage across split boundaries.

  ------------------------------------------------------------------------
  Split                                   Pair count Period definition
  --------------------- ---------------------------- ---------------------
  Train                                       51,816 Target date on or
                                                     before 2018-12-31

  Validation                                   3,227 Input date from
                                                     2019-01-01; target
                                                     date on or before
                                                     2020-12-31

  Test                                         8,350 Input date from
                                                     2021-01-01

  Purged boundary pairs                        1,041 Excluded at temporal
                                                     boundaries
  ------------------------------------------------------------------------

### Temporal benchmark results

The following are the recorded benchmark metrics (metres for MAE and
RMSE):

  -------------------------------------------------------------------------------------
  Model           Validation   Validation   Validation   Test MAE  Test RMSE    Test R²
                         MAE         RMSE           R²                       
  ------------- ------------ ------------ ------------ ---------- ---------- ----------
  Persistence         2.4601       5.1316       0.5959     2.1422     4.4432     0.8531
  Baseline                                                                   

  Decision Tree       2.4214       3.9707       0.7581     2.5898     5.1238     0.8047

  Random Forest       2.0208       3.4945       0.8126     2.2895     4.8299     0.8265

  XGBoost             1.9906       3.5236       0.8095     2.3226     5.1618     0.8018

  LightGBM            1.9678       3.4425       0.8181     2.2404     4.9585     0.8171
  -------------------------------------------------------------------------------------

**Interpretation:** LightGBM achieved the lowest validation MAE, while
the persistence baseline achieved the lowest overall temporal test MAE
and RMSE. The initial machine-learning models therefore did not
outperform persistence on the overall temporal test set.

### Geographic district holdout

Seven districts were held out from training: Kanyakumari, Karur,
Perambalur, Ramanathapuram, Sivaganga, Tiruvallur, and Viluppuram.

  Model              MAE     RMSE       R²
  ------------- -------- -------- --------
  Persistence     1.7788   3.7439   0.5621
  LightGBM        1.6497   3.2552   0.6690

On this internal district-based holdout, LightGBM improved MAE by
approximately 7.3% and RMSE by approximately 13.1% relative to
persistence. LightGBM performed better in six of the seven held-out
districts and worse in Kanyakumari.

This is an internal Tamil Nadu geographic holdout, **not** independent
external validation or cross-state validation.

## Explainability

The repository includes SHAP global feature-importance outputs. The
leading reported contributions are:

  Feature                                     Global importance
  ----------------------------------------- -------------------
  Current groundwater level                             40.706%
  Historical mean groundwater level                     20.743%
  Current month                                         13.702%
  Two-observations-back groundwater level                5.974%
  Previous groundwater level                             4.856%

These values describe model attribution in the analyzed data. They do
not demonstrate causal relationships.

## Prediction intervals

The project includes preliminary residual-quantile prediction intervals.
The reported interval radius is approximately ±4.218 m, with about
89.96% test coverage overall. Coverage is lower for the 121--150 day gap
group (about 87.76%).

These intervals are exploratory and should not be treated as formally
calibrated uncertainty estimates.

## Repository structure

``` text
.
├── dashboard/
│   ├── app.py
│   ├── components/
│   └── pages/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── trained/
├── reports/
├── results/
│   ├── metrics/
│   └── shap/
├── scripts/
├── src/
├── requirements.txt
└── requirements-lock.txt
```

### Dashboard pages

1.  **Geographic Analysis** --- station and district coverage, map, and
    station-level trends.
2.  **Data Explorer** --- filters, observation records, distributions,
    data quality, and station histories.
3.  **Groundwater Prediction** --- select a station and model to predict
    its next available observation.
4.  **Model Performance** --- temporal benchmark, geographic holdout,
    district errors, and forecast-gap analysis.
5.  **Explainable AI** --- global SHAP feature importance and
    interpretation notes.
6.  **Research Insights** --- methodology, validation, uncertainty, and
    research limitations.

## Installation

The project was developed with Python 3.11.9 on Windows.

Open PowerShell in the project root directory and run:

``` powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the project already has a working `.venv`, activate it instead of
creating another environment.

## Run the dashboard locally

From the project root:

``` powershell
python -m streamlit run .\dashboard\app.py
```

To use a specific local port:

``` powershell
python -m streamlit run .\dashboard\app.py --server.port 8501
```

Then open the local address printed by Streamlit (typically
`http://localhost:8501`).

The dashboard is designed for local use. No cloud deployment is
required.

## Reproducing the research workflow

The `scripts/` directory contains scripts for data auditing, cleaning,
forecasting dataset construction, temporal splits, model training,
geographic validation, tuning, explainability, and interval analysis.

Run scripts from the repository root so project-relative paths resolve
correctly. Review each script before execution, especially scripts that
regenerate processed datasets, metrics, or trained models.

The dashboard reads saved artifacts from `data/processed/`,
`models/trained/`, `results/`, and `reports/`. It does not need to
retrain the models at startup.

## Saved models

The following trained model files are included in `models/trained/`:

-   `decision_tree.joblib`
-   `random_forest.joblib`
-   `xgboost.joblib`
-   `lightgbm.joblib`
-   `lightgbm_tuned.joblib`
-   `lightgbm_geographic_holdout.joblib`

The Random Forest model is tracked using Git Large File Storage (Git
LFS). Install Git LFS before cloning if you need the full model file:

``` powershell
git lfs install
git clone https://github.com/Vinayak119/Groundwater-Stress-Prediction.git
```

If a clone contains an LFS pointer instead of the model contents, run:

``` powershell
git lfs pull
```

## Important limitations

-   The station identifiers are provisional and are not verified
    official well IDs.
-   Missing coordinates and station-identity uncertainty affect
    geographic analysis.
-   The next-observation target has variable time gaps up to 150 days.
-   Persistence outperformed the initial ML models on the overall
    temporal test set.
-   Geographic validation is internal to Tamil Nadu and does not
    establish performance in other states.
-   Groundwater depth below ground level alone is not a scientifically
    sufficient groundwater-stress classification.
-   No official CGWB stress category is predicted by this project.
-   SHAP importance is not causality.
-   Prediction intervals are preliminary and not formally calibrated.
-   Results should be independently validated before operational or
    policy use.

## License and data attribution

No open-source software license is specified in this repository unless a
separate `LICENSE` file is added. Until then, do not assume that the
source code is licensed for unrestricted reuse.

Raw groundwater data are attributed to the Central Ground Water Board
(CGWB), Government of India. Retain source attribution and follow the
source provider's applicable data-use and redistribution terms.

## Repository

[GitHub ---
Groundwater-Stress-Prediction](https://github.com/Vinayak119/Groundwater-Stress-Prediction)
