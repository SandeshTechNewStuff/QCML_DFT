"""
PHASE 1 — FINAL VALIDATION OF QC_MEDIAN ML MODEL

Objective
---------
This script performs the complete Phase-1 validation workflow for predicting
QC_Median from independent structural, compositional and electronic descriptors.

Workflow
--------
1. Load 2DMPDIA_ML_Prepared.pkl
2. Construct QC_Median from QC_data
3. Remove QC/DOS-derived features to prevent target leakage
4. Remove identifiers and non-numeric/problematic columns
5. Split data into:
       Development set = 80%
       Independent test set = 20%
6. Select the top 40 features using Mutual Information on DEVELOPMENT data only
7. Train ExtraTreesRegressor using the selected 40 features
8. Perform 5-fold cross-validation on the development set
9. Evaluate once on the untouched test set
10. Calculate:
       - R²
       - MAE
       - RMSE
11. Calculate ExtraTrees feature importance
12. Calculate permutation importance on the test set
13. Generate:
       - Actual vs Predicted QC_Median
       - Residual distribution
       - Top feature importance plot
14. Save:
       - Predictions
       - Metrics
       - Feature importance
       - Selected features
       - CV results
       - Trained model
       - Scaler/metadata

IMPORTANT
---------
QC/DOS-derived descriptors are explicitly excluded from X.
QC_Median is calculated only from QC_data and is never allowed to enter X.

Author: Sandesh V. Gaikwad
"""

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import warnings
import pickle
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

warnings.filterwarnings("ignore")


# ============================================================
# 2. USER SETTINGS
# ============================================================

INPUT_FILE = "2DMPDIA_ML_Prepared.pkl"

OUTPUT_DIR = "Phase1_QC_Median_Final"

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

TEST_SIZE = 0.20

N_FEATURES = 40

N_ESTIMATORS = 300

N_JOBS_MODEL = 2

N_JOBS_PERMUTATION = 1


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("PHASE 1 — QC_MEDIAN MODEL VALIDATION")
print("=" * 70)

print("\nLoading dataset...")

with open(INPUT_FILE, "rb") as f:
    data = pickle.load(f)

print("Dataset loaded.")
print("Object type:", type(data))


# ============================================================
# 4. IDENTIFY DATAFRAME
# ============================================================

if isinstance(data, pd.DataFrame):

    df = data.copy()

elif isinstance(data, dict):

    print("\nDictionary keys:")
    print(list(data.keys()))

    # Try to locate the main dataframe
    df_candidates = []

    for key, value in data.items():
        if isinstance(value, pd.DataFrame):
            df_candidates.append((key, value))

    if len(df_candidates) == 0:
        raise ValueError(
            "No pandas DataFrame was found inside the pickle file."
        )

    # Prefer a dataframe containing mat_id
    selected = None

    for key, value in df_candidates:
        if "mat_id" in value.columns:
            selected = value
            print(f"\nUsing DataFrame: {key}")
            break

    if selected is None:
        selected = df_candidates[0][1]
        print(f"\nUsing first DataFrame: {df_candidates[0][0]}")

    df = selected.copy()

else:
    raise TypeError(
        "Unsupported pickle structure. "
        "Expected pandas DataFrame or dictionary containing DataFrame."
    )


print("\nData shape:")
print(df.shape)

print("\nFirst 10 columns:")
print(df.columns[:10].tolist())


# ============================================================
# 5. FIND QC_DATA COLUMN
# ============================================================

qc_candidates = [
    c for c in df.columns
    if str(c).lower() in [
        "qc_data",
        "qc",
        "quantum_capacitance_data"
    ]
]

if len(qc_candidates) == 0:

    # More flexible search
    qc_candidates = [
        c for c in df.columns
        if "qc_data" in str(c).lower()
    ]

if len(qc_candidates) == 0:

    raise ValueError(
        "QC_data column was not found.\n"
        "Please inspect the dataframe columns and update the QC_DATA_COLUMN."
    )

QC_DATA_COLUMN = qc_candidates[0]

print("\nQC data column:")
print(QC_DATA_COLUMN)


# ============================================================
# 6. CONSTRUCT QC_MEDIAN
# ============================================================

print("\nConstructing QC_Median...")

def calculate_qc_median(value):
    """
    Calculate median QC from QC_data.

    Handles common storage formats:
    - list
    - numpy array
    - tuple
    - pandas Series
    """

    if value is None:
        return np.nan

    if isinstance(value, str):

        # Try to interpret string representation
        try:
            value = eval(value)
        except Exception:
            return np.nan

    try:

        arr = np.asarray(value, dtype=float)

        arr = arr[np.isfinite(arr)]

        if arr.size == 0:
            return np.nan

        return np.median(arr)

    except Exception:

        return np.nan


df["QC_Median"] = df[QC_DATA_COLUMN].apply(
    calculate_qc_median
)


# ============================================================
# 7. CHECK TARGET
# ============================================================

print("\nQC_Median statistics:")

print(df["QC_Median"].describe())

print("\nMissing QC_Median:",
      df["QC_Median"].isna().sum())

print(
    "Zero QC_Median:",
    (df["QC_Median"] == 0).sum()
)

print(
    "Non-zero QC_Median:",
    (df["QC_Median"] > 0).sum()
)

print(
    "Zero fraction:",
    (df["QC_Median"] == 0).mean()
)


# Remove rows with missing target
df = df.dropna(subset=["QC_Median"]).reset_index(drop=True)


# ============================================================
# 8. PREVENT TARGET LEAKAGE
# ============================================================

print("\nRemoving leakage-prone descriptors...")

LEAKAGE_TERMS = [
    "qc",
    "quantum_capacitance",
    "quantumcapacitance",
    "dos",
    "kuban",
    "density_of_states",
    "densityofstates"
]

def is_leakage_column(column_name):

    name = str(column_name).lower()

    return any(term in name for term in LEAKAGE_TERMS)


leakage_columns = [
    c for c in df.columns
    if is_leakage_column(c)
]

print("\nLeakage columns identified:")

for c in leakage_columns:
    print("  ", c)


# ============================================================
# 9. REMOVE IDENTIFIERS / NON-PREDICTIVE COLUMNS
# ============================================================

ID_COLUMNS = [
    "mat_id",
    "material_id",
    "id",
    "relative_id"
]

DROP_COLUMNS = set(leakage_columns)

for c in ID_COLUMNS:
    if c in df.columns:
        DROP_COLUMNS.add(c)


# QC target itself must not be X
DROP_COLUMNS.add("QC_Median")


# QC energy and status also should not be X
for c in ["QC_energy", "QC_status"]:
    if c in df.columns:
        DROP_COLUMNS.add(c)


print("\nColumns removed from X:")
print(len(DROP_COLUMNS))


# ============================================================
# 10. BUILD X AND y
# ============================================================

feature_columns = [
    c for c in df.columns
    if c not in DROP_COLUMNS
]

X = df[feature_columns].copy()

y = df["QC_Median"].copy()


# ============================================================
# 11. KEEP NUMERIC FEATURES ONLY
# ============================================================

numeric_features = X.select_dtypes(
    include=[np.number]
).columns.tolist()

X = X[numeric_features].copy()

print("\nInitial independent numeric descriptors:")
print(X.shape[1])


# ============================================================
# 12. REMOVE INF / NaN FEATURES
# ============================================================

X = X.replace([np.inf, -np.inf], np.nan)


# Remove columns that are completely missing
all_nan_columns = [
    c for c in X.columns
    if X[c].isna().all()
]

if all_nan_columns:

    print("\nRemoving all-NaN columns:")
    print(all_nan_columns)

    X = X.drop(columns=all_nan_columns)


# Median imputation using development data will be performed later.


# ============================================================
# 13. TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting data...")

X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print("\nDevelopment set:")
print(X_dev.shape)

print("\nTest set:")
print(X_test.shape)


# ============================================================
# 14. MEDIAN IMPUTATION
# ============================================================

print("\nPerforming median imputation...")

# IMPORTANT:
# Calculate imputation values ONLY from development data.

imputation_values = X_dev.median(numeric_only=True)

X_dev = X_dev.fillna(imputation_values)
X_test = X_test.fillna(imputation_values)

# Any remaining NaN values are filled with zero
# only if a feature has no valid development median.

X_dev = X_dev.fillna(0)
X_test = X_test.fillna(0)


# ============================================================
# 15. REMOVE CONSTANT FEATURES
# ============================================================

constant_features = [
    c for c in X_dev.columns
    if X_dev[c].nunique() <= 1
]

if constant_features:

    print(
        "\nRemoving constant features:",
        len(constant_features)
    )

    X_dev = X_dev.drop(columns=constant_features)
    X_test = X_test.drop(columns=constant_features)


# ============================================================
# 16. MUTUAL INFORMATION FEATURE SELECTION
# ============================================================

print("\n" + "=" * 70)
print("MUTUAL INFORMATION FEATURE SELECTION")
print("=" * 70)

print(
    f"\nSelecting top {N_FEATURES} features "
    "using DEVELOPMENT DATA ONLY..."
)


mi_values = mutual_info_regression(
    X_dev,
    y_dev,
    random_state=RANDOM_STATE
)


mi_table = pd.DataFrame({
    "Feature": X_dev.columns,
    "Mutual_Information": mi_values
})

mi_table = mi_table.sort_values(
    "Mutual_Information",
    ascending=False
).reset_index(drop=True)


top_features = mi_table.head(N_FEATURES)["Feature"].tolist()


print("\nTop selected features:")

for i, feature in enumerate(top_features, 1):
    print(
        f"{i:2d}. {feature:<50} "
        f"MI = {mi_table.loc[i-1, 'Mutual_Information']:.6f}"
    )


# ============================================================
# 17. CREATE FINAL FEATURE MATRICES
# ============================================================

X_dev_selected = X_dev[top_features].copy()

X_test_selected = X_test[top_features].copy()


# ============================================================
# 18. DEFINE FINAL EXTRA TREES MODEL
# ============================================================

model = ExtraTreesRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=None,
    max_features=1.0,
    min_samples_split=2,
    min_samples_leaf=2,
    bootstrap=False,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS_MODEL
)


# ============================================================
# 19. 5-FOLD CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION")
print("=" * 70)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


cv_results = []

for fold, (train_idx, val_idx) in enumerate(
    kf.split(X_dev_selected),
    start=1
):

    X_train_fold = X_dev_selected.iloc[train_idx]
    X_val_fold = X_dev_selected.iloc[val_idx]

    y_train_fold = y_dev.iloc[train_idx]
    y_val_fold = y_dev.iloc[val_idx]

    fold_model = ExtraTreesRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=None,
        max_features=1.0,
        min_samples_split=2,
        min_samples_leaf=2,
        bootstrap=False,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS_MODEL
    )

    fold_model.fit(
        X_train_fold,
        y_train_fold
    )

    pred_fold = fold_model.predict(
        X_val_fold
    )

    r2 = r2_score(
        y_val_fold,
        pred_fold
    )

    mae = mean_absolute_error(
        y_val_fold,
        pred_fold
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_val_fold,
            pred_fold
        )
    )

    cv_results.append({
        "Fold": fold,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse
    })

    print(
        f"Fold {fold}: "
        f"R² = {r2:.6f}, "
        f"MAE = {mae:.6e}, "
        f"RMSE = {rmse:.6e}"
    )


cv_df = pd.DataFrame(cv_results)


cv_mean = cv_df[
    ["R2", "MAE", "RMSE"]
].mean()

cv_std = cv_df[
    ["R2", "MAE", "RMSE"]
].std()


print("\nCross-validation summary:")

print(
    f"Mean CV R²   = "
    f"{cv_mean['R2']:.6f} ± {cv_std['R2']:.6f}"
)

print(
    f"Mean CV MAE  = "
    f"{cv_mean['MAE']:.6e} ± {cv_std['MAE']:.6e}"
)

print(
    f"Mean CV RMSE = "
    f"{cv_mean['RMSE']:.6e} ± {cv_std['RMSE']:.6e}"
)


# ============================================================
# 20. TRAIN FINAL MODEL ON DEVELOPMENT DATA
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL TRAINING")
print("=" * 70)

model.fit(
    X_dev_selected,
    y_dev
)


# ============================================================
# 21. PREDICTIONS
# ============================================================

y_train_pred = model.predict(
    X_dev_selected
)

y_test_pred = model.predict(
    X_test_selected
)


# ============================================================
# 22. METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    dataset_name
):

    return {
        "Dataset": dataset_name,
        "R2": r2_score(
            y_true,
            y_pred
        ),
        "MAE": mean_absolute_error(
            y_true,
            y_pred
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )
    }


train_metrics = calculate_metrics(
    y_dev,
    y_train_pred,
    "Development / Train"
)

test_metrics = calculate_metrics(
    y_test,
    y_test_pred,
    "Independent Test"
)


metrics_df = pd.DataFrame([
    train_metrics,
    test_metrics
])


print("\n" + "=" * 70)
print("FINAL MODEL PERFORMANCE")
print("=" * 70)

print(metrics_df.to_string(index=False))


# ============================================================
# 23. FEATURE IMPORTANCE — EXTRA TREES
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

feature_importance_df = pd.DataFrame({
    "Feature": top_features,
    "ExtraTrees_Importance": model.feature_importances_
})

feature_importance_df = feature_importance_df.sort_values(
    "ExtraTrees_Importance",
    ascending=False
).reset_index(drop=True)

print("\nTop 20 features:")

print(
    feature_importance_df.head(20).to_string(
        index=False
    )
)


# ============================================================
# 24. PERMUTATION IMPORTANCE
# ============================================================

print("\nCalculating permutation importance on independent TEST set...")

perm = permutation_importance(
    model,
    X_test_selected,
    y_test,
    scoring="r2",
    n_repeats=10,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS_PERMUTATION
)


permutation_df = pd.DataFrame({
    "Feature": top_features,
    "Permutation_Mean_R2_Importance": perm.importances_mean,
    "Permutation_STD": perm.importances_std
})

permutation_df = permutation_df.sort_values(
    "Permutation_Mean_R2_Importance",
    ascending=False
).reset_index(drop=True)


print("\nTop 20 permutation-important features:")

print(
    permutation_df.head(20).to_string(
        index=False
    )
)


# ============================================================
# 25. COMBINED FEATURE IMPORTANCE TABLE
# ============================================================

importance_df = pd.merge(
    mi_table,
    feature_importance_df,
    on="Feature",
    how="inner"
)

importance_df = pd.merge(
    importance_df,
    permutation_df,
    on="Feature",
    how="left"
)

importance_df["MI_Rank"] = (
    importance_df["Mutual_Information"]
    .rank(
        ascending=False,
        method="min"
    )
)

importance_df["ExtraTrees_Rank"] = (
    importance_df["ExtraTrees_Importance"]
    .rank(
        ascending=False,
        method="min"
    )
)

importance_df["Permutation_Rank"] = (
    importance_df[
        "Permutation_Mean_R2_Importance"
    ]
    .rank(
        ascending=False,
        method="min"
    )
)

importance_df = importance_df.sort_values(
    "ExtraTrees_Importance",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 26. ACTUAL VS PREDICTED DATA
# ============================================================

train_prediction_df = pd.DataFrame({
    "Sample_Index": X_dev_selected.index,
    "Dataset": "Development",
    "Actual_QC_Median": y_dev.values,
    "Predicted_QC_Median": y_train_pred
})

test_prediction_df = pd.DataFrame({
    "Sample_Index": X_test_selected.index,
    "Dataset": "Independent Test",
    "Actual_QC_Median": y_test.values,
    "Predicted_QC_Median": y_test_pred
})

prediction_df = pd.concat(
    [
        train_prediction_df,
        test_prediction_df
    ],
    ignore_index=True
)


prediction_df["Residual"] = (
    prediction_df["Actual_QC_Median"]
    -
    prediction_df["Predicted_QC_Median"]
)


prediction_df["Absolute_Error"] = (
    np.abs(
        prediction_df["Residual"]
    )
)


# ============================================================
# 27. RESIDUAL METRICS
# ============================================================

prediction_df["Absolute_Percent_Error"] = np.where(
    prediction_df["Actual_QC_Median"] != 0,
    100 *
    prediction_df["Absolute_Error"]
    /
    np.abs(
        prediction_df["Actual_QC_Median"]
    ),
    np.nan
)


# ============================================================
# 28. ACTUAL VS PREDICTED PLOT
# ============================================================

print("\nGenerating Actual vs Predicted plot...")


plt.rcParams["font.family"] = "Times New Roman"

LABEL_SIZE = 13
TICK_SIZE = 10
LINE_WIDTH = 2.2
MARKER_SIZE = 90
BORDER_WIDTH = 1.8


fig, ax = plt.subplots(
    figsize=(7, 6)
)


# Development data
ax.scatter(
    y_dev,
    y_train_pred,
    s=MARKER_SIZE,
    alpha=0.65,
    label="Development"
)


# Independent test data
ax.scatter(
    y_test,
    y_test_pred,
    s=MARKER_SIZE,
    alpha=0.75,
    label="Independent test"
)


# Determine plotting range
all_values = np.concatenate([
    np.asarray(y_dev),
    np.asarray(y_test),
    np.asarray(y_train_pred),
    np.asarray(y_test_pred)
])

plot_min = np.nanmin(all_values)
plot_max = np.nanmax(all_values)

margin = 0.05 * (
    plot_max - plot_min
)

plot_min -= margin
plot_max += margin


# 1:1 line
ax.plot(
    [plot_min, plot_max],
    [plot_min, plot_max],
    linewidth=LINE_WIDTH,
    linestyle="--",
    label="1:1 line"
)


ax.set_xlim(
    plot_min,
    plot_max
)

ax.set_ylim(
    plot_min,
    plot_max
)


ax.set_xlabel(
    "Actual QC Median (F cm$^{-2}$)",
    fontsize=LABEL_SIZE
)

ax.set_ylabel(
    "Predicted QC Median (F cm$^{-2}$)",
    fontsize=LABEL_SIZE
)


ax.tick_params(
    axis="both",
    labelsize=TICK_SIZE,
    direction="out",
    width=BORDER_WIDTH,
    top=False,
    right=False
)


# Square plot
ax.set_box_aspect(1)


# Border
for spine in ax.spines.values():

    spine.set_linewidth(
        BORDER_WIDTH
    )


# Metrics text
ax.text(
    0.05,
    0.95,
    f"Test R² = {test_metrics['R2']:.3f}\n"
    f"MAE = {test_metrics['MAE']:.2e}\n"
    f"RMSE = {test_metrics['RMSE']:.2e}",
    transform=ax.transAxes,
    verticalalignment="top",
    fontsize=10
)


ax.legend(
    fontsize=9,
    frameon=False
)


plt.tight_layout()


actual_predicted_path = os.path.join(
    OUTPUT_DIR,
    "QC_Median_Actual_vs_Predicted.png"
)

plt.savefig(
    actual_predicted_path,
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 29. RESIDUAL DISTRIBUTION
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 5)
)


ax.hist(
    y_test - y_test_pred,
    bins=30,
    edgecolor="black"
)


ax.axvline(
    0,
    linestyle="--",
    linewidth=LINE_WIDTH
)


ax.set_xlabel(
    "Residual: Actual − Predicted (F cm$^{-2}$)",
    fontsize=LABEL_SIZE
)

ax.set_ylabel(
    "Frequency",
    fontsize=LABEL_SIZE
)


ax.tick_params(
    axis="both",
    labelsize=TICK_SIZE,
    direction="out",
    top=False,
    right=False
)


for spine in ax.spines.values():

    spine.set_linewidth(
        BORDER_WIDTH
    )


plt.tight_layout()


residual_path = os.path.join(
    OUTPUT_DIR,
    "QC_Median_Test_Residual_Distribution.png"
)

plt.savefig(
    residual_path,
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 30. FEATURE IMPORTANCE PLOT
# ============================================================

top_n_plot = 20

plot_importance = (
    feature_importance_df
    .head(top_n_plot)
    .sort_values(
        "ExtraTrees_Importance"
    )
)


fig, ax = plt.subplots(
    figsize=(8, 7)
)


ax.barh(
    plot_importance["Feature"],
    plot_importance["ExtraTrees_Importance"]
)


ax.set_xlabel(
    "ExtraTrees Feature Importance",
    fontsize=LABEL_SIZE
)

ax.set_ylabel(
    "Feature",
    fontsize=LABEL_SIZE
)


ax.tick_params(
    axis="both",
    labelsize=TICK_SIZE,
    direction="out",
    top=False,
    right=False
)


for spine in ax.spines.values():

    spine.set_linewidth(
        BORDER_WIDTH
    )


plt.tight_layout()


importance_plot_path = os.path.join(
    OUTPUT_DIR,
    "QC_Median_Top20_Feature_Importance.png"
)

plt.savefig(
    importance_plot_path,
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 31. SAVE EXCEL RESULTS
# ============================================================

excel_path = os.path.join(
    OUTPUT_DIR,
    "QC_Median_Phase1_Final_Results.xlsx"
)


with pd.ExcelWriter(
    excel_path,
    engine="openpyxl"
) as writer:

    # Model metrics
    metrics_df.to_excel(
        writer,
        sheet_name="Model_Metrics",
        index=False
    )

    # CV folds
    cv_df.to_excel(
        writer,
        sheet_name="CV_Folds",
        index=False
    )

    # Selected features
    selected_features_df = pd.DataFrame({
        "Rank": range(
            1,
            len(top_features) + 1
        ),
        "Feature": top_features
    })

    selected_features_df.to_excel(
        writer,
        sheet_name="Selected_40_Features",
        index=False
    )

    # MI
    mi_table.to_excel(
        writer,
        sheet_name="Mutual_Information",
        index=False
    )

    # Combined importance
    importance_df.to_excel(
        writer,
        sheet_name="Feature_Importance",
        index=False
    )

    # Permutation importance
    permutation_df.to_excel(
        writer,
        sheet_name="Permutation_Importance",
        index=False
    )

    # Predictions
    prediction_df.to_excel(
        writer,
        sheet_name="Predictions",
        index=False
    )


# ============================================================
# 32. SAVE MODEL
# ============================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "QC_Median_ExtraTrees_Final.joblib"
)


joblib.dump(
    model,
    model_path
)


# ============================================================
# 33. SAVE IMPUTATION VALUES
# ============================================================

imputation_path = os.path.join(
    OUTPUT_DIR,
    "Development_Median_Imputation.joblib"
)


joblib.dump(
    imputation_values,
    imputation_path
)


# ============================================================
# 34. SAVE MODEL METADATA
# ============================================================

metadata = {

    "input_file": INPUT_FILE,

    "target": "QC_Median",

    "QC_source_column": QC_DATA_COLUMN,

    "n_total_samples": len(df),

    "development_samples": len(X_dev),

    "test_samples": len(X_test),

    "test_size": TEST_SIZE,

    "random_state": RANDOM_STATE,

    "n_selected_features": N_FEATURES,

    "model": "ExtraTreesRegressor",

    "n_estimators": N_ESTIMATORS,

    "max_depth": None,

    "max_features": 1.0,

    "min_samples_split": 2,

    "min_samples_leaf": 2,

    "bootstrap": False,

    "leakage_terms_excluded": LEAKAGE_TERMS,

    "selected_features": top_features,

    "CV_R2_mean": cv_mean["R2"],

    "CV_R2_std": cv_std["R2"],

    "CV_MAE_mean": cv_mean["MAE"],

    "CV_RMSE_mean": cv_mean["RMSE"],

    "Test_R2": test_metrics["R2"],

    "Test_MAE": test_metrics["MAE"],

    "Test_RMSE": test_metrics["RMSE"]
}


metadata_path = os.path.join(
    OUTPUT_DIR,
    "QC_Median_Model_Metadata.pkl"
)


with open(
    metadata_path,
    "wb"
) as f:

    pickle.dump(
        metadata,
        f
    )


# ============================================================
# 35. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 1 COMPLETED")
print("=" * 70)

print("\nTarget:")
print("QC_Median")

print("\nNumber of selected features:")
print(len(top_features))

print("\n5-fold CV:")
print(
    f"R²   = {cv_mean['R2']:.6f} ± {cv_std['R2']:.6f}"
)

print(
    f"MAE  = {cv_mean['MAE']:.6e}"
)

print(
    f"RMSE = {cv_mean['RMSE']:.6e}"
)

print("\nIndependent test:")
print(
    f"R²   = {test_metrics['R2']:.6f}"
)

print(
    f"MAE  = {test_metrics['MAE']:.6e}"
)

print(
    f"RMSE = {test_metrics['RMSE']:.6e}"
)


print("\nOutput files:")

print(
    "Excel:",
    excel_path
)

print(
    "Model:",
    model_path
)

print(
    "Actual vs Predicted:",
    actual_predicted_path
)

print(
    "Residual plot:",
    residual_path
)

print(
    "Feature importance:",
    importance_plot_path
)

print("\n" + "=" * 70)
