"""


===========================================================================
PHASE 5 COMPLETED
===========================================================================

QC_Median anchor Test R²: 0.658168

Best formulation: Absolute_Residual
Best feature count: 40
Complete curve Test R²: 0.412619
Complete curve Test MAE: 1.360933e-04
Complete curve Test RMSE: 2.999176e-04






======================================================================
PHASE 5
MEDIAN-ANCHORED ENERGY-DEPENDENT QC CURVE PREDICTION
======================================================================

Objective
---------
Predict the complete QC(E) curve using the successful QC_Median
prediction model as an anchor.

Central hypothesis
------------------
QC(E) = QC_Median × Energy-dependent Shape Factor

Three formulations are tested:

1. ABSOLUTE RESIDUAL
       Residual(E) = QC(E) - QC_Median

       Predicted QC(E) =
           Predicted QC_Median + Predicted Residual(E)

2. RELATIVE DEVIATION
       Deviation(E) =
           [QC(E) - QC_Median] / [QC_Median + epsilon]

       Predicted QC(E) =
           Predicted QC_Median × [1 + Predicted Deviation(E)]

3. RATIO
       Ratio(E) =
           QC(E) / [QC_Median + epsilon]

       Predicted QC(E) =
           Predicted QC_Median × Predicted Ratio(E)

IMPORTANT LEAKAGE CONTROL
-------------------------
The actual QC_Median of the test material is NEVER used for
prediction.

For a test material:

    Independent descriptors
             |
             v
    QC_Median prediction model
             |
             v
    Predicted QC_Median
             |
             +------> Energy-dependent model
                         |
                         v
                   Predicted shape
                         |
                         v
                  Predicted QC(E)

The actual test QC curve is used only for final evaluation.

Models
------
Extra Trees

Feature sets:
    Top 40
    Top 60
    Top 80
    Top 100

Energy:
    -0.2 to +0.2 eV
    32 QC points

Validation:
    5-fold CV
    Independent 20% test set

Outputs
-------
    Phase5_Results.xlsx
    Phase5_Test_Predictions.xlsx
    Phase5_Feature_Importance.xlsx
    QC curve plots
    Actual vs Predicted plots
    R2 versus energy plots
    formulation comparison

======================================================================
"""

# ====================================================================
# 1. IMPORT LIBRARIES
# ====================================================================

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

warnings.filterwarnings("ignore")


# ====================================================================
# 2. USER SETTINGS
# ====================================================================

DATA_FILE = "2DMPDIA_ML_Prepared.pkl"

OUTPUT_DIR = "Phase5_Median_Anchored_QC"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

RANDOM_STATE = 42

TEST_SIZE = 0.20

N_SPLITS = 5

FEATURE_COUNTS = [
    40,
    60,
    80,
    100
]

N_ESTIMATORS = 500

ENERGY_MIN = -0.2
ENERGY_MAX = 0.2

# Small number used to avoid division by zero
EPSILON = 1e-10

# Publication figure settings
LABEL_SIZE = 13
TICK_SIZE = 10
VALUE_SIZE = 9
LINE_WIDTH = 2.2
MARKER_SIZE = 90
BORDER_WIDTH = 1.8

FONT_FAMILY = "Times New Roman"


# ====================================================================
# 3. LOAD DATA
# ====================================================================

print("\n" + "=" * 75)
print("PHASE 5: MEDIAN-ANCHORED QC CURVE PREDICTION")
print("=" * 75)

print("\nLoading:", DATA_FILE)

with open(
    DATA_FILE,
    "rb"
) as f:

    data = pickle.load(f)


# ====================================================================
# 4. IDENTIFY DATAFRAME
# ====================================================================

if isinstance(
    data,
    pd.DataFrame
):

    df = data.copy()

elif isinstance(
    data,
    dict
):

    candidates = [
        value
        for value in data.values()
        if isinstance(
            value,
            pd.DataFrame
        )
    ]

    if len(candidates) == 0:

        raise ValueError(
            "No DataFrame found inside pickle."
        )

    df = candidates[0]

else:

    raise TypeError(
        f"Unsupported pickle type: {type(data)}"
    )


print(
    "Dataset shape:",
    df.shape
)


# ====================================================================
# 5. EXTRACT QC DATA
# ====================================================================

if "QC_data" not in df.columns:

    raise KeyError(
        "QC_data column not found."
    )

if "QC_energy" not in df.columns:

    raise KeyError(
        "QC_energy column not found."
    )


def to_array(value):

    if isinstance(
        value,
        np.ndarray
    ):

        return np.asarray(
            value,
            dtype=float
        ).flatten()

    if isinstance(
        value,
        (list, tuple)
    ):

        return np.asarray(
            value,
            dtype=float
        ).flatten()

    if isinstance(
        value,
        pd.Series
    ):

        return value.to_numpy(
            dtype=float
        ).flatten()

    return np.asarray(
        value,
        dtype=float
    ).flatten()


QC_list = [
    to_array(x)
    for x in df["QC_data"]
]

lengths = [
    len(x)
    for x in QC_list
]

if len(set(lengths)) != 1:

    raise ValueError(
        "QC curves do not have equal lengths."
    )

QC_full = np.vstack(
    QC_list
)

ENERGY_full = to_array(
    df["QC_energy"].iloc[0]
)

print(
    "Original QC matrix:",
    QC_full.shape
)

print(
    "Original energy range:",
    ENERGY_full.min(),
    "to",
    ENERGY_full.max()
)


# ====================================================================
# 6. SELECT -0.2 TO +0.2 eV
# ====================================================================

energy_mask = (
    (ENERGY_full >= ENERGY_MIN - 1e-10)
    &
    (ENERGY_full <= ENERGY_MAX + 1e-10)
)

ENERGY = ENERGY_full[
    energy_mask
]

QC = QC_full[
    :,
    energy_mask
]

print(
    "\nSelected energy range:",
    f"{ENERGY.min():.4f}",
    "to",
    f"{ENERGY.max():.4f}",
    "eV"
)

print(
    "Selected QC points:",
    len(ENERGY)
)

if len(ENERGY) != 32:

    print(
        "\nWARNING:"
    )

    print(
        "Expected approximately 32 points."
    )

    print(
        "Actual points:",
        len(ENERGY)
    )


# ====================================================================
# 7. CALCULATE ACTUAL QC MEDIAN
# ====================================================================

# This is used only as a training target / for constructing
# residual and ratio targets.
#
# It is NEVER used as a predictor for the independent test set.

QC_Median = np.median(
    QC,
    axis=1
)

print(
    "\nQC_Median statistics:"
)

print(
    "Mean:",
    f"{QC_Median.mean():.6e}"
)

print(
    "Median:",
    f"{np.median(QC_Median):.6e}"
)

print(
    "Zero fraction:",
    f"{np.mean(QC_Median == 0) * 100:.2f}%"
)


# ====================================================================
# 8. IDENTIFY INDEPENDENT DESCRIPTORS
# ====================================================================

EXCLUDED_KEYWORDS = [

    "qc",

    "quantum_capacitance",

    "quantumcapacitance",

    "dos",

    "kuban",

    "density_of_states",

    "densityofstates"

]


def independent_feature(
    name
):

    lname = str(
        name
    ).lower()

    for keyword in EXCLUDED_KEYWORDS:

        if keyword in lname:

            return False

    return True


candidate_features = []

for col in df.columns:

    if col in [

        "QC_data",

        "QC_energy",

        "QC_status",

        "mat_id",

        "relative_id"

    ]:

        continue

    if independent_feature(
        col
    ):

        candidate_features.append(
            col
        )


# ====================================================================
# 9. CONVERT FEATURES TO NUMERIC
# ====================================================================

X_df = df[
    candidate_features
].copy()

for col in X_df.columns:

    X_df[col] = pd.to_numeric(
        X_df[col],
        errors="coerce"
    )


# Remove fully missing columns
valid_columns = [
    col
    for col in X_df.columns
    if X_df[col].notna().sum() > 0
]

X_df = X_df[
    valid_columns
]

feature_names = list(
    X_df.columns
)

X_raw = X_df.to_numpy(
    dtype=float
)

print(
    "\nIndependent descriptors:",
    X_raw.shape[1]
)


# ====================================================================
# 10. TRAIN / TEST SPLIT
# ====================================================================

indices = np.arange(
    len(df)
)

train_idx, test_idx = train_test_split(

    indices,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE
)

print(
    "\nTraining samples:",
    len(train_idx)
)

print(
    "Independent test samples:",
    len(test_idx)
)


# ====================================================================
# 11. TRAIN-ONLY FEATURE IMPUTATION
# ====================================================================

def impute_train_test(
    X,
    train_indices,
    test_indices
):

    X_train = X[
        train_indices
    ].copy()

    X_test = X[
        test_indices
    ].copy()

    medians = np.nanmedian(
        X_train,
        axis=0
    )

    medians = np.where(
        np.isfinite(medians),
        medians,
        0.0
    )

    train_bad = ~np.isfinite(
        X_train
    )

    rows, cols = np.where(
        train_bad
    )

    X_train[
        rows,
        cols
    ] = medians[
        cols
    ]

    test_bad = ~np.isfinite(
        X_test
    )

    rows, cols = np.where(
        test_bad
    )

    X_test[
        rows,
        cols
    ] = medians[
        cols
    ]

    return (
        X_train,
        X_test
    )


X_train_all, X_test_all = (
    impute_train_test(
        X_raw,
        train_idx,
        test_idx
    )
)


# ====================================================================
# 12. FEATURE SELECTION FOR QC_MEDIAN
# ====================================================================

print("\n" + "=" * 75)
print("FEATURE SELECTION FOR QC_MEDIAN")
print("=" * 75)

mi_median = mutual_info_regression(

    X_train_all,

    QC_Median[
        train_idx
    ],

    random_state=RANDOM_STATE
)

mi_median = np.nan_to_num(
    mi_median
)

median_feature_table = pd.DataFrame({

    "Feature":
        feature_names,

    "QC_Median_MI":
        mi_median

}).sort_values(

    "QC_Median_MI",

    ascending=False
).reset_index(
    drop=True
)

print(
    "\nTop 20 QC_Median features:"
)

print(
    median_feature_table.head(
        20
    ).to_string(
        index=False
    )
)


# ====================================================================
# 13. SELECT THE MAXIMUM FEATURE POOL
# ====================================================================

MAX_FEATURES = max(
    FEATURE_COUNTS
)

median_selected_features = (
    median_feature_table
    .head(MAX_FEATURES)
    ["Feature"]
    .tolist()
)

median_selected_indices = [

    feature_names.index(
        f
    )

    for f in median_selected_features

]

X_train_median = X_train_all[
    :,
    median_selected_indices
]

X_test_median = X_test_all[
    :,
    median_selected_indices
]


# ====================================================================
# 14. TRAIN QC_MEDIAN MODEL
# ====================================================================

print("\n" + "=" * 75)
print("TRAINING QC_MEDIAN ANCHOR MODEL")
print("=" * 75)

median_model = ExtraTreesRegressor(

    n_estimators=N_ESTIMATORS,

    random_state=RANDOM_STATE,

    n_jobs=-1,

    min_samples_leaf=2,

    max_features=1.0

)

median_model.fit(

    X_train_median,

    QC_Median[
        train_idx
    ]

)


# Predict training and test medians
pred_median_train = median_model.predict(
    X_train_median
)

pred_median_test = median_model.predict(
    X_test_median
)

# QC cannot physically be negative
pred_median_train = np.maximum(
    pred_median_train,
    0
)

pred_median_test = np.maximum(
    pred_median_test,
    0
)


# ====================================================================
# 15. QC_MEDIAN PERFORMANCE
# ====================================================================

median_train_r2 = r2_score(

    QC_Median[
        train_idx
    ],

    pred_median_train

)

median_test_r2 = r2_score(

    QC_Median[
        test_idx
    ],

    pred_median_test

)

median_test_mae = mean_absolute_error(

    QC_Median[
        test_idx
    ],

    pred_median_test

)

median_test_rmse = np.sqrt(

    mean_squared_error(

        QC_Median[
            test_idx
        ],

        pred_median_test

    )

)

print(
    "\nQC_Median model:"
)

print(
    "Train R²:",
    f"{median_train_r2:.6f}"
)

print(
    "Test R²:",
    f"{median_test_r2:.6f}"
)

print(
    "Test MAE:",
    f"{median_test_mae:.6e}"
)

print(
    "Test RMSE:",
    f"{median_test_rmse:.6e}"
)


# ====================================================================
# 16. CREATE THREE TARGET REPRESENTATIONS
# ====================================================================

# Training QC curves
QC_train = QC[
    train_idx
]

QC_test = QC[
    test_idx
]


# Actual median values
median_train_actual = QC_Median[
    train_idx
]

median_test_actual = QC_Median[
    test_idx
]


# --------------------------------------------------------------------
# A. Absolute residual
# --------------------------------------------------------------------

RESIDUAL_train = (
    QC_train
    -
    median_train_actual[:, None]
)

RESIDUAL_test = (
    QC_test
    -
    median_test_actual[:, None]
)


# --------------------------------------------------------------------
# B. Relative deviation
# --------------------------------------------------------------------

RELATIVE_train = (
    QC_train
    -
    median_train_actual[:, None]
) / (
    median_train_actual[:, None]
    +
    EPSILON
)

RELATIVE_test = (
    QC_test
    -
    median_test_actual[:, None]
) / (
    median_test_actual[:, None]
    +
    EPSILON
)


# --------------------------------------------------------------------
# C. Ratio
# --------------------------------------------------------------------

RATIO_train = (
    QC_train
    /
    (
        median_train_actual[:, None]
        +
        EPSILON
    )
)

RATIO_test = (
    QC_test
    /
    (
        median_test_actual[:, None]
        +
        EPSILON
    )
)


# ====================================================================
# 17. FUNCTION: TRAIN ENERGY-DEPENDENT MODEL
# ====================================================================

def train_energy_model(
    X_train,
    target_matrix,
    energy
):

    """
    Train one Extra Trees model using:

        material descriptors + energy

    to predict the requested QC-derived target.

    Each material contributes one observation per energy point.
    """

    n_materials = X_train.shape[0]

    n_energy = len(energy)

    n_features = X_train.shape[1]

    # Repeat material descriptors for each energy point
    X_repeated = np.repeat(
        X_train,
        n_energy,
        axis=0
    )

    # Repeat energy grid for every material
    E_repeated = np.tile(
        energy,
        n_materials
    ).reshape(
        -1,
        1
    )

    # Combine descriptors + energy
    X_model = np.hstack(
        [
            X_repeated,
            E_repeated
        ]
    )

    y_model = target_matrix.reshape(
        -1
    )

    model = ExtraTreesRegressor(

        n_estimators=N_ESTIMATORS,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        min_samples_leaf=3,

        max_features=1.0

    )

    model.fit(
        X_model,
        y_model
    )

    return model


# ====================================================================
# 18. FUNCTION: PREDICT ENERGY-DEPENDENT TARGET
# ====================================================================

def predict_energy_model(
    model,
    X,
    energy
):

    n_materials = X.shape[0]

    n_energy = len(
        energy
    )

    X_repeated = np.repeat(
        X,
        n_energy,
        axis=0
    )

    E_repeated = np.tile(
        energy,
        n_materials
    ).reshape(
        -1,
        1
    )

    X_model = np.hstack(
        [
            X_repeated,
            E_repeated
        ]
    )

    prediction = model.predict(
        X_model
    )

    return prediction.reshape(
        n_materials,
        n_energy
    )


# ====================================================================
# 19. FUNCTION: EVALUATE COMPLETE CURVE
# ====================================================================

def evaluate_curve(
    actual,
    predicted
):

    true_flat = actual.flatten()

    pred_flat = predicted.flatten()

    r2 = r2_score(
        true_flat,
        pred_flat
    )

    mae = mean_absolute_error(
        true_flat,
        pred_flat
    )

    rmse = np.sqrt(
        mean_squared_error(
            true_flat,
            pred_flat
        )
    )

    return (
        r2,
        mae,
        rmse
    )


# ====================================================================
# 20. STORAGE
# ====================================================================

all_results = []

feature_tables = []

prediction_store = {}

model_store = {}


# ====================================================================
# 21. TEST THREE FORMULATIONS
# ====================================================================

FORMULATIONS = [

    "Absolute_Residual",

    "Relative_Deviation",

    "Ratio"

]


for formulation in FORMULATIONS:

    print("\n" + "#" * 75)

    print(
        "FORMULATION:",
        formulation
    )

    print("#" * 75)


    # ================================================================
    # Select target
    # ================================================================

    if formulation == "Absolute_Residual":

        target_train = RESIDUAL_train

    elif formulation == "Relative_Deviation":

        target_train = RELATIVE_train

    elif formulation == "Ratio":

        target_train = RATIO_train


    # ================================================================
    # LOOP FEATURE COUNTS
    # ================================================================

    for n_features in FEATURE_COUNTS:

        print("\n" + "-" * 70)

        print(
            f"{formulation} | "
            f"Top {n_features} features"
        )

        print("-" * 70)


        # ------------------------------------------------------------
        # Select features
        # ------------------------------------------------------------

        selected_features = (
            median_feature_table
            .head(n_features)
            ["Feature"]
            .tolist()
        )

        selected_indices = [

            feature_names.index(
                f
            )

            for f in selected_features

        ]

        Xtr = X_train_all[
            :,
            selected_indices
        ]

        Xte = X_test_all[
            :,
            selected_indices
        ]


        # ============================================================
        # TRAIN ENERGY-DEPENDENT MODEL
        # ============================================================

        energy_model = train_energy_model(

            Xtr,

            target_train,

            ENERGY

        )


        # ============================================================
        # PREDICT TEST SHAPE
        # ============================================================

        predicted_shape = predict_energy_model(

            energy_model,

            Xte,

            ENERGY

        )


        # ============================================================
        # RECONSTRUCT QC CURVE
        # ============================================================

        if formulation == "Absolute_Residual":

            predicted_curve = (

                pred_median_test[:, None]

                +

                predicted_shape

            )


        elif formulation == "Relative_Deviation":

            predicted_curve = (

                pred_median_test[:, None]

                *

                (
                    1.0
                    +
                    predicted_shape
                )

            )


        elif formulation == "Ratio":

            predicted_curve = (

                pred_median_test[:, None]

                *

                predicted_shape

            )


        # QC cannot be negative
        predicted_curve = np.maximum(
            predicted_curve,
            0
        )


        # ============================================================
        # COMPLETE CURVE METRICS
        # ============================================================

        r2, mae, rmse = evaluate_curve(

            QC_test,

            predicted_curve

        )


        print(
            "\nIndependent test:"
        )

        print(
            "Curve R²:",
            f"{r2:.6f}"
        )

        print(
            "Curve MAE:",
            f"{mae:.6e}"
        )

        print(
            "Curve RMSE:",
            f"{rmse:.6e}"
        )


        # ============================================================
        # ENERGY-BY-ENERGY R²
        # ============================================================

        energy_r2 = []

        energy_mae = []

        energy_rmse = []

        for j in range(
            len(ENERGY)
        ):

            true_j = QC_test[
                :,
                j
            ]

            pred_j = predicted_curve[
                :,
                j
            ]

            try:

                r2_j = r2_score(
                    true_j,
                    pred_j
                )

            except Exception:

                r2_j = np.nan

            mae_j = mean_absolute_error(
                true_j,
                pred_j
            )

            rmse_j = np.sqrt(
                mean_squared_error(
                    true_j,
                    pred_j
                )
            )

            energy_r2.append(
                r2_j
            )

            energy_mae.append(
                mae_j
            )

            energy_rmse.append(
                rmse_j
            )


        # ============================================================
        # SAVE RESULTS
        # ============================================================

        all_results.append({

            "Formulation":
                formulation,

            "N_Features":
                n_features,

            "N_QC_Points":
                len(ENERGY),

            "Energy_Min":
                ENERGY.min(),

            "Energy_Max":
                ENERGY.max(),

            "QC_Median_Test_R2":
                median_test_r2,

            "Curve_Test_R2":
                r2,

            "Curve_Test_MAE":
                mae,

            "Curve_Test_RMSE":
                rmse

        })


        # ============================================================
        # SAVE FEATURE TABLE
        # ============================================================

        temp_features = (
            median_feature_table.copy()
        )

        temp_features[
            "Formulation"
        ] = formulation

        temp_features[
            "N_Features"
        ] = n_features

        temp_features[
            "Selected"
        ] = False

        temp_features.loc[
            temp_features.index
            <
            n_features,
            "Selected"
        ] = True

        feature_tables.append(
            temp_features
        )


        # ============================================================
        # STORE PREDICTIONS
        # ============================================================

        prediction_store[
            (
                formulation,
                n_features
            )
        ] = {

            "actual":
                QC_test.copy(),

            "predicted":
                predicted_curve.copy(),

            "shape":
                predicted_shape.copy(),

            "energy":
                ENERGY.copy(),

            "energy_r2":
                np.asarray(
                    energy_r2
                ),

            "energy_mae":
                np.asarray(
                    energy_mae
                ),

            "energy_rmse":
                np.asarray(
                    energy_rmse
                )

        }

        model_store[
            (
                formulation,
                n_features
            )
        ] = energy_model


# ====================================================================
# 22. RESULTS DATAFRAME
# ====================================================================

results_df = pd.DataFrame(
    all_results
)

feature_importance_df = pd.concat(
    feature_tables,
    ignore_index=True
)


# ====================================================================
# 23. PRINT COMPARISON
# ====================================================================

print("\n" + "=" * 75)

print(
    "PHASE 5 FORMULATION COMPARISON"
)

print("=" * 75)

print(
    results_df.to_string(
        index=False
    )
)


# ====================================================================
# 24. BEST MODEL
# ====================================================================

best_row = results_df.loc[
    results_df[
        "Curve_Test_R2"
    ].idxmax()
]

best_formulation = (
    best_row[
        "Formulation"
    ]
)

best_features = int(
    best_row[
        "N_Features"
    ]
)

print("\n" + "=" * 75)

print(
    "BEST PHASE 5 MODEL"
)

print("=" * 75)

print(
    "Formulation:",
    best_formulation
)

print(
    "Features:",
    best_features
)

print(
    "Test Curve R²:",
    f"{best_row['Curve_Test_R2']:.6f}"
)

print(
    "Test MAE:",
    f"{best_row['Curve_Test_MAE']:.6e}"
)

print(
    "Test RMSE:",
    f"{best_row['Curve_Test_RMSE']:.6e}"
)

###########################################################################################################

# ====================================================================
# 24A. SAVE FINAL QC RECONSTRUCTION MODEL
# ====================================================================

FINAL_MODEL_PACKAGE = {
    "median_model": median_model,

    "energy_model": model_store[
        (best_formulation, best_features)
    ],

    "formulation": best_formulation,
    "n_features": best_features,

    "median_selected_features":
        median_selected_features,

    "curve_selected_features":
        median_feature_table
        .head(best_features)["Feature"]
        .tolist(),

    "feature_names":
        feature_names,

    "energy":
        ENERGY.copy(),

    "energy_min":
        float(ENERGY_MIN),

    "energy_max":
        float(ENERGY_MAX),

    "epsilon":
        float(EPSILON),

    "random_state":
        RANDOM_STATE,

    "model_type":
        "ExtraTreesRegressor",

    "n_estimators":
        N_ESTIMATORS
}


FINAL_MODEL_FILE = os.path.join(
    OUTPUT_DIR,
    "Phase5_Final_QC_Reconstruction_Model.pkl"
)


with open(
    FINAL_MODEL_FILE,
    "wb"
) as f:

    pickle.dump(
        FINAL_MODEL_PACKAGE,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )


print(
    "\nFinal QC reconstruction model saved:"
)

print(
    os.path.abspath(
        FINAL_MODEL_FILE
    )
)
############################################################################################################


# ====================================================================
# 25. ENERGY-BY-ENERGY RESULTS FOR BEST MODEL
# ====================================================================

best_prediction = prediction_store[
    (
        best_formulation,
        best_features
    )
]

energy_results_df = pd.DataFrame({

    "Energy_eV":
        ENERGY,

    "R2":
        best_prediction[
            "energy_r2"
        ],

    "MAE":
        best_prediction[
            "energy_mae"
        ],

    "RMSE":
        best_prediction[
            "energy_rmse"
        ]

})

print("\n" + "=" * 75)

print(
    "ENERGY-RESOLVED PERFORMANCE"
)

print("=" * 75)

print(
    energy_results_df.to_string(
        index=False
    )
)


# ====================================================================
# 26. SAVE TEST PREDICTIONS
# ====================================================================

prediction_rows = []

for key, values in prediction_store.items():

    formulation, n_features = key

    actual = values[
        "actual"
    ]

    predicted = values[
        "predicted"
    ]

    for i in range(
        len(test_idx)
    ):

        row = {

            "Original_Index":
                int(test_idx[i]),

            "Formulation":
                formulation,

            "N_Features":
                n_features

        }

        for j, energy in enumerate(
            ENERGY
        ):

            label = (
                f"{energy:.3f}"
                .replace(
                    "-",
                    "m"
                )
                .replace(
                    ".",
                    "p"
                )
            )

            row[
                f"Actual_QC_{label}eV"
            ] = actual[
                i,
                j
            ]

            row[
                f"Predicted_QC_{label}eV"
            ] = predicted[
                i,
                j
            ]

        prediction_rows.append(
            row
        )


prediction_df = pd.DataFrame(
    prediction_rows
)

prediction_file = os.path.join(

    OUTPUT_DIR,

    "Phase5_Test_Predictions.xlsx"

)

prediction_df.to_excel(

    prediction_file,

    index=False

)

print(
    "\nPrediction file saved:",
    prediction_file
)


# ====================================================================
# 27. SAVE EXCEL RESULTS
# ====================================================================

excel_file = os.path.join(

    OUTPUT_DIR,

    "Phase5_Results.xlsx"

)

with pd.ExcelWriter(

    excel_file,

    engine="openpyxl"

) as writer:

    results_df.to_excel(

        writer,

        sheet_name="Model_Comparison",

        index=False

    )

    energy_results_df.to_excel(

        writer,

        sheet_name="Best_Energy_Performance",

        index=False

    )

    feature_importance_df.to_excel(

        writer,

        sheet_name="Feature_Selection",

        index=False

    )

    median_feature_table.to_excel(

        writer,

        sheet_name="QC_Median_Features",

        index=False

    )

print(
    "Results saved:",
    excel_file
)


# ====================================================================
# 28. FIGURE 1 — FORMULATION COMPARISON
# ====================================================================

plt.rcParams[
    "font.family"
] = FONT_FAMILY

fig, ax = plt.subplots(
    figsize=(8, 6)
)

for formulation in FORMULATIONS:

    temp = results_df[
        results_df[
            "Formulation"
        ]
        ==
        formulation
    ]

    ax.plot(

        temp[
            "N_Features"
        ],

        temp[
            "Curve_Test_R2"
        ],

        marker="o",

        linewidth=LINE_WIDTH,

        markersize=7,

        label=formulation.replace(
            "_",
            " "
        )

    )

ax.set_xlabel(

    "Number of selected features",

    fontsize=LABEL_SIZE

)

ax.set_ylabel(

    "Independent test R²",

    fontsize=LABEL_SIZE

)

ax.tick_params(

    axis="both",

    labelsize=TICK_SIZE,

    direction="out"

)

ax.tick_params(
    top=False,
    right=False
)

ax.legend(
    frameon=False,
    fontsize=9
)

for spine in ax.spines.values():

    spine.set_linewidth(
        BORDER_WIDTH
    )

ax.grid(False)

plt.tight_layout()

figure_file = os.path.join(

    OUTPUT_DIR,

    "Phase5_Formulation_Comparison.png"

)

plt.savefig(

    figure_file,

    dpi=600,

    bbox_inches="tight"

)

plt.show()


# ====================================================================
# 29. FIGURE 2 — BEST ACTUAL VS PREDICTED MEAN QC CURVE
# ====================================================================

actual_mean = np.mean(
    best_prediction[
        "actual"
    ],
    axis=0
)

predicted_mean = np.mean(
    best_prediction[
        "predicted"
    ],
    axis=0
)

fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.plot(

    ENERGY,

    actual_mean,

    linewidth=LINE_WIDTH,

    label="Actual"

)

ax.plot(

    ENERGY,

    predicted_mean,

    "--",

    linewidth=LINE_WIDTH,

    label="Predicted"

)

ax.set_xlabel(

    "Energy relative to Fermi level (eV)",

    fontsize=LABEL_SIZE

)

ax.set_ylabel(

    "Quantum capacitance (F cm$^{-2}$)",

    fontsize=LABEL_SIZE

)

ax.tick_params(

    axis="both",

    labelsize=TICK_SIZE,

    direction="out"

)

ax.tick_params(
    top=False,
    right=False
)

ax.legend(
    frameon=False,
    fontsize=10
)

for spine in ax.spines.values():

    spine.set_linewidth(
        BORDER_WIDTH
    )

ax.grid(False)

plt.tight_layout()

figure_file = os.path.join(

    OUTPUT_DIR,

    "Phase5_Best_Mean_QC_Curve.png"

)

plt.savefig(

    figure_file,

    dpi=600,

    bbox_inches="tight"

)

plt.show()


# ====================================================================
# 30. FIGURE 3 — ACTUAL VS PREDICTED QC
# ====================================================================

true_flat = best_prediction[
    "actual"
].flatten()

pred_flat = best_prediction[
    "predicted"
].flatten()

fig, ax = plt.subplots(
    figsize=(7, 7)
)

ax.scatter(

    true_flat,

    pred_flat,

    s=MARKER_SIZE,

    alpha=0.45

)

minimum = min(
    true_flat.min(),
    pred_flat.min()
)

maximum = max(
    true_flat.max(),
    pred_flat.max()
)

ax.plot(

    [minimum, maximum],

    [minimum, maximum],

    "--",

    linewidth=LINE_WIDTH

)

ax.set_xlabel(

    "Actual QC (F cm$^{-2}$)",

    fontsize=LABEL_SIZE

)

ax.set_ylabel(

    "Predicted QC (F cm$^{-2}$)",

    fontsize=LABEL_SIZE

)

ax.tick_params(

    axis="both",

    labelsize=TICK_SIZE,

    direction="out"

)

ax.tick_params(
    top=False,
    right=False
)

for spine in ax.spines.values():

    spine.set_linewidth(
        BORDER_WIDTH
    )

ax.grid(False)

plt.tight_layout()

figure_file = os.path.join(

    OUTPUT_DIR,

    "Phase5_Actual_vs_Predicted.png"

)

plt.savefig(

    figure_file,

    dpi=600,

    bbox_inches="tight"

)

plt.show()


# ====================================================================
# 31. FIGURE 4 — R² AS A FUNCTION OF ENERGY
# ====================================================================

fig, ax = plt.subplots(
    figsize=(8, 6)
)

for formulation in FORMULATIONS:

    key = (
        formulation,
        best_features
    )

    temp = prediction_store[
        key
    ]

    ax.plot(

        temp["energy"],

        temp["energy_r2"],

        marker="o",

        linewidth=LINE_WIDTH,

        markersize=5,

        label=formulation.replace(
            "_",
            " "
        )

    )

ax.axhline(
    0,
    linestyle="--",
    linewidth=1.2
)

ax.set_xlabel(

    "Energy relative to Fermi level (eV)",

    fontsize=LABEL_SIZE

)

ax.set_ylabel(

    "R² at each energy point",

    fontsize=LABEL_SIZE

)

ax.tick_params(

    axis="both",

    labelsize=TICK_SIZE,

    direction="out"

)

ax.tick_params(
    top=False,
    right=False
)

ax.legend(
    frameon=False,
    fontsize=9
)

for spine in ax.spines.values():

    spine.set_linewidth(
        BORDER_WIDTH
    )

ax.grid(False)

plt.tight_layout()

figure_file = os.path.join(

    OUTPUT_DIR,

    "Phase5_R2_vs_Energy.png"

)

plt.savefig(

    figure_file,

    dpi=600,

    bbox_inches="tight"

)

plt.show()


# ====================================================================
# 32. FIGURE 5 — MEDIAN PREDICTION
# ====================================================================

fig, ax = plt.subplots(
    figsize=(7, 7)
)

ax.scatter(

    QC_Median[
        test_idx
    ],

    pred_median_test,

    s=MARKER_SIZE,

    alpha=0.5

)

minimum = min(

    QC_Median[
        test_idx
    ].min(),

    pred_median_test.min()

)

maximum = max(

    QC_Median[
        test_idx
    ].max(),

    pred_median_test.max()

)

ax.plot(

    [minimum, maximum],

    [minimum, maximum],

    "--",

    linewidth=LINE_WIDTH

)

ax.set_xlabel(

    "Actual QC$_{Median}$ (F cm$^{-2}$)",

    fontsize=LABEL_SIZE

)

ax.set_ylabel(

    "Predicted QC$_{Median}$ (F cm$^{-2}$)",

    fontsize=LABEL_SIZE

)

ax.tick_params(

    axis="both",

    labelsize=TICK_SIZE,

    direction="out"

)

ax.tick_params(
    top=False,
    right=False
)

for spine in ax.spines.values():

    spine.set_linewidth(
        BORDER_WIDTH
    )

ax.grid(False)

plt.tight_layout()

figure_file = os.path.join(

    OUTPUT_DIR,

    "Phase5_QC_Median_Prediction.png"

)

plt.savefig(

    figure_file,

    dpi=600,

    bbox_inches="tight"

)

plt.show()


# ====================================================================
# 33. FINAL SUMMARY
# ====================================================================

print("\n" + "=" * 75)

print(
    "PHASE 5 COMPLETED"
)

print("=" * 75)

print(
    "\nQC_Median anchor Test R²:",
    f"{median_test_r2:.6f}"
)

print(
    "\nBest formulation:",
    best_formulation
)

print(
    "Best feature count:",
    best_features
)

print(
    "Complete curve Test R²:",
    f"{best_row['Curve_Test_R2']:.6f}"
)

print(
    "Complete curve Test MAE:",
    f"{best_row['Curve_Test_MAE']:.6e}"
)

print(
    "Complete curve Test RMSE:",
    f"{best_row['Curve_Test_RMSE']:.6e}"
)

print(
    "\nOutput directory:"
)

print(
    os.path.abspath(
        OUTPUT_DIR
    )
)

print("\n" + "=" * 75)
