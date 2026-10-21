# QCML_DFT

## Machine Learning Prediction of Quantum Capacitance in Two-Dimensional Materials

This repository contains the processed dataset and Python code developed for machine-learning prediction of quantum capacitance in two-dimensional materials using independent material descriptors derived from structural, chemical, electronic, and magnetic properties.

## Repository Contents

- `data/`: Processed dataset used in the analysis.
- `code/`: Python scripts for model training, evaluation, and prediction.
- `requirements.txt`: Required Python packages.

## Methodology

The workflow includes data preprocessing, leakage-free feature selection, model development, cross-validation, and prediction of the median quantum capacitance. The extended workflow predicts potential-dependent quantum-capacitance curves using a median-anchored residual approach.

DOS-derived and QC-derived descriptors are excluded from the input features to prevent target leakage.

## Usage

1. Download or clone this repository.
2. Install the required Python packages listed in `requirements.txt`.
3. Follow the instructions provided with each Python script.
4. Use the appropriate processed input data and trained model files required by the prediction script.

## Data Source

The underlying materials data were obtained from the 2D Materials Properties Database (2DMPDIA). Please consult and cite the original database publication when using the source data.

## Citation

If you use this code or processed dataset, please cite the associated research article once it is published.
