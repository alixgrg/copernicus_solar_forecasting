# Copernicus Solar Forecasting

## Project Overview

This project focuses on **short-term solar irradiance forecasting** using satellite image sequences from the Copernicus program.

The goal is to predict the **Global Horizontal Irradiance (GHI)** for the next **15 to 60 minutes** based on:

* past satellite observations,
* clear-sky irradiance models,
* solar position information.

This is a **spatio-temporal regression problem** involving sequences of images.

---

## Objectives

* Predict future GHI images (up to +1 hour)
* Compare multiple machine learning approaches:

  * Baseline models (persistence, linear regression)
  * Supervised models (Random Forest, SVM, etc.)
  * Unsupervised learning (clustering)
  * Deep learning (CNN / ConvLSTM - optional)
* Evaluate performance using:

  * RMSE
  * MAE

---

## Data Description

Each sample contains:

### Inputs:

* **GHI (81×81×4)** → past irradiance images
* **GHIcls (81×81×8)** → clear-sky irradiance (past + future)
* **SZA (81×81×8)** → solar zenith angle
* **SAA (81×81×8)** → solar azimuth angle

### Output:

* **GHI (51×51×4)** → future irradiance (next 15, 30, 45, 60 min)

---

## Data Access

The dataset is not included in this repository due to its size.

Download it here:
**[INSERT GOOGLE DRIVE LINK]**

Then place the files in:

```
data/raw/
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/copernicus-solar-forecasting.git
cd copernicus-solar-forecasting
```

### 2. Create the environment (Conda)

```bash
conda env create -f environment.yml
conda activate copernicus-solar
```

### 3. Launch Jupyter

```bash
jupyter lab
```

---

## Project Structure

```
.
├── data/
│   ├── raw/            # raw data (not versioned)
│   └── processed/      # cleaned data
├── notebooks/          # experiments & EDA
├── src/                # reusable code
├── models/             # saved models (ignored)
├── reports/            # figures & results
├── environment.yml     # conda environment
└── README.md
```

---

## Methodology

The project follows a structured machine learning pipeline:

1. Data exploration and preprocessing
2. Feature engineering
3. Baseline modeling
4. Advanced supervised models
5. Unsupervised analysis (clustering)
6. Model interpretation (SHAP, feature importance)
7. Optional deep learning approaches

---

## Evaluation Metrics

Models are evaluated using:

* **RMSE (Root Mean Squared Error)**
* **MAE (Mean Absolute Error)**

Evaluation is performed:

* globally
* for each forecast horizon (15, 30, 45, 60 min)

---

## Contributors

* Alix GREGGIO

---

## Notes

* Large data files are not tracked by Git
* Use the provided download link
* Ensure correct folder structure before running notebooks

---

