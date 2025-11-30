# 🚀 Auto-ML Studio

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-Machine%20Learning-green.svg)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue)

**Auto-ML Studio** is an end-to-end Machine Learning platform designed to bridge the gap between static notebooks and production-grade applications. Built as part of the **SAIR (Sudanese Artificial Intelligence Road)** learning path, it empowers users to go from raw data to a tuned, deployment-ready model without writing a single line of code.

---

## 🌟 Key Features

### 1. 📂 Intelligent Data Handling
* **Leakage-Proof Splitting:** Automatically splits data into **Train, Validation, and Test** sets immediately upon upload.
* **Safety First:** Detects and drops high-cardinality columns (IDs, Names) that cause memory crashes.

### 2. 🔍 Advanced EDA & Profiling
* **Instant Insights:** One-click checks for missing values and data types.
* **Deep Dive:** Integrated **YData Profiling** generates a comprehensive HTML report (correlations, warnings, distributions) directly in the UI.

### 3. ⚙️ Adaptive Preprocessing Pipeline
The system automatically configures itself based on your problem type:
* **📉 Regression Mode:**
    * Detects outliers using IQR.
    * **Removes** outlier rows to clean the training signal.
    * Applies **Standard Scaling**.
* **📊 Classification Mode:**
    * **Preserves** outlier rows (critical for rare classes like fraud).
    * Applies **Robust Scaling** to handle skew.
    * Automatically Label Encodes target variables.

### 4. 🤖 Automated Training & Leaderboard
* **Multi-Model Training:** Trains a diverse portfolio of **8+ algorithms** simultaneously:
    * *Linear/Logistic Regression, Random Forest, XGBoost, LightGBM, SVM, Gradient Boosting, etc.*
* **Live Leaderboard:** Ranks models by **Validation R²** (Regression) or **Accuracy** (Classification).
* **Visual Comparison:** Generates actual-vs-predicted scatter plots and confusion matrices for top performers.
* **MLflow Integration:** Every single run is logged with parameters and metrics for full reproducibility.

### 5. 🎛️ Optimization & Testing
* **Hyperparameter Tuning:** Pick the winner and run **RandomizedSearchCV** to find the optimal configuration.
* **Real-World Verification:** Validates the final tuned model against the held-out **Test Set** to ensure it generalizes well.

---


automl-studio/
├── app.py                  # 🚀 Main entry point (UI & Logic)
├── config.py               # ⚙️ Global Configuration
├── data_utils.py           # 🧹 Data Processing & Transformations
├── model_utils.py          # 🤖 ML Training, Tuning & Logging
├── plot_utils.py           # 📊 Visualization Logic
├── mlruns/                 # 🗂️ MLflow Artifact Store (Auto-generated)
└── requirements.txt        # 📦 Project Dependencies



File Name,Description
app.py,"The controller of the application. It builds the Gradio UI (Tabs, Buttons, Plots) and connects the user inputs to the backend logic imported from the utility files. Run this file to start the server."
config.py,"Contains the global Config class. It manages settings like random states (42), CPU jobs (-1), Cross-Validation folds (5), and directory paths to ensure consistency across all modules."
data_utils.py,"Handles the ""ETL"" (Extract, Transform, Load) process. It includes functions for loading CSVs, performing Train/Val/Test splits, and the main preprocess_data pipeline. It also contains the custom OutlierHandler class."
model_utils.py,"The machine learning engine. It defines the Model Portfolios (dictionaries of algorithms), Hyperparameter Grids, and contains the core functions for Training (train_models_pipeline) and Tuning (tune_models_pipeline), including MLflow logging logic."
plot_utils.py,"Dedicated to generating visualizations. It contains functions like plot_task_analysis (Boxplots/Countplots), plot_correlation, and plot_all_distributions that return Matplotlib figures to the UI."
mlruns/,"A directory automatically created by MLflow to store experiment logs, metrics, and saved models (.pkl files)."

## 🛠️ Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/automl-studio.git](https://github.com/abdelhadi-osama/automl-studio.git)
cd automl-studio

# Create a virtual environment (Recommended: Python 3.12)
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv add gradio pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm mlflow "ydata-profiling" "numpy<2.0" "scipy<1.14"
