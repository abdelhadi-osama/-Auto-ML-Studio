import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import mlflow
import mlflow.sklearn
import gradio as gr
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, f1_score, 
    precision_score, recall_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from config import cfg

# --- MODEL PORTFOLIOS ---
regression_models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(random_state=cfg.RANDOM_STATE),
    'Lasso': Lasso(random_state=cfg.RANDOM_STATE),
    'ElasticNet': ElasticNet(random_state=cfg.RANDOM_STATE),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=cfg.RANDOM_STATE, n_jobs=cfg.N_JOBS),
    'Gradient Boosting': GradientBoostingRegressor(random_state=cfg.RANDOM_STATE),
    'SVR': SVR()
}

classification_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=cfg.RANDOM_STATE),
    'Decision Tree': DecisionTreeClassifier(random_state=cfg.RANDOM_STATE),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=cfg.RANDOM_STATE, n_jobs=cfg.N_JOBS),
    'Gradient Boosting': GradientBoostingClassifier(random_state=cfg.RANDOM_STATE),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=cfg.RANDOM_STATE, n_jobs=cfg.N_JOBS, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(n_estimators=100, random_state=cfg.RANDOM_STATE, n_jobs=cfg.N_JOBS, verbose=-1),
    'SVC': SVC(probability=True, random_state=cfg.RANDOM_STATE),
    'KNN': KNeighborsClassifier(n_jobs=cfg.N_JOBS)
}

# --- HYPERPARAMETER GRIDS ---
param_grids_regression = {
    'Linear Regression': {},
    'Ridge': {'alpha': [0.01, 0.1, 1.0, 10.0]},
    'Lasso': {'alpha': [0.001, 0.01, 0.1, 1.0]},
    'ElasticNet': {'alpha': [0.01, 0.1, 1.0], 'l1_ratio': [0.2, 0.5, 0.8]},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
    'Gradient Boosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},
    'SVR': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
}

param_grids_classification = {
    'Logistic Regression': [{'penalty': ['l2'], 'C': [0.1, 1.0, 10.0]}],
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
    'Gradient Boosting': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]},
    'XGBoost': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]},
    'LightGBM': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'num_leaves': [31, 63]},
    'SVC': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'KNN': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
}

# --- HELPER FUNCTIONS ---
def update_model_choices(task_type):
    if task_type == "Regression":
        return gr.Dropdown.update(choices=list(param_grids_regression.keys()), value="Random Forest")
    return gr.Dropdown.update(choices=list(param_grids_classification.keys()), value="Random Forest")

# --- CORE TRAINING WORKER ---
def evaluate_model_advanced(model, X_train, y_train, X_val, y_val, model_name, task_type):
    try:
        start = time.time()
        model.fit(X_train, y_train)
        duration = time.time() - start
        preds = model.predict(X_val)
        
        metrics = {"Model": model_name, "Time (s)": round(duration, 2)}
        
        if task_type == "Regression":
            score = r2_score(y_val, preds)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            metrics.update({"Score": round(score, 4), "RMSE": round(rmse, 4)})
            cv_metric = 'r2'
        else:
            score = accuracy_score(y_val, preds)
            metrics.update({"Score": round(score, 4), "F1": round(f1_score(y_val, preds, average='weighted'), 4)})
            cv_metric = 'accuracy'
            
        # Log to MLflow
        with mlflow.start_run(run_name=f"Train_{model_name}", nested=True):
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            
        # Quick CV Check (3 folds for speed)
        cv = cross_val_score(model, X_train, y_train, cv=3, scoring=cv_metric, n_jobs=cfg.N_JOBS)
        metrics["CV Mean"] = round(cv.mean(), 4)
        
        return metrics, model, score

    except Exception as e:
        print(f"Skip {model_name}: {e}")
        return None, None, -np.inf

# --- PIPELINE MANAGER: TRAIN ALL ---
def train_models_pipeline(X_train, y_train, X_val, y_val, X_test, y_test, task_type):
    if X_train is None: return None, None, None, "Please run Preprocessing first."
    
    # Flatten targets for sklearn
    y_train = np.ravel(y_train); y_val = np.ravel(y_val); y_test = np.ravel(y_test)
    
    models = regression_models if task_type == "Regression" else classification_models
    metric_name = "R²" if task_type == "Regression" else "Accuracy"
    
    results = []
    best_score = -np.inf
    best_model = None
    best_name = ""
    
    # Loop through portfolio
    for name, model in models.items():
        met, trained_model, score = evaluate_model_advanced(
            model, X_train, y_train, X_val, y_val, name, task_type
        )
        if met:
            results.append(met)
            if score > best_score:
                best_score = score
                best_model = trained_model
                best_name = name
    
    if not results: return None, None, None, "Error: No models trained."
    
    # Create Leaderboard
    leaderboard = pd.DataFrame(results).sort_values(by="Score", ascending=False)
    
    # Create Comparison Chart
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(data=leaderboard, x="Score", y="Model", palette="viridis")
    plt.title(f"Model Comparison: {metric_name}")
    plt.tight_layout()
    if task_type == "Classification": plt.xlim(0, 1)
    
    # Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    # Final Test on Best Model
    test_pred = best_model.predict(X_test)
    if task_type == "Regression": 
        final_sc = r2_score(y_test, test_pred)
    else: 
        final_sc = accuracy_score(y_test, test_pred)
    
    joblib.dump(best_model, "best_model.pkl")
    status = f"✅ Done!\n🏆 Best: {best_name}\n🚀 Test Set {metric_name}: {final_sc:.4f}"
    
    return leaderboard, buf, "best_model.pkl", status

# --- PIPELINE MANAGER: TUNE ONE ---
def tune_models_pipeline(X_train, y_train, X_val, y_val, X_test, y_test, model_name, task_type):
    if X_train is None: return None, None, None, "Preprocessing needed."
    y_train = np.ravel(y_train); y_test = np.ravel(y_test)
    
    if task_type == "Regression":
        base = regression_models.get(model_name)
        grid = param_grids_regression.get(model_name)
        metric = 'r2'
    else:
        base = classification_models.get(model_name)
        grid = param_grids_classification.get(model_name)
        metric = 'accuracy'
        
    if not base or not grid: return None, None, None, "Model or Grid not found."
    
    try:
        with mlflow.start_run(run_name=f"Tune_{model_name}"):
            search = RandomizedSearchCV(
                base, grid, n_iter=10, cv=3, scoring=metric, 
                n_jobs=cfg.N_JOBS, random_state=cfg.RANDOM_STATE
            )
            search.fit(X_train, y_train)
            
            best = search.best_estimator_
            test_pred = best.predict(X_test)
            
            # Plot Result
            fig, ax = plt.subplots(figsize=(6, 6))
            if task_type == "Regression":
                score = r2_score(y_test, test_pred)
                sns.scatterplot(x=y_test, y=test_pred, alpha=0.5, ax=ax)
                min_v, max_v = y_test.min(), y_test.max()
                ax.plot([min_v, max_v], [min_v, max_v], 'r--')
                ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
            else:
                score = accuracy_score(y_test, test_pred)
                cm = confusion_matrix(y_test, test_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax)
                ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            
            joblib.dump(best, "tuned_model.pkl")
            
            return f"✅ Tuned!\nBest {metric}: {score:.4f}\nParams: {search.best_params_}", buf, "tuned_model.pkl", "Done"
            
    except Exception as e: return str(e), None, None, "Error"