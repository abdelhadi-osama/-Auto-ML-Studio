import os

class Config:
    """
    Global Configuration for the AutoML App.
    """
    # Reproducibility
    RANDOM_STATE = 42
    
    # Data Splitting
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    
    # Training Settings
    CV_FOLDS = 5
    N_JOBS = -1  # Use all available CPU cores
    
    # Paths
    MODEL_DIR = "models"
    EXPERIMENT_DIR = "experiments"
    
    # Create directories immediately
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

# Create a global instance to be imported by other files
cfg = Config()