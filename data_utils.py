import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from ydata_profiling import ProfileReport

# --- CUSTOM CLASSES ---
class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Custom Transformer to Cap (Winsorize) Outliers using IQR.
    Compatible with Scikit-Learn Pipelines.
    """
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bounds_ = []
        self.upper_bounds_ = []
    
    def fit(self, X, y=None):
        self.lower_bounds_ = []
        self.upper_bounds_ = []
        
        # Ensure input is a numpy array
        if isinstance(X, pd.DataFrame): 
            X = X.values
            
        for i in range(X.shape[1]):
            col_data = X[:, i]
            # Ignore NaNs for calculation
            col_data = col_data[~np.isnan(col_data)]
            
            if len(col_data) == 0:
                self.lower_bounds_.append(-np.inf)
                self.upper_bounds_.append(np.inf)
                continue

            Q1 = np.percentile(col_data, 25)
            Q3 = np.percentile(col_data, 75)
            IQR = Q3 - Q1
            
            self.lower_bounds_.append(Q1 - self.factor * IQR)
            self.upper_bounds_.append(Q3 + self.factor * IQR)
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame): 
            X_data = X.values.copy()
        else: 
            X_data = X.copy()
            
        for i in range(X_data.shape[1]):
            X_data[:, i] = np.clip(X_data[:, i], self.lower_bounds_[i], self.upper_bounds_[i])
        return X_data

# --- HELPER FUNCTIONS ---
def update_task_type(new_value):
    """Updates the task type state based on radio button selection."""
    return new_value

# --- DATA LOADING & EDA ---
def load_and_store_data(file):
    """Loads CSV file and returns it for display and state."""
    if file is None: return None, None
    try:
        df = pd.read_csv(file.name)
        return df, df 
    except Exception as e: return None, None 

def find_missing_values(df):
    if df is None: return None
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty: return pd.DataFrame(columns=["Column", "Missing Count"])
    return missing.to_frame(name="Missing Count").reset_index().rename(columns={'index': 'Column'})

def find_categorical_columns(df):
    if df is None: return None
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) == 0: return pd.DataFrame(columns=["Categorical Columns"])
    return pd.DataFrame(cat_cols, columns=["Categorical Columns"])

def get_shape(df):
    if df is None: return "No data loaded."
    return f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns."

def generate_profile_report(df):
    if df is None: return "Please upload a file first."
    try:
        # minimal=True is critical for performance in web apps
        profile = ProfileReport(df, title="Dataset Profiling Report", minimal=False)
        return profile.to_html()
    except Exception as e: return f"Error: {str(e)}"

# --- SPLITTING & PREPROCESSING ---
def split_and_save_data(df, y_column_name):
    """
    Splits data into X and y, and then into Train/Val/Test sets.
    Also drops high-cardinality columns to prevent memory crashes.
    """
    if df is None: return "Please upload a file first.", *([None]*8)
    if y_column_name not in df.columns: return f"Error: '{y_column_name}' not found.", *([None]*8)
    
    # 1. Drop High Cardinality Columns (Safety Check)
    # Columns with >50 unique values are likely IDs or Names, which break OneHotEncoder
    potential_drops = [col for col in df.select_dtypes(include=['object']).columns if df[col].nunique() > 50]
    if y_column_name in potential_drops: potential_drops.remove(y_column_name)
    df_cleaned = df.drop(columns=potential_drops)
    
    # 2. Handle Target
    df_cleaned = df_cleaned.dropna(subset=[y_column_name])
    y = df_cleaned[[y_column_name]]
    X = df_cleaned.drop(columns=[y_column_name])
    
    # 3. Split (60% Train, 20% Val, 20% Test)
    # First split: 80% (Train+Val) / 20% (Test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Second split: 75% (Train) / 25% (Val) -> Results in 60% total for Train
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
    
    msg = f"Success! Split Complete.\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}"
    if potential_drops:
        msg += f"\n(Dropped high-cardinality columns: {potential_drops})"
        
    return msg, X_train.head(), y_train.head(), X_train, y_train, X_val, y_val, X_test, y_test

def preprocess_data(X_train, y_train, X_val, y_val, X_test, y_test, task_type):
    """
    The Main Pipeline: Imputation -> Outlier Handling -> Scaling -> Encoding.
    """
    if X_train is None: return "Please split data first.", *([None]*8)

    try:
        # 1. Define Feature Groups
        numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

        status_msg = ""

        # 2. Build Numeric Pipeline based on Task
        if task_type == "Regression":
            # Regression: Impute -> Cap Outliers -> Robust Scale
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('outliers', OutlierHandler(factor=1.5)), 
                ('scaler', RobustScaler()) 
            ])
            status_msg = "Regression Pipeline: Outliers Capped + RobustScaler"
        else:
            # Classification: Impute -> Robust Scale (No capping)
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ])
            
            # Handle Classification Target (Label Encoding)
            le = LabelEncoder()
            # We need to flatten y to 1D array for LabelEncoder
            y_col = y_train.columns[0]
            y_train = y_train.copy() # Avoid SettingWithCopy warning
            y_val = y_val.copy()
            y_test = y_test.copy()
            
            y_train[y_col] = le.fit_transform(y_train[y_col])
            y_val[y_col] = le.transform(y_val[y_col])
            y_test[y_col] = le.transform(y_test[y_col])
            status_msg = "Classification Pipeline: RobustScaler + LabelEncoder"

        # 3. Build Categorical Pipeline
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        # 4. Combine into Processor
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

        # 5. Fit and Transform
        preprocessor.fit(X_train)
        new_cols = preprocessor.get_feature_names_out()
        
        # Helper to convert sparse/dense arrays to DataFrame
        def to_df(data, cols):
            if hasattr(data, "toarray"): data = data.toarray()
            return pd.DataFrame(data, columns=cols)

        X_train_df = to_df(preprocessor.transform(X_train), new_cols)
        X_val_df = to_df(preprocessor.transform(X_val), new_cols)
        X_test_df = to_df(preprocessor.transform(X_test), new_cols)
        
        status = f"✅ Complete!\n{status_msg}\nFinal Train Shape: {X_train_df.shape}"
        
        return (status, X_train_df.head(), 
                X_train_df, y_train, X_val_df, y_val, X_test_df, y_test, 
                X_train_df) # Return Clean DF for plotting

    except Exception as e: return f"Error: {str(e)}", *([None]*8)