import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# --- VISUALIZATION FUNCTIONS ---

def plot_task_analysis(X, y, task_type):
    """
    Visualizes the dataset based on the task type.
    - Regression: Shows boxplots of numeric features (to see outliers).
    - Classification: Shows class balance of the target.
    """
    if X is None or y is None: 
        return None, "Please split your data in Tab 1 first."
    
    if not task_type: 
        task_type = "Regression"

    try:
        # Critical: We split first so we ONLY analyze the Training Set (avoid leakage)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        
        fig = plt.figure(figsize=(12, 8))
        msg = ""

        # --- REGRESSION LOGIC (Check Outliers) ---
        if task_type == "Regression":
            numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                return None, "No numeric columns to check for outliers."
            
            # Handle NaNs temporarily just for plotting
            X_temp = X_train[numeric_cols].copy()
            X_temp = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(X_temp), columns=numeric_cols)

            # Calculate Outliers (IQR Method) for reporting
            Q1 = X_temp.quantile(0.25)
            Q3 = X_temp.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((X_temp < (Q1 - 1.5 * IQR)) | (X_temp > (Q3 + 1.5 * IQR))).any(axis=1)
            num_outliers = outliers.sum()
            percent = (num_outliers / len(X_train)) * 100
            
            msg = (f"📉 **REGRESSION MODE DETECTED**\n"
                   f"Analyzed Training Data ({len(X_train)} rows).\n"
                   f"⚠️ Found **{num_outliers}** rows with outliers ({percent:.1f}%).\n"
                   f"✅ These will be **CAPPED** (Winsorized) in the Preprocessing step.")

            # Plot Boxplots
            num_cols = len(numeric_cols)
            grid_size = int(np.ceil(np.sqrt(num_cols)))
            
            # Handle single subplot edge case
            if num_cols == 1:
                ax = fig.gca()
                axes = [ax]
            else:
                axes = fig.subplots(grid_size, grid_size).flatten()

            for i, col in enumerate(numeric_cols):
                if i < len(axes):
                    sns.boxplot(x=X_temp[col], ax=axes[i], color="#ff7f0e")
                    axes[i].set_title(col)
            
            # Hide unused axes
            for j in range(len(numeric_cols), len(axes)): 
                axes[j].axis('off')

        # --- CLASSIFICATION LOGIC (Check Balance) ---
        else:
            # Check Class Balance
            # Handle y being DataFrame or Series
            if isinstance(y_train, pd.DataFrame):
                target_col = y_train.columns[0]
                y_series = y_train[target_col]
            else:
                y_series = y_train
                target_col = "Target"
                
            value_counts = y_series.value_counts()
            
            msg = (f"📊 **CLASSIFICATION MODE DETECTED**\n"
                   f"Analyzed Training Set ({len(y_train)} rows).\n\n")
            
            for label, count in value_counts.items():
                msg += f"• Class '{label}': {count} rows\n"
            
            msg += "\n🛡️ Outliers in X will be **KEPT** (RobustScaler will be used)."

            # Plot Countplot
            sns.countplot(x=y_series, palette="viridis")
            plt.title(f"Target Distribution: '{target_col}'")
            plt.xlabel("Class")
            plt.ylabel("Count")

        plt.tight_layout()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf, msg

    except Exception as e:
        return None, f"Error: {str(e)}"

def plot_all_distributions(X_train_df):
    """Plots histograms for all columns in the processed dataframe."""
    if X_train_df is None:
        return None
    
    try:
        # Limit columns to avoid crashing plotting with too many features
        cols = X_train_df.columns.tolist()
        if len(cols) > 20:
            cols = cols[:20] # Limit to first 20 for performance
            
        num_cols = len(cols)
        grid_size = int(np.ceil(np.sqrt(num_cols)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(cols):
            sns.histplot(X_train_df[col], kde=True, ax=axes[i])
            axes[i].set_title(col, fontsize=10)
        
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        
        return buf
    except Exception as e:
        print(f"Error plotting distributions: {e}")
        return None

def plot_correlation(X_train_df):
    """Plots a correlation heatmap for the processed dataframe."""
    if X_train_df is None:
        return None
        
    try:
        fig = plt.figure(figsize=(12, 10))
        
        # Calculate correlation matrix
        corr = X_train_df.corr(numeric_only=True) 
        
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", 
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title("Correlation Heatmap (Processed Data)")
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        return buf
    
    except Exception as e:
        print(f"Error plotting correlation: {e}")
        return None