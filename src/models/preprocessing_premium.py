"""
Shared Preprocessing Script for Modeling
This file contains the function to prepare the data
for both Lasso and BIC models, ensuring identical
data is used for both.
"""
import pandas as pd
from datetime import datetime
import sys
import numpy as np

# Import scikit-learn components
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
except ImportError:
    print("Error in preprocessing.py: Could not import sklearn.")
    print("Please ensure scikit-learn is installed in your .venv")
    sys.exit(1)


def preprocess_data_for_modeling(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
    """
    Performs full data preprocessing for modeling.
    - Cleans data
    - Creates features (e.g., ages)
    - Handles missing values (imputation)
    - Encodes categorical variables
    - Scales numerical variables
    - Splits data into training and testing sets
    
    Returns: (X_train_final, X_test_final, y_train, y_test) on success, or (pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()) on failure.
    """
    try:
        print(f"Starting preprocessing. Original shape: {df.shape}")

        # --- 1. Feature Creation & Initial Filtering ---
        df_processed = df.copy()
        
        # Create Driver_Age
        if 'date_birth' in df_processed.columns:
            df_processed['driver_age'] = (datetime.now() - pd.to_datetime(df_processed['date_birth'], errors='coerce')).dt.days / 365.25
            df_processed = df_processed[(df_processed['driver_age'] >= 18) & (df_processed['driver_age'] <= 90)]
            print(f"Filtered by driver age. New shape: {df_processed.shape}")
        
        # Create Vehicle_Age
        if 'year_matriculation' in df_processed.columns:
            current_year = datetime.now().year
            df_processed['vehicle_age'] = current_year - pd.to_numeric(df_processed['year_matriculation'], errors='coerce')
            df_processed = df_processed[df_processed['vehicle_age'] >= 0]

        # Create Driving_Experience
        if 'date_driving_licence' in df_processed.columns:
            df_processed['driving_experience'] = (datetime.now() - pd.to_datetime(df_processed['date_driving_licence'], errors='coerce')).dt.days / 365.25
            df_processed = df_processed[df_processed['driving_experience'] >= 0]
        
        print(f"Shape after feature creation: {df_processed.shape}")

        # --- 2. Define Features (X) and Target (y) ---
        
        if target_column not in df_processed.columns:
            print(f"Error: Target column '{target_column}' not found.")
            return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
        y = df_processed[target_column]

        # --- THIS IS THE FIX ---
        # We will *keep* n_claims_history and r_claims_history as valid predictors.
        # We ONLY drop IDs, raw dates, and the *outcomes* of the policy year.
        cols_to_drop = [
            'id', 'date_start_contract', 'date_last_renewal', 
            'date_next_renewal', 'date_birth', 'date_driving_licence', 'date_lapse',
            'year_matriculation',
            
            # --- CRITICAL: Drop BOTH potential targets/leaks from X ---
            'premium', 
            'cost_claims_year',
            
            # Drop policy *outcomes* (not predictors)
            'n_claims_year',
            'lapse' # 'lapse' is an outcome of the year, not an input
        ]
        
        X = df_processed.drop(columns=[col for col in cols_to_drop if col in df_processed.columns], errors='ignore')
        
        # --- 3. Identify Feature Types ---
        categorical_features = [
            'distribution_channel', 'payment', 'type_risk', 
            'area', 'second_driver', 'n_doors', 'type_fuel'
        ]
        categorical_features = [col for col in categorical_features if col in X.columns]
        numerical_features = [col for col in X.columns if col not in categorical_features]

        print(f"Using {len(X.columns)} predictor columns:")
        print(f"  Numerical: {numerical_features}")
        print(f"  Categorical: {categorical_features}")

        # --- 4. Split Data ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # --- 5. Create Preprocessing Pipelines ---
        numerical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # --- 6. Combine Pipelines with ColumnTransformer ---
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, numerical_features),
                ('cat', categorical_pipeline, categorical_features)
            ],
            remainder='passthrough'
        )

        # --- 7. Apply Pipeline to Data ---
        print("Applying preprocessing pipeline (imputing, scaling, encoding)...")
        
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        try:
            ohe_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
        except Exception:
            ohe_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names(categorical_features)
            
        all_feature_names = numerical_features + list(ohe_feature_names)
        
        if X_train_processed.shape[1] != len(all_feature_names):
            raise ValueError(f"Shape mismatch: Processed data has {X_train_processed.shape[1]} columns, but we found {len(all_feature_names)} names.")

        X_train_final = pd.DataFrame(X_train_processed, columns=all_feature_names, index=X_train.index)
        X_test_final = pd.DataFrame(X_test_processed, columns=all_feature_names, index=X_test.index)
        
        print("✅ Preprocessing complete.")
        
        # Return all four sets of data
        return X_train_final, X_test_final, y_train, y_test

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()

