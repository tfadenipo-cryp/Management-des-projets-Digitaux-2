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
    Returns: (X_train, X_test, y_train, y_test, preprocessor_object, final_feature_names)
    """
    try:
        print(f"Starting preprocessing. Original shape: {df.shape}")

        # Initial Filtering 
        df_processed = df.copy()
        
        # Creating Driver_Age
        if 'date_birth' in df_processed.columns:
            df_processed['driver_age'] = (datetime.now() - pd.to_datetime(df_processed['date_birth'], errors='coerce')).dt.days / 365.25
            df_processed = df_processed[(df_processed['driver_age'] >= 18) & (df_processed['driver_age'] <= 90)]
        
        # Creating Vehicle_Age
        if 'year_matriculation' in df_processed.columns:
            current_year = datetime.now().year
            df_processed['vehicle_age'] = current_year - pd.to_numeric(df_processed['year_matriculation'], errors='coerce')
            df_processed = df_processed[df_processed['vehicle_age'] >= 0]

        # Creating Driving_Experience
        if 'date_driving_licence' in df_processed.columns:
            df_processed['driving_experience'] = (datetime.now() - pd.to_datetime(df_processed['date_driving_licence'], errors='coerce')).dt.days / 365.25
            df_processed = df_processed[df_processed['driving_experience'] >= 0]
        
        print(f"Shape after feature creation: {df_processed.shape}")


        if target_column not in df_processed.columns:
            print(f"Error: Target column '{target_column}' not found.")
            return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series(), None, []
        y = df_processed[target_column]

        cols_to_drop = [
            'id', 'date_start_contract', 'date_last_renewal', 
            'date_next_renewal', 'date_birth', 'date_driving_licence', 'date_lapse',
            'year_matriculation', 'premium', 'n_claims_year', 
            'cost_claims_year', 'n_claims_history', 'r_claims_history', 'lapse'
        ]
        
        X = df_processed.drop(columns=[col for col in cols_to_drop if col in df_processed.columns], errors='ignore')
        
        categorical_features = [
            'distribution_channel', 'payment', 'type_risk', 
            'area', 'second_driver', 'n_doors', 'type_fuel'
        ]
        categorical_features = [col for col in categorical_features if col in X.columns]
        numerical_features = [col for col in X.columns if col not in categorical_features]

        print(f"Using {len(X.columns)} predictor columns.")

        # Splitting the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )


        numerical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, numerical_features),
                ('cat', categorical_pipeline, categorical_features)
            ],
            remainder='passthrough'
        )

        print("Applying preprocessing pipeline (imputing, scaling, encoding)...")
        
        # only the training data
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        try:
            ohe_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
        except Exception:
            ohe_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names(categorical_features)
            
        all_feature_names = numerical_features + list(ohe_feature_names)
        
        if X_train_processed.shape[1] != len(all_feature_names):
            raise ValueError("Shape mismatch after preprocessing.")

        X_train_final = pd.DataFrame(X_train_processed, columns=all_feature_names, index=X_train.index)
        X_test_final = pd.DataFrame(X_test_processed, columns=all_feature_names, index=X_test.index)
        
        print("✅ Preprocessing complete.")
        
      
        return X_train_final, X_test_final, y_train, y_test, preprocessor, all_feature_names

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series(), None, []

