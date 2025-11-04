"""
Streamlit Page: Premium Predictor
This file is designed to be imported as a function
by the main_dashboard.py script.
"""
import streamlit as st
import pandas as pd
import joblib
import statsmodels.api as sm
import numpy as np
from pathlib import Path

# Define paths
# We calculate the ROOT_DIR from this file's location
# (src/functions/premium_predictor.py)
ROOT_DIR = Path(__file__).resolve().parents[2]
PREPROCESSOR_PATH = ROOT_DIR / "models/premium_preprocessor.joblib"
MODEL_PATH = ROOT_DIR / "models/premium_model.pkl"

# These are the 14 features your BIC model selected 
# (Copied from your successful log)
FEATURES_LIST = [ 
    'n_doors_0', 'vehicle_age', 'value_vehicle', 'payment_0', 'type_risk_4', 
    'second_driver_0', 'driving_experience', 'type_fuel_0', 
    'policies_in_force', 'lapse', 'area_0', 'seniority', 
    'distribution_channel_0', 'type_risk_1'
]

# Model Loading
@st.cache_resource
def load_models():
    """
    Loads the preprocessor and the trained GLM model from disk.
    This is cached so it only runs once per session.
    """
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        # Load the statsmodels model
        model = sm.load_results(MODEL_PATH)
        return preprocessor, model
    except FileNotFoundError:
        st.error(f"Error: Model or preprocessor file not found. Please run 'src/models/train_premium_model.py' first to generate them.")
        return None, None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def premium_predictor():
    """
    This is the main function to be called by your main_dashboard.py
    It displays the full predictor page.
    """
    preprocessor, model = load_models()

    # Page Title
    st.title("💰 Premium Price Predictor")
    st.markdown("Enter your (or a client's) details below to get an estimated annual premium.")

    #User Input Form 
    if preprocessor and model:
        with st.form(key="prediction_form"):
            st.subheader("Vehicle Details")
            
            col1, col2 = st.columns(2)
            with col1:
                value_vehicle = st.number_input("Vehicle Value (€)", min_value=500, max_value=100000, value=20000, step=500)
                vehicle_age = st.slider("Vehicle Age (Years)", min_value=0, max_value=40, value=5)
                # Use format_func to show user-friendly names
                type_risk = st.selectbox("Vehicle Type", options=[1, 2, 3, 4], 
                                         format_func=lambda x: {1: "Motorbike", 2: "Van", 3: "Passenger Car", 4: "Agricultural"}.get(x))
            with col2:
                power = st.number_input("Power (hp)", min_value=40, max_value=500, value=110)
                type_fuel = st.selectbox("Fuel Type", options=[1, 2, 0], 
                                         format_func=lambda x: {1: "Petrol", 2: "Diesel", 0: "Other"}.get(x))
                n_doors = st.selectbox("Number of Doors", options=[0, 2, 3, 4, 5, 6], 
                                       format_func=lambda x: f"{x} doors" if x > 0 else "N/A (e.g., Motorbike)")

            st.subheader("Driver & Policy Details")
            col3, col4 = st.columns(2)
            with col3:
                driver_age = st.slider("Driver Age", min_value=18, max_value=90, value=35)
                driving_experience = st.slider("Driving Experience (Years)", min_value=0, max_value=72, value=15)
                area = st.radio("Area", options=[0, 1], 
                                format_func=lambda x: {0: "Rural", 1: "Urban"}.get(x))
            with col4:
                seniority = st.slider("Seniority with Insurer (Years)", min_value=0, max_value=50, value=3)
                # Note: 'lapse' was a feature in your final model
                lapse = st.radio("Has the policy lapsed before?", options=[0, 1], 
                                 format_func=lambda x: {0: "No", 1: "Yes"}.get(x))
                second_driver = st.radio("Is there a second driver?", options=[0, 1], 
                                         format_func=lambda x: {0: "No", 1: "Yes"}.get(x))

            
            policies_in_force = 1
            payment = 0 
            distribution_channel = 0 
            
            submit_button = st.form_submit_button(label="Predict Premium")

        #Prediction Logic
        if submit_button:
           
            input_data = pd.DataFrame({
                'seniority': [seniority],
                'policies_in_force': [policies_in_force],
                'max_policies': [policies_in_force], 
                'max_products': [1], 
                'lapse': [lapse],
                'power': [power],
                'cylinder_capacity': [1600], 
                'value_vehicle': [value_vehicle],
                'length': [4.5], 
                'weight': [1300], 
                'driver_age': [driver_age],
                'vehicle_age': [vehicle_age],
                'driving_experience': [driving_experience],
                
               
                'distribution_channel': [distribution_channel],
                'payment': [payment],
                'type_risk': [type_risk],
                'area': [area],
                'second_driver': [second_driver],
                'n_doors': [n_doors],
                'type_fuel': [type_fuel]
            })
            
            # Transforming the input data using the *loaded* preprocessor
            input_processed = preprocessor.transform(input_data)
            
            num_features = preprocessor.named_transformers_['num'].feature_names_in_
            cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out()
            all_feature_names = list(num_features) + list(cat_features)

            input_df = pd.DataFrame(input_processed, columns=all_feature_names)
            
            # Filtering for the 14 features the model *actually* uses
            final_model_features = []
            for f in FEATURES_LIST:
                if f in input_df.columns:
                    final_model_features.append(f)
                else:
                   
                    input_df[f] = 0
                    final_model_features.append(f)
                    
            input_final = input_df[final_model_features]
            
            # Adding the constant (intercept)
            input_final_with_const = sm.add_constant(input_final, prepend=True)

            # Making the prediction
            prediction_log = model.predict(input_final_with_const)
            
            # The model predicts the log of the premium, so we use np.exp()
            predicted_premium = np.exp(prediction_log.iloc[0])

            # Displaying the result
            st.success(f"## Estimated Annual Premium: €{predicted_premium:,.2f}")
            
            with st.expander("See model inputs (for debugging)"):
                st.write("Raw User Input:", input_data)
                st.write("Processed & Filtered Input (sent to model):", input_final_with_const)

    else:
        st.warning("Model files not found. Please contact the administrator to train the model.")