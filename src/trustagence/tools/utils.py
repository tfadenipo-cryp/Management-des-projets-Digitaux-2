import streamlit as st


def format_result_display(title: str, results: dict):
    html = f"<h3 style='color:#0a84ff;'>{title}</h3>"
    for k, v in results.items():
        html += f"<p><b>{k}:</b> {v}</p>"
    st.markdown(html, unsafe_allow_html=True)


def reright_of(variable):
    mapping = {
        "id": "Insured ID",
        "date_start_contract": "Contract start date",
        "date_last_renewal": "Last contract renewal date",
        "date_next_renewal": "Next expected renewal date",
        "date_birth": "Insured's birth date",
        "date_driving_licence": "Driving licence issue date",
        "distribution_channel": "Distribution channel (e.g., agency, broker, online)",
        "seniority": "Client seniority (in years)",
        "policies_in_force": "Number of active policies for this client",
        "max_policies": "Maximum number of policies held by the client",
        "max_products": "Maximum number of products subscribed by the client",
        "lapse": "Contract lapse indicator (1 = cancelled, 0 = active)",
        "date_lapse": "Contract lapse or expiration date",
        "payment": "Payment frequency (monthly, yearly, etc.)",
        "premium": "Premium amount paid for the contract",
        "cost_claims_year": "Total cost of claims during the year",
        "n_claims_year": "Number of claims declared during the year",
        "n_claims_history": "Total number of claims in client's history",
        "r_claims_history": "Ratio or total amount of historical claims",
        "type_risk": "Risk type (insurance category)",
        "area": "Geographical residence area of the insured",
        "second_driver": "Presence of a second driver (1 = yes, 0 = no)",
        "year_matriculation": "Vehicle first registration year",
        "power": "Vehicle power (in HP or kW)",
        "cylinder_capacity": "Engine displacement (in cm³)",
        "value_vehicle": "Insured vehicle value (in euros)",
        "n_doors": "Number of vehicle doors",
        "type_fuel": "Fuel type (gasoline, diesel, etc.)",
        "length": "Vehicle length (in cm)",
        "weight": "Vehicle weight (in kg)",
    }
    return mapping.get(variable, variable)


def get_name(variable):
    mapping = {
        "Insured ID": "id",
        "Contract start date": "date_start_contract",
        "Last contract renewal date": "date_last_renewal",
        "Next expected renewal date": "date_next_renewal",
        "Insured's birth date": "date_birth",
        "Driving licence issue date": "date_driving_licence",
        "Distribution channel (e.g., agency, broker, online)": "distribution_channel",
        "Client seniority (in years)": "seniority",
        "Number of active policies for this client": "policies_in_force",
        "Maximum number of policies held by the client": "max_policies",
        "Maximum number of products subscribed by the client": "max_products",
        "Contract lapse indicator (1 = cancelled, 0 = active)": "lapse",
        "Contract lapse or expiration date": "date_lapse",
        "Payment frequency (monthly, yearly, etc.)": "payment",
        "Premium amount paid for the contract": "premium",
        "Total cost of claims during the year": "cost_claims_year",
        "Number of claims declared during the year": "n_claims_year",
        "Total number of claims in client's history": "n_claims_history",
        "Ratio or total amount of historical claims": "r_claims_history",
        "Risk type (insurance category)": "type_risk",
        "Geographical residence area of the insured": "area",
        "Presence of a second driver (1 = yes, 0 = no)": "second_driver",
        "Vehicle first registration year": "year_matriculation",
        "Vehicle power (in HP or kW)": "power",
        "Engine displacement (in cm³)": "cylinder_capacity",
        "Insured vehicle value (in euros)": "value_vehicle",
        "Number of vehicle doors": "n_doors",
        "Fuel type (gasoline, diesel, etc.)": "type_fuel",
        "Vehicle length (in cm)": "length",
        "Vehicle weight (in kg)": "weight",
    }

    return mapping.get(variable, variable)
