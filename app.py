import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your trained model
model = joblib.load('model/model.pkl')  # Adjust the path as needed

def user_input_features():
    with st.form("user_input_form"):
        st.header('Customer Details')
        st.caption('Input the details of the Customer to predict their risk of Churning ')
        
        # Layout improvements using columns to make the form more compact and organized
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox('Gender', ['Male', 'Female'])
            senior_citizen = st.selectbox('Senior Citizen', [0, 1])
            partner = st.selectbox('Partner', ['Yes', 'No'])
            dependents = st.selectbox('Dependents', ['Yes', 'No'])
            
        with col2:
            phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
            multiple_lines = st.selectbox('Multiple Lines', ['Phone service', 'No', 'Yes'])
            internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
            online_security = st.selectbox('Online Security', ['Online Security', 'No', 'Yes'])
            
        with col3:
            online_backup = st.selectbox('Online Backup', ['Online Backup', 'No', 'Yes'])
            device_protection = st.selectbox('Device Protection', ['Device Protection', 'No', 'Yes'])
            tech_support = st.selectbox('Tech Support', ['Tech Support', 'No', 'Yes'])
            streaming_tv = st.selectbox('Streaming TV', ['Streaming TV', 'No', 'Yes'])
            streaming_movies = st.selectbox('Streaming Movies', ['Streaming Movies', 'No', 'Yes'])
        
        col4, col5 = st.columns([3,2])
        with col4:
            contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
            paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
            payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
        
        with col5:
            tenure = st.slider('Tenure', 0, 72, 1)
            monthly_charges = st.number_input('Monthly Charges', min_value=0.0, max_value=150.0, value=70.0, step=0.01)
            total_charges = st.number_input('Total Charges', min_value=0.0, max_value=10000.0, value=100.0, step=0.01)
        
        submitted = st.form_submit_button("Predict")
        if submitted:
            data = {'gender': gender, 'SeniorCitizen': senior_citizen, 'Partner': partner, 'Dependents': dependents,
                    'tenure': tenure, 'PhoneService': phone_service, 'MultipleLines': multiple_lines,
                    'InternetService': internet_service, 'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
                    'DeviceProtection': device_protection, 'TechSupport': tech_support, 'StreamingTV': streaming_tv,
                    'StreamingMovies': streaming_movies, 'Contract': contract, 'PaperlessBilling': paperless_billing,
                    'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges}
            features = pd.DataFrame(data, index=[0])
            return features
    return None

def main():
    st.title('Customer Churn Prediction App 🚀')
    
    input_df = user_input_features()
    
    if input_df is not None:
        # Note: Ensure you preprocess your input_df as required before prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        # Enhanced presentation of the prediction results
        st.subheader('Prediction Results 📊')
        churn_probability = np.round(prediction_proba[0][1], 2)
        st.metric(label="Churn Probability", value=f"{churn_probability * 100} %")
        result_text = 'Churn' if prediction[0] == 1 else 'No Churn'
        st.success(f'Prediction: **{result_text}** 🎯')
        st.caption("Churn Probability indicates the likelihood that the customer will leave the service. A higher value suggests a greater risk of churn.")

if __name__ == '__main__':
    main()







