import streamlit as st
import pandas as pd
import pickle
import joblib
from sklearn import metrics
# model = joblib.load('model_svc.joblib')

# Load model
model = pickle.load(open('model_pkl_svc_latest', 'rb'))

# Title
st.title("🧠 ML Classifier Demo")

def user_input_features():

    #st.write("Upload your input file here")
    uploaded_file = st.file_uploader("Choose a CSV file only", type=["csv"])
    st.markdown("""
    <style>
        .stButton button {
            background-color: #4CAF50; /* Green */
            color: white;
            font-size: 16px;
            border-radius: 5px;
            height: 40px;
            width: 200px;
        }
        .stButton button:hover {
            background-color: #45a049; /* Darker green on hover */
        }
    </style>
    """, unsafe_allow_html=True)
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of the uploaded dataset:")
        st.write(df.head())
        st.write(f"Data shape: {df.shape[0]} rows and {df.shape[1]} columns")
        input_data = df.drop('Diabetes_binary', axis=1)  # Features (adjust 'target' if needed)
        y = df['Diabetes_binary'] 
        if st.button("Make Predictions"):
            try:
                #input_data = df.drop('Diabetes_binary', axis=1)  # Drop target column if present

                predictions = model.predict_proba(input_data)[:,1]
                predictions = [0 if (y<0.5)else 1 for y in predictions]
                st.subheader("Predictions:")
                st.write(predictions)

            except Exception as e:
                st.error(f"Error making predictions: {e}")

            predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
            
            auc = metrics.roc_auc_score(y,predictions)

            # Calculate AUC (only applicable for binary classification)
            accuracy = metrics.accuracy_score(y,predictions)

            # Display results
            st.subheader(f"Accuracy: {accuracy:.4f}")
            st.subheader(f"AUC: {auc:.4f}")
            # Download the predictions
            st.download_button(
                label="Download Predictions",
                data=predictions_df.to_csv(index=False),
                file_name='predictions.csv',
                mime='text/csv'
            )

    else:
        st.info("Please upload a CSV file to make predictions.")

user_input_features()



# Optional: show probabilities, charts, SHAP, etc.
