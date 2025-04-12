import streamlit as st
import pandas as pd
import pickle
import joblib
from sklearn import metrics


model = pickle.load(open('model_pkl_svc_latest', 'rb'))

# App Title
st.title("Hello DOC 🩺!")

# Select mode
mode = st.radio("Choose Input Method", ["Single User", "CSV Upload"])

# Single User mode
if mode == "Single User":
    st.sidebar.header("Select Feature Flags")

    features = {
        "HighBP": st.sidebar.checkbox("HighBP"),
        "HighChol": st.sidebar.checkbox("HighChol"),
        "CholCheck": st.sidebar.checkbox("CholCheck"),
        "Smoker": st.sidebar.checkbox("Smoker"),
        "Stroke": st.sidebar.checkbox("Stroke"),
        "HeartDiseaseorAttack": st.sidebar.checkbox("HeartDiseaseorAttack"),
        "PhysActivity": st.sidebar.checkbox("PhysActivity"),
        "Fruits": st.sidebar.checkbox("Fruits"),
        "Veggies": st.sidebar.checkbox("Veggies"),
        "HvyAlcoholConsump": st.sidebar.checkbox("HvyAlcoholConsump"),
        "AnyHealthcare": st.sidebar.checkbox("AnyHealthcare"),
        "NoDocbcCost": st.sidebar.checkbox("NoDocbcCost"),
        "DiffWalk": st.sidebar.checkbox("DiffWalk"),
        "Sex": st.sidebar.checkbox("Sex") 
    }
    bmi_value = st.sidebar.slider("BMI (1-100)", min_value=12, max_value=30, value=12)
    GenHlth_value = st.sidebar.slider("GenHlth (1-5)", min_value=1, max_value=5, value=1)
    MentHlth_value = st.sidebar.slider("MentHlth (0-30)", min_value=0, max_value=30, value=0)
    PhysHlth_value = st.sidebar.slider("PhysHlth (0-30)", min_value=0, max_value=30, value=0)
    Age_value = st.sidebar.slider("Age-category (1-15)", min_value=1, max_value=15, value=5)
    Education_value = st.sidebar.slider("Education (1-6)", min_value=1, max_value=6, value=4)
    Income_value = st.sidebar.slider("Income (1-8)", min_value=1, max_value=8, value=3)
    
    input_df = pd.DataFrame([features])
    input_df["BMI"] = bmi_value
    input_df["GenHlth"] = GenHlth_value
    input_df["MentHlth"] = MentHlth_value
    input_df["PhysHlth"] = PhysHlth_value 
    input_df["Age"] = Age_value
    input_df["Education"] = Education_value
    input_df["Income"] = Income_value

    input_df = input_df[["HighBP", "HighChol","CholCheck","BMI","Smoker","Stroke","HeartDiseaseorAttack","PhysActivity","Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth","MentHlth","PhysHlth","DiffWalk","Sex","Age","Education","Income"]]

    st.subheader("Input Data")
    st.write(input_df)
    st.write('Click on know your diabetes button to get you diabetes status.')
    # Get model response
    if st.button("know your diabetes"):
        predictions = model.predict_proba(input_df)[:,1]

        st.subheader("Model Output")
        predictions = [0 if (y<0.5)else 1 for y in predictions]
        st.subheader("Your Diabetes status:")
        #print(predictions)
        if predictions[0]:
            st.write("oops! You have diabetes. pls consult your doctor.")
        else:
            st.write("Wow! You don't have any diabetes")
        #st.success(response[0])
        predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
        st.subheader("More details download the response:")
        st.download_button(
            label="Download Predictions",
            data=predictions_df.to_csv(index=False),
            file_name='predictions.csv',
            mime='text/csv'
        )

# CSV Upload mode
else:
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of the uploaded dataset:")
        st.write(df.head())
        st.write(f"Data shape: {df.shape[0]} rows and {df.shape[1]} columns")
        input_data = df.drop('Diabetes_binary', axis=1)  # Features (adjust 'target' if needed)
        y = df['Diabetes_binary'] 
        if st.button("know your diabetes"):
            try:
                #input_data = df.drop('Diabetes_binary', axis=1)  # Drop target column if present

                predictions = model.predict_proba(input_data)[:,1]
                #auc = metrics.roc_auc_score(y, predictions)
                predictions = [0 if (y<0.5)else 1 for y in predictions]
                st.subheader("Your Diabetes status:")
                #print(predictions)
                st.write("This is batch testing- Pls download the predicted status by clicking on below button")

            except Exception as e:
                st.error(f"Error making predictions: {e}")

            predictions_df = pd.DataFrame(predictions, columns=['Prediction'])

            st.subheader("More details download the response:")
            st.download_button(
                label="Download Predictions",
                data=predictions_df.to_csv(index=False),
                file_name='predictions.csv',
                mime='text/csv'
            )

