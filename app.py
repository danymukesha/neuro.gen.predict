import streamlit as st
import pandas as pd
import pickle
from neurogenpredict import NeuroGenPredict

st.title("NeuroGenPredict: Alzheimer's Disease Genetic Risk Assessment Tool")

population = st.selectbox("Select Population", ["EUR", "AFR", "EAS", "AMR", "SAS"])

mode = st.radio("Mode", ["Train Model", "Predict Risk"])

if mode == "Train Model":
    st.subheader("Train the Model")
    genotype_file = st.file_uploader("Upload Genotype Data (CSV)", type="csv")
    labels_file = st.file_uploader("Upload Labels (CSV with 'label' column)", type="csv")
    clinical_file = st.file_uploader("Upload Clinical Data (CSV, optional)", type="csv")

    if st.button("Train"):
        if genotype_file and labels_file:
            genotype_data = pd.read_csv(genotype_file)
            y = pd.read_csv(labels_file)['label'].values
            clinical_data = pd.read_csv(clinical_file) if clinical_file else None

            predictor = NeuroGenPredict(population=population)
            X = predictor.prepare_features(genotype_data, clinical_data)
            cv_scores = predictor.train_ensemble(X, y)
            st.write("Cross-Validation Scores:", cv_scores)

            # Save model to bytes for download
            model_bytes = pickle.dumps(predictor)
            st.download_button("Download Trained Model", model_bytes, file_name="trained_model.pkl")
            st.success("Model trained successfully!")

else:
    st.subheader("Predict Risk")
    model_file = st.file_uploader("Upload Trained Model (PKL)", type="pkl")
    genotype_file = st.file_uploader("Upload Genotype Data for Prediction (CSV)", type="csv")
    clinical_file = st.file_uploader("Upload Clinical Data (CSV, optional)", type="csv")
    sample_id = st.text_input("Sample ID", "Sample001")

    if st.button("Predict"):
        if model_file and genotype_file:
            predictor = pickle.load(model_file)
            genotype_data = pd.read_csv(genotype_file)
            clinical_data = pd.read_csv(clinical_file) if clinical_file else None

            X = predictor.prepare_features(genotype_data, clinical_data)
            predictions = predictor.predict_risk(X)
            report = predictor.generate_report(predictions, sample_id, index=0)  # Assumes first sample; extend for multi if needed
            st.text_area("Risk Report", report, height=400)
