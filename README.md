# Malaria Disease Prediction Using Machine Learning

## Project Overview

This project demonstrates a binary classification workflow using a Random Forest Classifier and a Streamlit web interface.

Users can enter sample features such as age, symptoms, hemoglobin, and platelet count to obtain a predicted dataset class.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Random Forest Classifier
- Streamlit
- Joblib

## Dataset

The project uses 1,000 synthetically generated records with nine input features and one target column, `Malaria_Positive`.

**Important:** The dataset labels are artificially generated and are not verified malaria diagnoses. The model is for educational demonstration only and must not be used for medical diagnosis.

## Model Development

1. Load and inspect the dataset.
2. Prepare the input features and target.
3. Split the data into training and testing sets.
4. Train a Random Forest Classifier.
5. Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
6. Build a Streamlit interface for interactive predictions.

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

Keep `malaria_random_forest_model.pkl` and `malaria_feature_names.pkl` in the same directory as `app.py`.

## Limitations

The synthetic labels do not represent real clinical outcomes. Evaluation metrics describe performance on this artificial dataset and do not establish real-world malaria detection performance.
