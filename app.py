import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, request, render_template
import os
import joblib

app = Flask(__name__)

# Paths for saving model and encoders
MODEL_PATH = 'xgboost_model.pkl'
GENDER_ENCODER_PATH = 'gender_encoder.pkl'
CANCER_ENCODER_PATH = 'cancer_encoder.pkl'

# Load and preprocess the dataset
def load_and_preprocess_data():
    try:
        data = pd.read_csv("lung_cancer.csv")
    except FileNotFoundError:
        print("Error: lung_cancer.csv not found. Please ensure the dataset is in the same directory.")
        return None, None, None, None

    # Validate dataset columns
    required_columns = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
                       'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING',
                       'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN', 'LUNG_CANCER']
    if not all(col in data.columns for col in required_columns):
        print("Error: Dataset missing required columns.")
        return None, None, None, None

    # Display dataset info
    print("Dataset Info:")
    print(data.info())
    print("\nDataset Summary Statistics:")
    print(data.describe())
    print("\nFirst 5 rows of the dataset:")
    print(data.head())
    print("\nLast 5 rows of the dataset:")
    print(data.tail())
    print(f"\nDataset Shape: {data.shape}")

    # Preprocess the data
    le_gender = LabelEncoder()
    le_cancer = LabelEncoder()
    data['GENDER'] = le_gender.fit_transform(data['GENDER'])
    data['LUNG_CANCER'] = le_cancer.fit_transform(data['LUNG_CANCER'])

    # Save encoders
    joblib.dump(le_gender, GENDER_ENCODER_PATH)
    joblib.dump(le_cancer, CANCER_ENCODER_PATH)

    # Define features and target
    X = data.drop('LUNG_CANCER', axis=1)
    y = data['LUNG_CANCER']

    return X, y, le_gender, le_cancer

# Train and save the XGBoost model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train XGBoost model
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\nXGBoost Model Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    feature_importance = model.feature_importances_
    feature_names = X.columns
    plt.bar(feature_names, feature_importance, color='#4CAF50')
    plt.xticks(rotation=45)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.title('Feature Importance in XGBoost Model', fontsize=14)
    plt.tight_layout()
    plt.savefig('static/feature_importance.png')
    plt.close()

    return model, X_test, y_test

# Load the saved model and encoders
def load_model_and_encoders():
    try:
        model = joblib.load(MODEL_PATH)
        le_gender = joblib.load(GENDER_ENCODER_PATH)
        le_cancer = joblib.load(CANCER_ENCODER_PATH)
        print(f"Loaded model from {MODEL_PATH}")
        return model, le_gender, le_cancer
    except FileNotFoundError:
        print("Model or encoders not found. Training new model...")
        return None, None, None

# Flask route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Flask route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load model and encoders
        model, le_gender, le_cancer = load_model_and_encoders()
        if model is None:
            return render_template('index.html', prediction_text='Error: Model not found. Please train the model first.')

        # Get form data
        gender = request.form['gender']
        if gender not in ['M', 'F']:
            return render_template('index.html', prediction_text='Error: Invalid gender value. Use M or F.')

        # Map Yes/No to 1/2
        yes_no_map = {'No': 1, 'Yes': 2}
        features = [
            gender,
            float(request.form['age']),
            yes_no_map.get(request.form['smoking'], 0),
            yes_no_map.get(request.form['yellow_fingers'], 0),
            yes_no_map.get(request.form['anxiety'], 0),
            yes_no_map.get(request.form['peer_pressure'], 0),
            yes_no_map.get(request.form['chronic_disease'], 0),
            yes_no_map.get(request.form['fatigue'], 0),
            yes_no_map.get(request.form['allergy'], 0),
            yes_no_map.get(request.form['wheezing'], 0),
            yes_no_map.get(request.form['alcohol_consuming'], 0),
            yes_no_map.get(request.form['coughing'], 0),
            yes_no_map.get(request.form['shortness_of_breath'], 0),
            yes_no_map.get(request.form['swallowing_difficulty'], 0),
            yes_no_map.get(request.form['chest_pain'], 0)
        ]

        # Validate binary inputs
        for i, val in enumerate(features[2:], 2):
            if val not in [1, 2]:
                return render_template('index.html', prediction_text=f'Error: Invalid value for feature {i-1}. Use Yes or No.')

        # Validate age
        if features[1] <= 0:
            return render_template('index.html', prediction_text='Error: Age must be positive.')

        # Preprocess input
        features[0] = le_gender.transform([features[0]])[0]
        features = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]
        prediction_text = le_cancer.inverse_transform([prediction])[0]

        return render_template('index.html', prediction_text=f'Prediction: {prediction_text}', show_plot=True)
    except ValueError as e:
        return render_template('index.html', prediction_text=f'Error: Invalid input format. {str(e)}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

# Main execution
if __name__ == '__main__':
    # Create static directory for saving plots
    if not os.path.exists('static'):
        os.makedirs('static')

    # Check if model exists, else train a new one
    if not os.path.exists(MODEL_PATH):
        print("Training new model...")
        X, y, le_gender, le_cancer = load_and_preprocess_data()
        if X is not None and y is not None:
            model, X_test, y_test = train_model(X, y)
        else:
            print("Failed to load data. Exiting...")
            exit(1)
    else:
        print(f"Model already exists at {MODEL_PATH}. Using saved model.")

    # Run Flask app
    app.run(debug=True)