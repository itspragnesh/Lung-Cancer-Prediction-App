# 🫁 Lung Cancer Detection Web App

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

---

## 📘 Overview

This is a **Flask-based web application** for predicting lung cancer risk using machine learning.  
It leverages an **XGBoost classifier** trained on patient symptom data and risk factors.  
Users can input their details via a **responsive Bootstrap form**, and the app predicts the **likelihood of lung cancer** (YES/NO) while displaying a **feature importance plot**.

### 🔹 Key Components:
- **Frontend**: Bootstrap 5 for a modern UI.  
- **Backend**: Flask for routing and logic.  
- **Model**: XGBoost (binary classification).  
- **Visualization**: Matplotlib feature importance chart.

**Demo:** [Live Demo](https://your-app-url.herokuapp.com) *(Replace with your deployed link)*

---

## ✨ Features

- Interactive form to input gender, age, and 14 binary risk factors.
- Real-time input validation (JavaScript).
- Model prediction with decoded output (“YES” / “NO”).
- Dynamic feature importance plot generation.
- Custom error handling and responsive layout.
- Lightweight and deployment-ready.

---

## ⚙️ Requirements

- Python 3.8+
- Flask  
- XGBoost  
- scikit-learn  
- Pandas  
- NumPy  
- Joblib  
- Matplotlib  

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/lung-cancer-detection-app.git
cd lung-cancer-detection-app
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# Activate:
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install flask xgboost scikit-learn pandas numpy joblib matplotlib
```

### 4. Download Dataset
- Place `lung_cancer.csv` in the project root.
- Source: [UCI Lung Cancer Dataset](https://archive.ics.uci.edu/dataset/554/lung+cancer)  
  *(or Kaggle equivalent)*  
- Expected columns:
  ```
  GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE,
  CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING,
  COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN, LUNG_CANCER
  ```

---

## 📂 Folder Structure

```
lung-cancer-detection-app/
│
├── app.py                    # Main Flask app
├── lung_cancer.csv           # Dataset
├── requirements.txt          # Dependencies
├── xgboost_model.pkl         # Trained ML model
├── gender_encoder.pkl        # Gender label encoder
├── cancer_encoder.pkl        # Cancer label encoder
│
├── templates/
│   └── index.html            # Frontend (Bootstrap)
│
└── static/
    ├── styles.css            # Custom CSS
    ├── script.js             # JavaScript validation
    └── feature_importance.png# Generated feature plot
```

---

## 🧠 Usage

### 1. Run the App
```bash
python app.py
```
- Trains model on first run (creates `.pkl` files).  
- Opens local server at **http://127.0.0.1:5000**.

### 2. Access in Browser
- Fill out the form and click **Predict**.
- View **result** and **feature importance graph**.

### 3. Prediction Flow
- Encodes user inputs (Yes/No → 1/2, Gender → encoded).
- XGBoost predicts binary output (0 → NO, 1 → YES).
- Model accuracy: **~95%** (varies by dataset).

---

## 🧩 Model Training Details

- **Preprocessing**
  - Label encoding for categorical fields.  
  - Train/test split: 80/20.

- **XGBoost Parameters**
  ```python
  use_label_encoder=False
  eval_metric='logloss'
  random_state=42
  ```

- **Evaluation**
  - Console output: Accuracy + Classification Report.  
  - Saves `feature_importance.png` to `static/`.

To retrain, delete the `.pkl` files and rerun `app.py`.

---

## ☁️ Deployment

### 🟣 Heroku / Local
- Add a `Procfile`:
  ```
  web: gunicorn app:app
  ```
- Set environment variable:
  ```
  PYTHONHASHSEED=0
  ```

### 🐳 Docker (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```
Build and run:
```bash
docker build -t lung-cancer-app .
docker run -p 5000:5000 lung-cancer-app
```

---

## 🤝 Contributing

1. Fork this repository  
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a **Pull Request** 🎉

---

## 📜 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgments

- **Dataset**: UCI Machine Learning Repository  
- **UI**: Bootstrap 5  
- **ML**: XGBoost, scikit-learn  

---

> For issues or questions, open a GitHub Issue!  
> 🚀 *Made with ❤️ by Pragnesh*
