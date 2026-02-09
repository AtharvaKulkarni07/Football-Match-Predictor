# ⚽ Football Match Predictor

A machine learning–powered **Streamlit web application** that predicts whether the **home team will win** a football match based on historical performance statistics.

The app uses an **ensemble of models** (Logistic Regression, Random Forest, Gradient Boosting) to provide a probability-based prediction with confidence scores.

---

## 🚀 Features

- Interactive Streamlit dashboard
- Three trained ML models with ensemble averaging
- Probability-based predictions (not just win/lose)
- Clean UI with metrics, confidence levels, and explanations
- Cached model loading for fast performance

---

## 🧠 Models Used

- **Logistic Regression** – Interpretable baseline model  
- **Random Forest** – Handles non-linear feature interactions  
- **Gradient Boosting** – Best ROC AUC (0.69)  
- **Ensemble Model** – Average of all three for robust predictions  

---

## 📊 Dataset Overview

- Total matches: **6,840**
- Training set: **5,472 matches (80%)**
- Test set: **1,368 matches (20%)**
- Historical home win rate: **46.43%**
- Total features used: **18**

**Example features:**
- Goals scored & conceded
- Team points
- Matches played
- Goal differences
- Per-match averages

---

## 🖥️ Application Pages

### 🏠 Home
- Overview of models and dataset
- Accuracy and ROC AUC metrics
- Usage instructions

### 🔮 Prediction
- Input home & away team statistics
- Individual model probabilities
- Ensemble prediction with confidence score

### ℹ️ About
- Project explanation
- Model details
- Dataset statistics
- Disclaimer

---

## 📦 Project Structure
```
Football Match Prediction/
│
├── app.py
├── models/
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Create a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app
```bash
streamlit run app.py
```

---

## 📄 Requirements

Example `requirements.txt`:
```
streamlit
pandas
numpy
scikit-learn
plotly
```

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.

Football match outcomes depend on many unpredictable factors such as injuries, weather, tactics, and motivation, which are not captured by the model.

---

## 📌 Version

**Football Match Prediction Dashboard v1.0**

---

## 👤 Author

Developed as a machine learning project for football match outcome prediction using historical data.

---

**If you want, I can also:**
- Tailor this README for **research submission**
- Shorten it for **GitHub recruiters**
- Add **model training details**
- Or write a **paper-style project description**

Just say the word ⚽📊