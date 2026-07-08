# ❤️ Heart Disease Risk Predictor

A machine learning web application that predicts a patient's risk of heart disease based on clinical and demographic data, built with **scikit-learn** and deployed with **Streamlit**.

**🔗 Live App:** [Add your Streamlit Cloud URL here after deployment]

---

## 📋 Overview

This project uses a Random Forest classifier trained on a public heart disease dataset to estimate the probability that a patient has heart disease, based on 13 clinical features such as age, blood pressure, cholesterol, and ECG results.

The trained model is served through an interactive Streamlit interface where a user can input patient details and receive an instant prediction with an associated probability score.

---

## 🧠 Model & Methodology

The full training process is documented in [`Predicting_Heart_Disease.ipynb`](./Predicting_Heart_Disease.ipynb), and includes:

- **Exploratory data analysis** and outlier detection
- **Feature scaling** — `StandardScaler` applied to numerical features (`age`, `resting_blood_pressure`, `cholesterol`, `max_heart_rate`, `st_depression`)
- **Model comparison** across K-Nearest Neighbors, Logistic Regression, and Random Forest
- **Hyperparameter tuning** via k-fold cross-validation
- **Feature importance analysis** using permutation importance
- **Final model:** Random Forest Classifier (`n_estimators=100`, `max_depth=6`)

The trained model, fitted scaler, and expected feature order are bundled together in [`rf_pipeline.pkl`](./rf_pipeline.pkl) using `joblib`, ensuring inference always matches the exact preprocessing used during training.

### Dataset Features

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | 0 = Female, 1 = Male |
| `chest_pain_type` | 0–3 (typical angina → asymptomatic) |
| `resting_blood_pressure` | Resting blood pressure (mm Hg) |
| `cholesterol` | Serum cholesterol (mg/dl) |
| `fasting_blood_sugar` | 1 if fasting blood sugar > 120 mg/dl, else 0 |
| `ecg` | Resting electrocardiographic results (0–2) |
| `max_heart_rate` | Maximum heart rate achieved |
| `exercise_induced_chest_pain` | 1 = Yes, 0 = No |
| `st_depression` | ST depression induced by exercise relative to rest |
| `st_slope` | Slope of the peak exercise ST segment (0–2) |
| `stained_blood_vessels` | Number of major vessels stained by fluoroscopy (0–4) |
| `blood_disorder` | Thalassemia code (0–3) |
| `heart_disease` | Target: 1 = disease present, 0 = no disease |

---

## 📁 Project Structure

```
Heart_Disease/
├── app.py                          # Streamlit application
├── rf_pipeline.pkl                 # Trained model + scaler + feature order
├── requirements.txt                # Python dependencies
├── data_01.csv                     # Training dataset
├── Predicting_Heart_Disease.ipynb  # Full model development notebook
├── knn_compare_model.png           # KNN train/test accuracy comparison plot
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started (Local Setup)

### Prerequisites
- Python 3.9+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/heart-disease-app.git
   cd heart-disease-app
   ```

2. **Create and activate a virtual environment**

   *macOS / Linux:*
   ```bash
   python3 -m venv dataApp
   source dataApp/bin/activate
   ```

   *Windows (PowerShell):*
   ```powershell
   python -m venv dataApp
   .\dataApp\Scripts\Activate.ps1
   ```

   *Windows (Command Prompt):*
   ```cmd
   python -m venv dataApp
   dataApp\Scripts\activate.bat
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

   The app will open automatically at `http://localhost:8501`.

---

## ☁️ Deployment

This app is deployed on **Streamlit Community Cloud**:

1. Push the repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select this repository and branch, and set the main file path to `app.py`.
4. Click **Deploy**.

---

## 🛠️ Built With

- [Python](https://www.python.org/)
- [scikit-learn](https://scikit-learn.org/) — model training
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data handling
- [Streamlit](https://streamlit.io/) — web app framework
- [joblib](https://joblib.readthedocs.io/) — model serialization

---

## ⚠️ Disclaimer

This application is built for **educational purposes** only. It is trained on a public dataset and its predictions **should not** be used as a substitute for professional medical diagnosis or advice. Always consult a qualified healthcare provider for medical concerns.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**ChymaOge**