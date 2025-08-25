# 🩺 Diabetes Prediction with Automated DVC Pipeline

This project is an **industry-grade Machine Learning pipeline** that predicts whether a person is diabetic or not.  
It is built with **modular coding practices** and integrates tools like **DVC, MLflow, Git, FastAPI, and logging** to ensure reproducibility, scalability, and automation.

---

## 🚀 Features
- **Automated DVC Pipeline** with modular stages:
  1. **Data Ingestion** – Reads raw CSV data.  
  2. **Preprocessing** – Applies required preprocessing and splits into train/test sets.  
  3. **Feature Engineering** – Creates meaningful features, cleans data, and saves final processed dataset.  
  4. **Model Training** – Trains a Balanced Random Forest Classifier and saves the model.  
  5. **Model Evaluation** – Evaluates trained model and stores results as JSON.  
  6. **Frontend** – FastAPI endpoint for real-time predictions.

- **Balanced Random Forest Classifier** to handle imbalanced diabetes dataset.  
- **Experiment tracking** with MLflow.  
- **Versioning** with Git + DVC.  
- **FastAPI service** for prediction.  

---

## 📊 Dataset
The dataset contains patient health metrics used to predict diabetes.

**Raw Data Columns:**
- Pregnancies (int)
- Glucose (int)
- BloodPressure (int)
- SkinThickness (int)
- Insulin (int)
- BMI (float)
- DiabetesPedigreeFunction (float)
- Age (int)
- Outcome (int - target)

**Engineered Features (Cleaned Data):**
- Pregnancies  
- Glucose  
- BloodPressure  
- Insulin  
- DiabetesPedigreeFunction  
- AgeGroup  
- BMIGroup  
- GTIratio  
- isInsulinFlag  

---

## 🛠️ Tech Stack
- **Machine Learning**: Scikit-learn (Balanced Random Forest)  
- **Pipeline Orchestration**: DVC  
- **Experiment Tracking**: MLflow  
- **Framework**: FastAPI  
- **Version Control**: Git + GitHub  
- **Logging & Utilities**: Python logging, OS module  


---

## ⚡ Setup & Usage

### 1️. Clone the repository
```bash
git clone https://github.com/your-username/diabetes-prediction-pipeline.git
cd diabetes-prediction-pipeline
```

### 2. Create virtual environment & install dependencies
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\\Scripts\\activate    # Windows
pip install -r requirements.txt
```

### 3. Run DVC Pipeline 
```bash
DVC Pipeline
```

### 4. Start FastAPI server
```bash
uvicorn src.fastapi_app:app --reload
```
### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/diabetes-prediction-pipeline.git
cd diabetes-prediction-pipeline
```

---

## 📈 Results
- Balanced Random Forest Classifier provided stable performance on imbalanced data.
- Evaluation metrics are stored in metrics.json for tracking and reproducibility.

---

## 👨‍💻 Author
Sunag M P
- 💼 Machine Learning Engineer
- 🚀 Skilled in ML, MLOps, LangChain, FastAPI, and automation pipelines.
