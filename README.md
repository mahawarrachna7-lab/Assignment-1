diabetes-prediction-logistic-regression/
│
├── data/
│   └── diabetes2.csv          # Dataset (not included in repo, add manually)
│
├── notebooks/
│   └── diabetes_prediction.ipynb   # Jupyter notebook with full workflow
│
├── src/
│   └── model.py               # Python script for training and evaluation
│
├── results/
│   └── confusion_matrix.png   # Model confusion matrix plot
│   └── metrics.txt            # Saved evaluation metrics
│
├── requirements.txt           # Dependencies
├── README.md                  # Project overview and instructions
└── .gitignore                 # Ignore unnecessary files
# 🩺 Diabetes Prediction Using Logistic Regression

This project predicts the presence of diabetes in patients using diagnostic health parameters and a Logistic Regression model.

---

## 🎯 Problem Statement
The objective is to build a reliable classification model that predicts **Outcome**:
- `1` → Diabetic  
- `0` → Non-Diabetic  

Dataset: `diabetes2.csv` (health-related features + Outcome variable).

---

## 📊 Abstract
We employ Logistic Regression to classify patients based on diagnostic features.  
The model is evaluated with **Accuracy, Recall, F1 Score, and Confusion Matrix**.  

Results show ~**75.3% accuracy**, with good recall (73.1%), making the model effective for diabetes prediction.

---

## ⚙️ Methodology
1. **Data Loading**: Import dataset with features  
   - Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age  
   - Target: `Outcome`

2. **Data Visualization**:  
   - Histogram of glucose levels  
   - Bar chart of outcomes  
   - Correlation heatmap  
   - Age distribution boxplot  

3. **Data Processing**:  
   - Train-test split  
   - Feature scaling using `StandardScaler`

4. **Model Training**:  
   - Logistic Regression model (`scikit-learn`)  

5. **Evaluation**:  
   - Accuracy: 0.7532  
   - Recall: 0.7311  
   - F1 Score: 0.63  
   - Confusion Matrix  

---

## 📈 Results
|                 | Predicted Non-Diabetic | Predicted Diabetic |
|-----------------|-------------------------|--------------------|
| **Actual Non-Diabetic** | 84 (TN)                  | 25 (FP)            |
| **Actual Diabetic**     | 13 (FN)                  | 32 (TP)            |

- **Accuracy**: 75.32%  
- **Recall**: 73.11%  
- **F1 Score**: 63%  

---

## 🚀 How to Run
1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/diabetes-prediction-logistic-regression.git
   cd diabetes-prediction-logistic-regression
   pip install -r requirements.txt
jupyter notebook notebooks/diabetes_prediction.ipynb
python src/model.py
pandas
numpy
scikit-learn
matplotlib
seaborn
pandas
numpy
scikit-learn
matplotlib
seaborn

