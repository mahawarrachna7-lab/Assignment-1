Diabetes Prediction Using Logistic Regression
Overview

This project aims to predict the presence of diabetes in patients using diagnostic parameters through a machine learning classification approach. The model leverages logistic regression to classify patients as diabetic or non-diabetic based on health indicators from the dataset diabetes2.csv.

Problem Statement 🎯

The objective is to develop a reliable classification model that predicts the Outcome variable — where:

1 indicates a positive diabetes diagnosis.

0 indicates no diabetes.

The dataset contains multiple health-related features for different patients, which serve as predictors.

Abstract 🎯

This study employs machine learning techniques to build a predictive model for diabetes diagnosis. A logistic regression classifier is trained using diagnostic metrics, then evaluated using key performance indicators such as accuracy, recall, F1 score, and confusion matrix analysis. The approach focuses on providing a practical tool to assist in identifying diabetic patients effectively.

Methodology 🎯

Data Loading:
The dataset diabetes2.csv is imported into a Pandas DataFrame. Features include:

Pregnancies

Glucose

Blood Pressure

Skin Thickness

Insulin

BMI

Diabetes Pedigree Function

Age
The target variable is Outcome.

Data Visualization:
Various plots are generated to explore the data:

Histogram of Glucose levels

Bar chart of diabetes outcome counts

Correlation heatmap of variables

Boxplot of age distribution by outcome

Data Splitting:
The dataset is split into training and testing subsets to allow model training and unbiased evaluation.

Feature Scaling:
Features are standardized using StandardScaler to normalize data and enhance model performance.

Model Training:
A logistic regression model is trained on the scaled training data to learn the relationship between features and diabetes outcomes.

Model Evaluation:
The model's predictions on the test set are evaluated using:

Accuracy

Recall

F1 Score

Confusion Matrix

Classification Report

Results and Output
Metric	Value
Accuracy	0.7532
Recall	0.7311
F1 Score	0.63
Confusion Matrix Breakdown
	Predicted Non-Diabetic	Predicted Diabetic
Actual Non-Diabetic	84 (True Negatives)	25 (False Positives)
Actual Diabetic	13 (False Negatives)	32 (True Positives)
Conclusion

The logistic regression model demonstrated a strong ability to predict diabetes based on the provided health metrics, achieving an accuracy of approximately 75.3%. The recall of 73.1% indicates the model successfully identifies most diabetic cases. With a balanced F1 score of 63%, the model shows good precision and recall balance. Further improvements may include exploring alternative models and feature engineering.

How to Run the Project

Clone the repository.

Ensure you have Python installed along with the necessary libraries:

pandas, numpy, scikit-learn, matplotlib, seaborn


Load the diabetes2.csv dataset into the project directory.

Run the notebook or script to execute the full workflow from data loading to model evaluation.


