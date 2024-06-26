# Customer-Churn-ML-Prediction
## Introduction
In the highly competitive telecommunications industry, retaining customers is crucial for maintaining revenue and profitability. Customer churn, the phenomenon where customers stop using a company's services, poses a significant challenge. To address this, companies are increasingly leveraging machine learning techniques to predict churn and devise strategies to enhance customer retention.

This project aims to assist a telecommunication company in understanding their customer data, identifying factors contributing to churn, and predicting which customers are likely to churn. By building and evaluating classification models, we can provide actionable insights to help the company mitigate churn and improve customer loyalty.

## Objectives
Data Exploration: Understand the structure and characteristics of the customer data, identifying key features that influence churn.
Data Preprocessing: Clean the data, handle missing values, and perform feature engineering to prepare it for modeling.
Model Building: Develop and compare various classification models, including Logistic Regression, Decision Trees, Support Vector Machines (SVM), and Random Forest, to predict customer churn.
Model Evaluation and Interpretation: Evaluate model performance using accuracy, precision, recall, and other relevant metrics. Utilize techniques like LIME and SHAP for model interpretation and to identify important features driving churn.
Model Optimization: Optimize model performance through hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
Prediction and Testing: Apply the best-performing model to a separate test dataset to validate its predictive accuracy.
Data Visualization: Create interactive dashboards using Power BI or Tableau to visualize key metrics, trends, and insights derived from the data and models.

## Data Sources
Training Data: The first 3000 records retrieved from a remote SQL Server database.
Additional Training Data: An additional 2000 records from a CSV file hosted on GitHub.
Testing Data: The final 2000 records from an Excel file hosted on OneDrive, used exclusively for testing model accuracy.

## Hypothesis
Null Hypothesis (H0): Customer demographics have no significant effect on churn.
Alternative Hypothesis (H1): Customer demographics have a significant effect on churn.
