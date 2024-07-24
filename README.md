# Customer-Churn-ML-Prediction
## Introduction
In the highly competitive telecommunications industry, retaining customers is crucial for maintaining revenue and profitability. Customer churn, the phenomenon where customers stop using a company's services, poses a significant challenge. To address this, companies are increasingly leveraging machine learning techniques to predict churn and devise strategies to enhance customer retention.

This project aims to assist a telecommunication company in understanding their customer data, identifying factors contributing to churn, and predicting which customers are likely to churn. By building and evaluating classification models, we can provide actionable insights to help the company mitigate churn and improve customer loyalty.

## Objectives
- Data Exploration: Understand the structure and characteristics of the customer data, identifying key features that influence churn.
- Data Preprocessing: Clean the data, handle missing values, and perform feature engineering to prepare it for modeling.
- Model Building: Develop and compare various classification models, including Logistic Regression, Decision Trees, Support Vector Machines (SVM), and Random Forest, to predict customer churn.
- Model Evaluation and Interpretation: Evaluate model performance using accuracy, precision, recall, and other relevant metrics. Utilize techniques like LIME and SHAP for model interpretation and to identify important features driving churn.
- Model Optimization: Optimize model performance through hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
- Prediction and Testing: Apply the best-performing model to a separate test dataset to validate its predictive accuracy.
- Data Visualization: Create interactive dashboards using Power BI or Tableau to visualize key metrics, trends, and insights derived from the data and models.

## Data Sources
- Training Data: The first 3000 records retrieved from a remote SQL Server database.
- Additional Training Data: An additional 2000 records from a CSV file hosted on GitHub.
- Testing Data: The final 2000 records from an Excel file hosted on OneDrive, used exclusively for testing model accuracy.

# Data Understanding

### Importations 

```dotnetcli
# Data Manipulation Packages 

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pyodbc
from dotenv import dotenv_values
import scipy.stats as stats
import optuna
import warnings

from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score 
from sklearn.model_selection import cross_val_score, GridSearchCV
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as imbPipeline



warnings.filterwarnings('ignore')

```

### Load Datasets 

*The data is located in 3 different locations;*
- MySQL Database 3000 train dataset
- A csv in a github repo 2000 train data
- A csv in onedrive- test data

#### Loading the SQL Data 

```dotnetcli
# Load environment variables from .env file into a dictionary
environment_variables = dotenv_values (r'C:\Users\Admin\OneDrive\OneDrive-Azubi\Customer-Churn-Prediction-\.env')

# Get the values for the credentials you set in the '.env' file
server = environment_variables.get('SERVER')
database = environment_variables.get('DATABASE')
username = environment_variables.get('USERNAME')
password = environment_variables.get('PASSWORD')

# Create a connection string
connection_string = f"DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password};MARS_Connection=yes;MinProtocolVersion=TLSv1.2;"


connection = pyodbc.connect(connection_string)
```

```dotnetcli
# Loading the First 3000 dataset
query = "SELECT * FROM LP2_Telco_churn_first_3000"

data = pd.read_sql(query, connection)

data.head()
```

#### Loading train Data from the csv

``` 
# Loading the second 2000 data
df=pd.read_csv('../data/LP2_Telco-churn-second-2000.csv')
df.head()
```

#### Merge Data

```
# Combine DataFrames
churn_prime = pd.concat([data, df], ignore_index=True)

churn_prime.head()

```


<<<<<<< HEAD
After Merging the train data we did some data inspection using funstions such as .info(), .shape, .describe(), .isnull(), .duplicated()


#### Hypothesis Testing

#### Chi-Square Test
- **Hypothesis**:
  - Null Hypothesis (H0): There is no significant difference in churn rates among customers with different contract types.
  - Alternative Hypothesis (H1): There is a significant difference in churn rates among customers with different contract types.
- **Result**:
  - Chi-Square Statistic: 881.6208905118242
  - P-value: 3.61789584641233e-192
  - Degrees of Freedom: 2
  - Conclusion: The p-value is extremely low, providing strong evidence against the null hypothesis. Therefore, we reject the null hypothesis for all contract types tested, indicating a significant difference in churn rates among customers with different contract types
    
#### Handling Missing Values

- The `TotalCharges` column's missing values are filled with the median.
- Missing values in categorical columns (`MultipleLines`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Churn`) are filled with the mode.
- The `customerID` column is dropped as it does not provide statistical or computational significance.

#### Final Data Check

- The dataset has no duplicated rows.
- The dataset is not evenly distributed, with mean and median significantly differing for numerical columns.
- The dataset contains outliers as seen in the long tails of KDE plots and histograms for `TotalCharges`.

## Exploratory Data Analysis (EDA)

### Key Insights

- **Churn Distribution**: The churn rate is higher among customers with multiple services.
- **Tenure and Churn**: Most customers tend to churn early; those who stay longer are less likely to churn.
- **Payment Methods and Churn**: There is a significant difference in churn rates among customers with different payment methods.
- **Demographics and Churn**: Customers with different demographic factors (e.g., gender, senior citizen status) show varied churn rates.
- **Internet Service and Churn**: The type of internet service impacts churn rates.
- **Contract Type and Churn**: Different contract types show varied churn rates, with month-to-month contracts having higher churn.

### Data Transformation

### Encoding Categorical Variables

- Categorical columns are encoded using appropriate encoding techniques to prepare the data for modeling.

### Handling Imbalanced Data

- The dataset exhibits a significant class imbalance, with non-churn instances significantly outnumbering churn instances.

### Data Splitting

- The data is split into training and testing sets using an 80-20 split, with stratification to maintain the distribution of the target variable.

#### Skewness Check

- Numeric features show varying levels of skewness, indicating the need for appropriate scaling or transformation.

#### Target Encoding

- The target variable `y` is encoded using `LabelEncoder` to prepare it for modeling


## Data Preprocessing

### Scaling Decision

- Standard Scaler was disqualified due to non-normal distribution of data.
- MinMax Scaler was disqualified due to the presence of outliers.
- Robust Scaler was chosen to handle biases in the training data.
- Quantile Transformer was used to transform the data to a more evenly distributed shape.

## Feature Engineering

*Numerical and categorical features were processed separately:*

```csharp-interactive


numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler()),
    ('QuantileTransformation', QuantileTransformer()),
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder()),
])

preprocessor = ColumnTransformer(transformers=[
    ('num_pipeline', numeric_pipeline, numerical_columns),
    ('cat_pipeline', categorical_pipeline, categorical_columns),
])
``` 

## **Model Training and Evaluation**

### Initial Training on Imbalanced Data

*Multiple models were trained on imbalanced data:*

- Logistic Regression
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Gradient Boosting Machine (GBM)
- Neural Network

*The initial results showed a bias towards the majority class ("No" churn)*
*Addressing Class Imbalance*
*SMOTE (Synthetic Minority Over-sampling Technique) was applied to balance the dataset*

#### Model Performance on Balanced Data

| Model              | Accuracy | Precision | Recall  | F1-Score |
|--------------------|----------|-----------|---------|----------|
| KNN                | 0.685828 | 0.757547  | 0.685828| 0.703403 |
| Neural Network     | 0.758176 | 0.774210  | 0.758176| 0.764313 |
| Logistic Regression| 0.753221 | 0.808432  | 0.753221| 0.766126 |
| SVM                | 0.762141 | 0.793921  | 0.762141| 0.771759 |
| GBM                | 0.783944 | 0.801629  | 0.783944| 0.790134 |
| Random Forest      | 0.790882 | 0.791642  | 0.790882| 0.791254 |


#### Key Findings

- All models achieved relatively high accuracy scores (0.686 to 0.791) on balanced data.
- Random Forest performed best in accuracy (0.791) and F1-score (0.791).
- Logistic Regression and GBM showed the highest precision (0.808 and 0.802 respectively).
- Random Forest and GBM achieved the highest recall scores (0.791 and 0.784)

#### Model Comparison and Recommendations

- Random Forest and GBM consistently performed well across all metrics.
- Logistic Regression and SVM demonstrated strong performance with high precision scores.
- Neural Network showed competitive performance but slightly lower precision.
- KNN exhibited the lowest recall among the models.

### Conclusion in Regards to Modeling and Evaluation

- Ensemble methods (Random Forest and GBM) are recommended for their balanced performance across all metrics. They are particularly suitable for applications where F1-score is the primary consideration.


### Hperparameter Tuning 

#### Hyperparameter Tuning Results

#### Gradient Boosting Machine (GBM)

- **Error Handling:** The error messages encountered during trials were due to setting `'max_features': 'auto'`, which is not compatible with `cross_val_score` in scikit-learn. This issue will be addressed in future tuning iterations.

- **Optimization Progress:** Despite encountering errors in some trials, the study completed all 100 trials as specified (`n_trials=100`).

- **Best F1-Score:** The best F1-score observed during the study was **0.794**, achieved in Trial 63, which represents a slight improvement.

#### Random Forest

- **Trial Number:** Trial 99 was the 99th trial conducted during the optimization process.

- **Trial Result:** The F1-score observed for Trial 99 was **0.6353**.

- **Best Trial:** The best F1-score observed overall throughout all trials was **0.6424**, achieved in Trial 40.

- **Performance Drop:** There was a drop in F1 Score from an initial 0.791 to 0.642, indicating a need for further hyperparameter adjustment to enhance performance.

#### Challenges and Moving Forward 

- The only Major Challenge was with the hyperparameter tunings of our 2 best performing models which we will seek to work it out with the best hyperparameters so that we can move forward to sellecting the best model for our test data
- Other Challenges were learning oportnities :blush:
- We will also be exporting core machine learning Components for future use in other projects 

#### Preview into Our Data Visualization Dashboard on Power Bi
![Reference Image](<Cutomer Churn Prediction.jpg>)
