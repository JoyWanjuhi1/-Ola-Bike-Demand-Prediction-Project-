# Ola Bike Demand Prediction Project 🏍️

An end-to-end data science project focused on predicting bike-sharing demand for Ola. This project processes historical rental data, applies unsupervised clustering to capture complex weather patterns, and evaluates machine learning regression pipelines to accurately forecast trip counts.

## 🚀 Project Overview
* **Objective:** Predict total hourly bike rental demand (`count`) based on environmental, seasonal, and temporal factors.
* **Approach:** Combined exploratory data analysis, K-Means clustering for weather pattern extraction, and rigorous regression modeling.
* **Key Achievement:** Successfully identified and corrected a critical data leakage issue (removing `casual` and `registered` user columns) to ensure an honest, production-ready model evaluation.

## 🛠️ Tech Stack & Tools
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn (StandardScaler, KMeans, LinearRegression, RandomForestRegressor), Matplotlib, Seaborn

## 📈 Project Workflow & Methodology

### 1. Data Cleaning & Feature Engineering
* **Temporal Features:** Converted `datetime` to extract granular time elements (hour, day of the week, month) to map cyclical usage trends.
* **Categorical Encoding:** Converted `season` and `weather` columns into dummy variables.
* **Outlier Removal:** Applied Interquartile Range (IQR) filters to clean up anomalies in `humidity`, `windspeed`, and the target variable `count`.

### 2. Weather Clustering (Unsupervised Learning)
* To better capture complex meteorological relationships, environmental features (`temp`, `humidity`, `windspeed`) were scaled using `StandardScaler` and clustered using **K-Means Clustering**. 
* The generated labels were engineered into a new feature—`weather_cluster`—adding rich context to the training data.

### 3. Model Training & Eliminating Data Leakage
Initially, models showed suspiciously low errors. A deeper look revealed **data leakage**: the `casual` and `registered` columns directly summed up to the target `count`. 
* **The Fix:** Dropped `casual` and `registered` columns entirely to force the models to rely solely on predictive, real-time available features.
* **Models Trained:** Linear Regression vs. Random Forest Regressor (80/20 train/test split).

## 🏆 Evaluation & Key Results
To guarantee robustness and model generalization, **5-Fold Cross-Validation** was used, prioritizing **Mean Absolute Error (MAE)** to measure exactly how many rides the predictions deviated on average.

| Model | 5-Fold CV MAE | Notes |
| :--- | :--- | :--- |
| **Linear Regression** | **50.93** | Performed marginally better; robust, stable error distribution. |
| **Random Forest Regressor** | **52.34** | Slightly higher error; susceptible to overfitting on the current feature set. |

* Residual plots were utilized for the Linear Regression model to visually confirm a balanced error distribution across predictions.

## ⚙️ How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git](https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git)
