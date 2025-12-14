# 🚗 Car Price Prediction using Machine Learning

## 📌 Project Overview

This project aims to build a **machine learning model** that predicts car prices based on their specifications and features. The goal is to create a reliable regression model that generalizes well to unseen data and avoids common issues such as data leakage and overfitting.

---

## 🎯 Problem Statement

Car prices depend on many factors such as production year, mileage, engine specifications, brand, and other attributes. Manually estimating prices can be inaccurate and subjective.

**Objective:**
Predict the car price as accurately as possible using supervised machine learning techniques.

---

## 📊 Dataset Description

The dataset contains multiple features related to cars, including:

* Numerical features (e.g. mileage, engine volume, production year)
* Categorical features (e.g. brand, model, fuel type)
* Target variable: **Price**

---

## 🧹 Data Preprocessing

All preprocessing steps were handled inside a **Pipeline** to ensure consistency and prevent data leakage:

* Handling missing values
* Encoding categorical variables using One-Hot Encoding
* Scaling numerical features

This approach guarantees that the same transformations are applied during training, validation, and testing.

---

## 🔀 Data Splitting Strategy

The dataset was split as follows:

* **Training set:** ~64%

* **Validation set:** ~16%

* **Test set:** ~20%

* The training set was used for model training and cross-validation.

* The validation set was used for intermediate evaluation during development.

* The test set was used only once for final evaluation.

---

## 🤖 Model Selection

The chosen model is **RandomForestRegressor**, due to its ability to:

* Handle nonlinear relationships
* Work well with mixed data types
* Reduce overfitting compared to single decision trees

---

## 🔧 Hyperparameter Tuning

Hyperparameters were optimized using **GridSearchCV** with 3-fold cross-validation.

### Tuned Parameters:

* `n_estimators`: [50, 100, 200]
* `max_depth`: [None, 5, 10]
* `min_samples_split`: [2, 5]

---

## 📈 Evaluation Metrics

The final model was evaluated using multiple regression metrics:

* **R² Score** – Measures how well the model explains variance
* **MAE (Mean Absolute Error)** – Average absolute prediction error
* **RMSE (Root Mean Squared Error)** – Penalizes larger errors

### ✅ Final Test Results:

* **Test R²:** 0.75
* **Test MAE:** ~4,048
* **Test RMSE:** ~7,772

These results indicate strong predictive performance and good generalization on unseen data.

---

## 🏁 Conclusion

* The model successfully captures the relationship between car features and price.
* The use of Pipelines and proper data splitting ensures a clean and professional workflow.
* The achieved performance is suitable for real-world pricing scenarios.

---

## 🚀 Future Improvements

* Apply log transformation to the target variable
* Experiment with boosting models such as XGBoost or LightGBM
* Perform deeper feature importance analysis
* Deploy the model using a simple web interface

---

## 🛠️ Tools & Libraries

* Python
* Pandas, NumPy
* Scikit-learn
* Joblib

---

## 👤 Author

**Ahmed Mahmoud**

Machine Learning & Data Analysis Enthusiast

---

📌 *This project was developed as part of a machine learning portfolio and demonstrates an end-to-end regression workflow.*

