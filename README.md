# Advertising Sales Prediction

Predicting product sales based on advertising spend using advanced regression techniques and feature engineering. This project helps businesses optimize marketing budgets by estimating sales outcomes from TV and Radio advertising investments.

---

## 🚀 Project Overview

This project analyzes advertising data to predict sales based on spending across **TV, Radio, and Newspaper** channels. The workflow includes:

- Exploratory Data Analysis (EDA) with correlation heatmaps.
- Feature selection based on correlation analysis (TV & Radio selected; Newspaper excluded).
- Polynomial regression (degree 2) to model non-linear interactions.
- Hyperparameter tuning of **Ridge** and **Lasso** regression using 5-Fold Cross-Validation.
- Benchmarking linear models against a non-linear **Random Forest Regressor**.
- Residual and feature importance visualization for interpretability.
- Practical prediction function for new marketing scenarios.

---

## 📊 Dataset

- `Advertising Dataset.csv` contains:
  - `TV` – TV advertising spend (in $k)
  - `Radio` – Radio advertising spend (in $k)
  - `Newspaper` – Newspaper advertising spend (in $k)
  - `Sales` – Product sales (in $k)

---

## 🔧 Technologies & Libraries

- **Python** | **NumPy** | **Pandas** | **Matplotlib** | **Seaborn**  
- **Scikit-learn**: LinearRegression, Ridge, Lasso, RandomForestRegressor, PolynomialFeatures, StandardScaler, GridSearchCV, cross_val_score

---

## ⚙️ Key Steps

1. **EDA & Feature Selection**
   - Correlation analysis revealed strong positive correlation of TV (0.78) and Radio (0.58) with Sales.
   - Newspaper had low correlation (0.23) and was excluded to improve model performance.

2. **Feature Engineering**
   - Applied Standard Scaling.
   - Generated polynomial features (degree 2) to capture interaction effects (e.g., TV × Radio).

3. **Modeling & Hyperparameter Tuning**
   - Ridge and Lasso regression were tuned using cross-validated RMSE to find optimal alpha.
   - Linear Regression and Random Forest used as benchmarks.

4. **Model Evaluation**
   - Cross-validated RMSE used for fair comparison.
   - Ridge Regression achieved the **lowest RMSE: 0.600**.
   - Random Forest was higher (RMSE = 0.861), confirming linear model suitability.

5. **Residual Analysis & Feature Importance**
   - Residuals plots and histograms validated assumptions.
   - Coefficients from Ridge Regression revealed the importance of TV, Radio, and their interaction.

6. **Bonus: Newspaper Feature Justification**
   - Including Newspaper increased RMSE to 0.613.
   - Exclusion was confirmed as the optimal choice.

---

## 📈 Results

| Model                | CV RMSE |
|---------------------|---------|
| Ridge Regression    | 0.600   |
| Lasso Regression    | 0.600   |
| Linear Regression   | 0.600   |
| Random Forest       | 0.861   |

**Best Model:** Ridge Regression (CV RMSE = 0.600)


Clone the repository:

git clone https://github.com/manasa190/Advertising-Sales-Prediction.git
cd Advertising-Sales-Prediction


Install dependencies:

pip install -r requirements.txt


Run the notebook:

jupyter notebook Sales_Prediction.ipynb


Use the predict_sales function for new predictions.
