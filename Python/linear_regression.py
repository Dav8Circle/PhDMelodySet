import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

# Load x and y values from R-generated file
data = pd.read_csv("R/xy_values.txt", sep=" ")
x = data['x'].values
y = data['y'].values
n = len(x)

# Prepare data for least squares
X = np.vstack([x, np.ones(len(x))]).T

# Fit linear regression model using least squares
beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
slope, intercept = beta

# Calculate statistics
y_pred = slope * x + intercept
r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
n_features = 1  # number of predictors
df_model = n_features
df_residual = n - n_features - 1
f_statistic = (r_squared * df_residual) / ((1 - r_squared) * df_model)
p_value = 1 - stats.f.cdf(f_statistic, df_model, df_residual)

# Print results
print("\nPython Linear Regression Results")
print("=============================\n")
print(f"Model Coefficients:")
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"\nR-squared: {r_squared:.4f}")
print(f"F-statistic: {f_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

