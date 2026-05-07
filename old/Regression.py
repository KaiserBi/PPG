import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

data = pd.read_csv("totaldata.csv")

# Separate predictors (X) and target (y)
X = data[['BPS', 'HRV', 'LF/HF', 'VLF', 'LF', 'HF']]
y = data['1/Mean']
print(y)
print(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)
lr = LinearRegression()

# Coefficients
print("Intercept:", ridge.intercept_)
print("Coefficients:")
for col, coef in zip(X.columns, ridge.coef_):
    print(f"{col}: {coef:.4f}")


ridgescores = cross_val_score(ridge, X_scaled, y, cv=5, scoring='r2')
lrscores = cross_val_score(lr, X_scaled, y, cv=5, scoring='r2')
print("lrscores:")
print(lrscores, lrscores.mean())


print("ridgescores:")
print("\n5-fold cross-validation R² scores:", ridgescores)
print("Mean CV R²:", np.mean(ridgescores))

