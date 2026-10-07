import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("quotes_30.csv")

print("=== Dataset Overview ===")
print(df.shape)
print(df.columns)
print(df.head())

# Feature Engineering
df["Quote_Length"] = df["Quote"].str.len()
df["Word_Count"] = df["Quote"].str.split().str.len()

print("\n=== Features Created ===")
print(df[["Quote", "Quote_Length", "Word_Count"]].head())

# Data Quality Check
print("\n=== Data Quality Check ===")

print("Missing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nDataset shape:")
print(df.shape)

# Define Features and Target
X = df[["Word_Count"]]
y = df["Quote_Length"]

print("\n=== Features and Target ===")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Train/Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n=== Train/Test Split ===")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# Train Linear Regression
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print("\n=== Model Training ===")
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

# Make Predictions
predictions = model.predict(X_test)

print("\n=== Predictions ===")
print(predictions)

# Actual vs Predicted
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

print("\n=== Actual vs Predicted ===")
print(comparison.round(2))

# Model Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\n=== Model Evaluation ===")
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2:", round(r2, 4))

# Actual vs Predicted Visualization
plt.figure(figsize=(8, 5))

plt.scatter(y_test, predictions)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.xlabel("Actual Quote Length")
plt.ylabel("Predicted Quote Length")
plt.title("Actual vs Predicted")

plt.show()

# Residual Analysis
residuals = y_test.values - predictions

print("\n=== Residuals ===")
print(residuals.round(2))

plt.figure(figsize=(8, 5))

plt.scatter(predictions, residuals)

plt.axhline(y=0)

plt.xlabel("Predicted Quote Length")
plt.ylabel("Residual")
plt.title("Residual Plot")

plt.show()

# Final Project Summary

summary = pd.DataFrame({
    "Metric": [
        "Dataset Size",
        "Training Samples",
        "Test Samples",
        "MAE",
        "RMSE",
        "R2"
    ],
    "Value": [
        len(df),
        len(X_train),
        len(X_test),
        mae,
        rmse,
        r2
    ]
})

print("\n=== Final Project Summary ===")
print(summary.round(4))

# ==========================================
# PROJECT INSIGHTS
# ==========================================

# 1. The dataset contains 30 quotes.
# 2. There are no missing values or duplicate rows.
# 3. Word_Count is strongly related to Quote_Length.
# 4. Linear Regression achieved the best performance among
#    the tested models.
# 5. The final model achieved:
#    MAE = 7.258
#    RMSE = 10.371
#    R2 = 0.9228
# 6. The model can predict Quote_Length reasonably well
#    from Word_Count.
# 7. The dataset is small, so the model should not be
#    generalized to larger datasets without further testing.

comparison.to_csv("buoi12_predictions.csv", index=False)

print("\nPrediction results saved to buoi12_predictions.csv")