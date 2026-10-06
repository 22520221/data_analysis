# ============================================================
# 1. IMPORT LIBRARIES + LOAD DATA
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("quotes_30.csv")


# ============================================================
# 2. ANALYSIS: AUTHORS
# ============================================================

# Count number of quotes for each author
author_counts = df["Author"].value_counts()

#print("=== Quote count by author ===")
#print(author_counts)


# Top 10 authors
top_authors = author_counts.head(10)

#print("\n=== Top 10 authors ===")
#print(top_authors)


# Percentage of quotes by author
author_percentage = (author_counts / len(df) * 100).round(2)

#print("\n=== Quote percentage by author ===")
#print(author_percentage)


# Plot top authors
#top_authors.plot(kind="bar")

#plt.title("Top 10 Authors by Number of Quotes")
#plt.xlabel("Author")
#plt.ylabel("Number of Quotes")
#plt.xticks(rotation=45)
#plt.tight_layout()
#plt.show()


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

# Number of characters in each quote
df["Quote_Length"] = df["Quote"].str.len()


# Number of words in each quote
df["Word_Count"] = df["Quote"].str.split().str.len()


#print("\n=== New features ===")
#print(df[["Quote", "Quote_Length", "Word_Count"]].head(10))


# ============================================================
# 4. ANALYSIS: QUOTE LENGTH
# ============================================================

# Mean
mean_length = df["Quote_Length"].mean()

# Median
median_length = df["Quote_Length"].median()

#print("\n=== Quote Length Statistics ===")
#print("Mean:", mean_length)
#print("Median:", median_length)


# Full descriptive statistics
#print("\n=== Descriptive Statistics ===")
#print(df["Quote_Length"].describe())


# Distribution of quote length
#df["Quote_Length"].plot(kind="hist", bins=10)

#plt.title("Distribution of Quote Length")
#plt.xlabel("Quote Length")
#plt.ylabel("Number of Quotes")
#plt.show()


# ============================================================
# 5. OUTLIER ANALYSIS
# ============================================================

# Calculate Q1, Q3 and IQR
Q1 = df["Quote_Length"].quantile(0.25)
Q3 = df["Quote_Length"].quantile(0.75)

IQR = Q3 - Q1

#print("\n=== Outlier Analysis ===")
#print("Q1:", Q1)
#print("Q3:", Q3)
#print("IQR:", IQR)


# Calculate upper bound
upper_bound = Q3 + 1.5 * IQR

#print("Upper Bound:", upper_bound)


# Find outliers
outliers = df[df["Quote_Length"] > upper_bound]

#print("\n=== Outliers ===")
#print(outliers[["Author", "Quote_Length"]])


# Compare mean before and after removing outliers
mean_before = df["Quote_Length"].mean()

mean_after = (
    df[df["Quote_Length"] <= upper_bound]["Quote_Length"].mean()
)

#print("\n=== Mean comparison ===")
#print("Mean before:", mean_before)
#print("Mean after:", mean_after)


# ============================================================
# 6. CORRELATION + REGRESSION
# ============================================================

# Calculate correlation
correlation = df["Word_Count"].corr(df["Quote_Length"])

#print("\n=== Correlation ===")
#print("Correlation:", correlation)


# Scatter plot
#plt.scatter(df["Word_Count"], df["Quote_Length"])

#plt.title("Word Count vs Quote Length")
#plt.xlabel("Word Count")
#plt.ylabel("Quote Length")

#plt.show()


# Linear regression
x = df["Word_Count"]
y = df["Quote_Length"]

m, b = np.polyfit(x, y, 1)


# Regression equation
#print("\n=== Linear Regression ===")
#print("m =", m)
#print("b =", b)

#print(
#    f"Equation: Quote_Length = {m:.4f} * Word_Count + {b:.4f}"
#)


# Scatter plot + regression line
#plt.scatter(x, y)

#plt.plot(x, m * x + b)

#plt.title("Word Count vs Quote Length with Regression Line")
#plt.xlabel("Word Count")
#plt.ylabel("Quote Length")

#plt.show()


# R-squared
r_squared = correlation ** 2

#print("\n=== R-squared ===")
#print("R-squared:", r_squared)

# ============================================================
# 7. MACHINE LEARNING
# ============================================================

print("\n=== Machine Learning Data ===")

X = df[["Word_Count"]]
y = df["Quote_Length"]

print("X:")
print(X.head())

print("\ny:")
print(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n=== Train Test Split ===")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

model = LinearRegression()

model.fit(X_train, y_train)

print("\n=== Model Training ===")
print("Model has been trained.")

predictions = model.predict(X_test)

print("\n=== Predictions ===")
print(predictions)

print("\n=== Actual vs Predicted ===")

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

print(comparison)

mae = mean_absolute_error(y_test, predictions)

print("\n=== Model Evaluation ===")
print("MAE:", mae)

mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5

print("\n=== RMSE ===")
print("RMSE:", rmse)

r2 = r2_score(y_test, predictions)

print("\n=== R-squared ===")
print("R-squared:", r2)

# ============================================================
# 8. ACTUAL VS PREDICTED
# ============================================================

print("\n=== Check Plot Data ===")

print("y_test:")
print(y_test)

print("\npredictions:")
print(predictions)

actual = y_test.to_numpy()
predicted = predictions

plt.scatter(actual, predicted)

plt.plot(
    [actual.min(), actual.max()],
    [actual.min(), actual.max()]
)

plt.xlabel("Actual Quote Length")
plt.ylabel("Predicted Quote Length")
plt.title("Actual vs Predicted")

plt.show()

residuals = y_test.values - predictions

print("\n=== Residuals ===")
print(residuals)

plt.scatter(predictions, residuals)

plt.axhline(y=0)

plt.xlabel("Predicted Quote Length")
plt.ylabel("Residual")
plt.title("Residual Plot")

plt.show()

evaluation = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2"],
    "Value": [mae, rmse, r2]
})

print("\n=== Model Evaluation Summary ===")
print(evaluation)

print("\n=== Available Features ===")
print(df.columns)

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

print("\n=== Polynomial Features ===")

print("Original X_train shape:", X_train.shape)
print("Polynomial X_train shape:", X_train_poly.shape)

poly_model = LinearRegression()

poly_model.fit(X_train_poly, y_train)

print("\n=== Polynomial Model Training ===")
print("Polynomial model has been trained.")

poly_predictions = poly_model.predict(X_test_poly)

print("\n=== Polynomial Predictions ===")
print(poly_predictions)

poly_mae = mean_absolute_error(y_test, poly_predictions)

poly_rmse = mean_squared_error(
    y_test,
    poly_predictions
) ** 0.5

poly_r2 = r2_score(y_test, poly_predictions)

print("\n=== Polynomial Model Evaluation ===")
print("MAE:", poly_mae)
print("RMSE:", poly_rmse)
print("R2:", poly_r2)

comparison = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2"],
    "Linear Regression": [mae, rmse, r2],
    "Polynomial Regression": [poly_mae, poly_rmse, poly_r2]
})

print("\n=== Model Comparison ===")
print(comparison)

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="r2"
)

print("\n=== Cross-Validation R2 ===")
print(cv_scores)

cv_mean = cv_scores.mean()

print("\n=== Cross-Validation Summary ===")
print("Mean R2:", cv_mean)
print("Std R2:", cv_scores.std())

print("\n=== Number of Authors ===")
print(df["Author"].nunique())

print("\n=== Authors ===")
print(df["Author"].unique())

author_encoded = pd.get_dummies(
    df["Author"],
    prefix="Author"
)

print("\n=== Encoded Author ===")
print(author_encoded.head())

X_author = pd.concat(
    [
        df[["Word_Count"]],
        author_encoded
    ],
    axis=1
)

print("\n=== New Features ===")
print(X_author.head())
print("\nShape:", X_author.shape)

X_author_train, X_author_test, y_author_train, y_author_test = train_test_split(
    X_author,
    y,
    test_size=0.2,
    random_state=42
)

print("\n=== Author Model Train/Test ===")
print("X_train:", X_author_train.shape)
print("X_test:", X_author_test.shape)
print("y_train:", y_author_train.shape)
print("y_test:", y_author_test.shape)

author_model = LinearRegression()

author_model.fit(
    X_author_train,
    y_author_train
)

print("\n=== Author Model Training ===")
print("Author model has been trained.")

author_predictions = author_model.predict(X_author_test)

print("\n=== Author Model Predictions ===")
print(author_predictions)

author_mae = mean_absolute_error(
    y_author_test,
    author_predictions
)

author_rmse = mean_squared_error(
    y_author_test,
    author_predictions
) ** 0.5

author_r2 = r2_score(
    y_author_test,
    author_predictions
)

print("\n=== Author Model Evaluation ===")
print("MAE:", author_mae)
print("RMSE:", author_rmse)
print("R2:", author_r2)

model_summary = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Polynomial Regression",
        "Linear + Author"
    ],
    "MAE": [
        mae,
        poly_mae,
        author_mae
    ],
    "RMSE": [
        rmse,
        poly_rmse,
        author_rmse
    ],
    "R2": [
        r2,
        poly_r2,
        author_r2
    ]
})

print("\n=== Final Model Comparison ===")
print(model_summary.round(4))

plt.figure(figsize=(9, 5))

plt.bar(
    model_summary["Model"],
    model_summary["MAE"]
)

plt.xlabel("Model")
plt.ylabel("MAE")
plt.title("MAE Comparison")
plt.xticks(rotation=15)

plt.show()

plt.figure(figsize=(9, 5))

plt.bar(
    model_summary["Model"],
    model_summary["R2"]
)

plt.xlabel("Model")
plt.ylabel("R2")
plt.title("R2 Comparison")
plt.xticks(rotation=15)

plt.show()