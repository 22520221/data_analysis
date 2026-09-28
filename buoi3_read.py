# ============================================================
# 1. IMPORT LIBRARIES + LOAD DATA
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_csv("quotes_30.csv")


# ============================================================
# 2. ANALYSIS: AUTHORS
# ============================================================

# Count number of quotes for each author
author_counts = df["Author"].value_counts()

print("=== Quote count by author ===")
print(author_counts)


# Top 10 authors
top_authors = author_counts.head(10)

print("\n=== Top 10 authors ===")
print(top_authors)


# Percentage of quotes by author
author_percentage = (author_counts / len(df) * 100).round(2)

print("\n=== Quote percentage by author ===")
print(author_percentage)


# Plot top authors
top_authors.plot(kind="bar")

plt.title("Top 10 Authors by Number of Quotes")
plt.xlabel("Author")
plt.ylabel("Number of Quotes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

# Number of characters in each quote
df["Quote_Length"] = df["Quote"].str.len()


# Number of words in each quote
df["Word_Count"] = df["Quote"].str.split().str.len()


print("\n=== New features ===")
print(df[["Quote", "Quote_Length", "Word_Count"]].head(10))


# ============================================================
# 4. ANALYSIS: QUOTE LENGTH
# ============================================================

# Mean
mean_length = df["Quote_Length"].mean()

# Median
median_length = df["Quote_Length"].median()

print("\n=== Quote Length Statistics ===")
print("Mean:", mean_length)
print("Median:", median_length)


# Full descriptive statistics
print("\n=== Descriptive Statistics ===")
print(df["Quote_Length"].describe())


# Distribution of quote length
df["Quote_Length"].plot(kind="hist", bins=10)

plt.title("Distribution of Quote Length")
plt.xlabel("Quote Length")
plt.ylabel("Number of Quotes")
plt.show()


# ============================================================
# 5. OUTLIER ANALYSIS
# ============================================================

# Calculate Q1, Q3 and IQR
Q1 = df["Quote_Length"].quantile(0.25)
Q3 = df["Quote_Length"].quantile(0.75)

IQR = Q3 - Q1

print("\n=== Outlier Analysis ===")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)


# Calculate upper bound
upper_bound = Q3 + 1.5 * IQR

print("Upper Bound:", upper_bound)


# Find outliers
outliers = df[df["Quote_Length"] > upper_bound]

print("\n=== Outliers ===")
print(outliers[["Author", "Quote_Length"]])


# Compare mean before and after removing outliers
mean_before = df["Quote_Length"].mean()

mean_after = (
    df[df["Quote_Length"] <= upper_bound]["Quote_Length"].mean()
)

print("\n=== Mean comparison ===")
print("Mean before:", mean_before)
print("Mean after:", mean_after)


# ============================================================
# 6. CORRELATION + REGRESSION
# ============================================================

# Calculate correlation
correlation = df["Word_Count"].corr(df["Quote_Length"])

print("\n=== Correlation ===")
print("Correlation:", correlation)


# Scatter plot
plt.scatter(df["Word_Count"], df["Quote_Length"])

plt.title("Word Count vs Quote Length")
plt.xlabel("Word Count")
plt.ylabel("Quote Length")

plt.show()


# Linear regression
x = df["Word_Count"]
y = df["Quote_Length"]

m, b = np.polyfit(x, y, 1)


# Regression equation
print("\n=== Linear Regression ===")
print("m =", m)
print("b =", b)

print(
    f"Equation: Quote_Length = {m:.4f} * Word_Count + {b:.4f}"
)


# Scatter plot + regression line
plt.scatter(x, y)

plt.plot(x, m * x + b)

plt.title("Word Count vs Quote Length with Regression Line")
plt.xlabel("Word Count")
plt.ylabel("Quote Length")

plt.show()


# R-squared
r_squared = correlation ** 2

print("\n=== R-squared ===")
print("R-squared:", r_squared)