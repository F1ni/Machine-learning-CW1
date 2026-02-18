import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import r_regression, mutual_info_regression
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance


df = pd.read_csv("CW1_train.csv")


# basic structure of data
print(df.shape)
print(df.info())
print(df.head())

print(df.describe())

print(df.isnull().sum())


# numeric feature distribution
numeric_cols = df.select_dtypes(include="number").columns.drop("outcome")

df[numeric_cols].hist(bins=30, figsize=(12, 8))
plt.suptitle("Numeric Feature distribution")
plt.tight_layout()
plt.show()


#correlation matrix heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(
    df[numeric_cols].corr().round(2),
    cmap="coolwarm",
    center=0,
    
)
plt.title("Correlation Matrix Heatmap for numeric")
plt.show()



# mutual information

X = df.drop(columns=["outcome"])
y = df["outcome"]

X_num = X.select_dtypes(include="number")

mutual_info = mutual_info_regression(X_num, y, random_state=42)
mutual_info_score = pd.Series(mutual_info, index=X_num.columns).sort_values(ascending=False)

print("\nMutual Information scores:\n", mutual_info_score)
