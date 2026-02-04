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

# outcome distribution
# plt.figure()
# plt.hist(df["outcome"], bins=40)
# plt.title("Distribution of outcome")
# plt.xlabel("Outcome")
# plt.ylabel("Frequency")
# plt.show()


# numeric feature distribution
numeric_cols = df.select_dtypes(include="number").columns.drop("outcome")

df[numeric_cols].hist(bins=30, figsize=(12, 8))
plt.suptitle("Numeric Feature distribution")
plt.show()


# pearson correlation
correlation_pearson = df[numeric_cols.tolist() + ["outcome"]].corr(method="pearson")["outcome"]
correlation_pearson = correlation_pearson.sort_values(ascending=False)
print("Pearson correlation:\n", correlation_pearson)


# spearman correlation
correlation_spearman = df[numeric_cols.tolist() + ["outcome"]].corr(method="spearman")["outcome"]
correlation_spearman = correlation_spearman.sort_values(ascending=False)
print("Spearman correlation:\n", correlation_spearman)


# correlation matrix heatmap
# plt.figure(figsize=(12, 10))
# sns.heatmap(
#     df[numeric_cols].corr(),
#     cmap="coolwarm",
#     center=0
# )
# plt.title("Correlation Matrix Heatmap for numeric")
# plt.show()


# categorical features 
categorical_features = ["cut", "color", "clarity"]

for col in categorical_features:
    print(f"\nMean outcome by {col}:")
    print(df.groupby(col)["outcome"].mean().sort_values())


# one hot encoding
categorical_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)


# correlation of encoded features
correlation_encoded = categorical_encoded.corr()["outcome"].sort_values(ascending=False)
print(f"\nMean outcome by {col}:")
print(df.groupby(col)["outcome"].mean().sort_values())


# mutual information

X = df.drop(columns=["outcome"])
y = df["outcome"]

X_num = X.select_dtypes(include="number")

mutual_info = mutual_info_regression(X_num, y, random_state=42)
mutual_info_score = pd.Series(mutual_info, index=X_num.columns).sort_values(ascending=False)

print("\nMutual Information scores:\n", mutual_info_score)

# permutation importance

df_encoded = pd.get_dummies(df, columns=["cut", "color", "clarity"], drop_first=True)


X = df_encoded.drop(columns=["outcome"])
y = df_encoded["outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestRegressor(
    n_estimators=100, 
    random_state=42, 
    n_jobs=-1
)
model.fit(X_train, y_train)


print("Calculating Permutation Importance... (this may take a moment)")
perm = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
)



perm_importance = pd.DataFrame({
    "feature": X.columns,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std
}).sort_values(by="importance_mean", ascending=False)

print("\nTop Feature Importances:")
print(perm_importance.head(20))