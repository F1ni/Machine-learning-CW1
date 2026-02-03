import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns


# we want to predict "outcome" column

df = pd.read_csv("CW1_train.csv")

# print(df.head())
# print(df.info())
# print("---------------------------------------------")
# print(df.describe())
# print("---------------------------------------------")
# print(df.shape)

# print(df.duplicated().sum()) - no duplicate values

# data preprocessing 
# missing values - median imputation
# feature scaling 



y = df.iloc[:, 0]
X = df.iloc[:, 1:]


# numeric_df = df.select_dtypes(include="number")

# corr_matrix = numeric_df.corr()
# corr_with_target = numeric_df.corr()["outcome"].sort_values(ascending=False)
# print(corr_with_target)

# print(corr_matrix)

trn = pd.get_dummies(df, columns=['cut', 'color', 'clarity'], drop_first=True)

corr_outcome = (
    trn.corr()['outcome']
       .sort_values(ascending=False)
)

print(corr_outcome)


# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
# plt.title("Correlation Matrix of Numeric Features")
# plt.show()

# plt.figure(figsize=(6, 10))
# sns.heatmap(
#     corr_with_target.to_frame(),
#     annot=True,
#     cmap="coolwarm",
#     center=0
# )
# plt.title("Correlation with Outcome")
# plt.show()

# distribution of outcome
# plt.hist(df[4], bins=30)
# plt.title("Distribution of Outcome")
# plt.show()

# numeric_cols = df.select_dtypes(include="number").columns


# corr = df[numeric_cols].corr()["outcome"].sort_values(ascending=False)
# print(corr)

# for cat in ["cut","color","clarity"]:
#     print(df.groupby(cat)["outcome"].mean())