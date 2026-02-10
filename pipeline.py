from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
import joblib

columns_to_remove = [

    'a6', 'a7', 'a8', 'a9', 'a10',
    'b6', 'b7', 'b8', 'b9', 'b10',

    # can keep a8 and b10
]

df = pd.read_csv("CW1_train.csv")

df = df.drop(columns=columns_to_remove, errors="ignore")

y = df["outcome"]
X = df.drop(columns=['outcome'])
categorical_features = ["cut", "color", "clarity"]

numeric_features = [col for col in X.columns if col not in categorical_features]



numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())         
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


final_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", HistGradientBoostingRegressor(random_state=42))
])


models = {
    "regressor": LinearRegression(),
    "ridge": Ridge(alpha=1.0),
    "random forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient boosting": HistGradientBoostingRegressor(random_state=42)
}

# create csv file

X_tst = pd.read_csv('CW1_test.csv') 

final_model.fit(X, y)

yhat_lm = final_model.predict(X_tst)

# Format submission:
# This is a single-column CSV with nothing but your predictions
out = pd.DataFrame({'yhat': yhat_lm})
out.to_csv('CW1_submission_K23166817.csv', index=False) # Please use your k-number here



# joblib.dump(final_model, "best_model_gradient_boosting.joblib")
# print("Model saved as 'best_model_gradient_boosting.joblib'")

# results = {}

# for type, model in models.items():
#     pipeline = Pipeline([
#         ("preprocessor", preprocessor),
#         ("regressor", model)
#     ])

#     crossvalidation_scores = cross_val_score(
#         pipeline,
#         X,
#         y,
#         cv=7,
#         scoring="r2"
#     )

#     results[type] = "mean: " + str(crossvalidation_scores.mean()) + "sd: " + str(crossvalidation_scores.std())

# print(results)


# print("Cross validation R2 scores: ", crossvalidation_scores)
# print("Mean R2", crossvalidation_scores.mean())
# print("Std R2", crossvalidation_scores.std())