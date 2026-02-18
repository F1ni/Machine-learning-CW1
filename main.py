from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score


columns_to_remove = [

    'a6', 'a7', 'a8', 'a9', 'a10',
    'b6', 'b7', 'b8', 'b9', 'b10',

]

df = pd.read_csv("CW1_train.csv")

df = df.drop(columns=columns_to_remove, errors="ignore")

y = df["outcome"]
X = df.drop(columns=['outcome'])
categorical_features = ["cut", "color", "clarity"]

numeric_features = [col for col in X.columns if col not in categorical_features]



numeric_pipeline = Pipeline([
    ("scaler", StandardScaler())         
])

categorical_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# pipeline used to create csv file
final_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(random_state=42, max_depth=3))
])

# all the models that were tested
models = {
    "regressor": LinearRegression(),
    "ridge": Ridge(alpha=1.0),
    "random forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Standard GB (Default)": GradientBoostingRegressor(random_state=42, max_depth=3),
    "GB (Tuned)": GradientBoostingRegressor(learning_rate=0.05, max_depth=4, random_state=42),
}


# print csv file using best model
def create_csv_using_final_model():
    X_tst = pd.read_csv('CW1_test.csv')

    X_tst = X_tst.drop(columns_to_remove, errors="ignore") 

    final_model.fit(X, y)

    yhat_lm = final_model.predict(X_tst)

    # Format submission:
    # This is a single-column CSV with nothing but your predictions
    out = pd.DataFrame({'yhat': yhat_lm})
    out.to_csv('CW1_submission_K23166817.csv', index=False) # Please use your k-number here


# R2 score for all models that I tested
def testing_all_models():
    results = {}

    for type, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])

        crossvalidation_scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=5,
            scoring="r2"
        )

        results[type] = "mean: " + str(crossvalidation_scores.mean()) + "sd: " + str(crossvalidation_scores.std())

    print(results)

# main
testing_all_models()
create_csv_using_final_model()

