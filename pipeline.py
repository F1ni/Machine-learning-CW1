from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("CW1_train.csv")

y = df.iloc[:, 0]
X = df.iloc[:, 1:]
categorical_features = ["cut", "color", "clarity"]
numeric_features = [col for col in X.columns if col not in categorical_features]


numeric_pipeline = Pipeline([# fill missing numeric values
    ("scaler", StandardScaler())                     # scale features to mean=0, std=1
])

categorical_pipeline = Pipeline([ # fill missing categories
    ("encoder", OneHotEncoder())    # convert words to numbers safely
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])


# Split training data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model.fit(X_train, y_train)

# Validate
y_pred = model.predict(X_val)
print("Validation R²:", r2_score(y_val, y_pred))