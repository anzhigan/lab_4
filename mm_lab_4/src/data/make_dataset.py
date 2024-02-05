import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def make_ds(PATH):
    df = pd.read_csv(PATH)
    df.dropna(inplace=True)

    X = df.drop("crossplane", axis=1)
    y = df["crossplane"]

    categorical_features = ["y", "x", "temp"]
    numeric_features = ["thick", "r"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(), categorical_features),
        ]
    )

    pipeline = Pipeline([("preprocessor", preprocessor)])

    X_processed = pipeline.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )
    return (X_train, X_test, y_train, y_test)
