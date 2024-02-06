from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from models.train_model_LR import LinearRegressionModel, train_LR

MODELS = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "SVR": SVR(),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "LR": LinearRegressionModel,
}


def train(X_train, y_train):
    res = {}
    for name, model in MODELS.items():
        if name == "LR":
            res[name] = train_LR(X_train, y_train)
        else:
            model.fit(X_train, y_train)
            res[name] = model

    return res
