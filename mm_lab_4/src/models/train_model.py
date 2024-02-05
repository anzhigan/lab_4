from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

MODELS = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "SVR": SVR(),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
}


def train(X_train, y_train):
    res = {}
    for name, model in MODELS.items():

        model.fit(X_train, y_train)
        res[name] = model

    return res
