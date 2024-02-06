from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from models.LR import LinearRegressionModel, train_LR
from models import NN

MODELS = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "SVR": SVR(),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "LR": LinearRegressionModel,
    "NN": NN.NN,
}


def train(X_train, y_train):
    res = {}
    for name, model in MODELS.items():
        print(name)
        if name == "LR":
            res[name] = train_LR(X_train, y_train)
        if name == "NN":
            res[name] = NN.train(X_train, y_train)
        else:
            model.fit(X_train, y_train)
            res[name] = model

    return res
