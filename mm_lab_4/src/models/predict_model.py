def predict(models, X):
    preds = {}
    for name, model in models.items():
        pred = model.predict(X)
        preds[name] = pred
    return preds
