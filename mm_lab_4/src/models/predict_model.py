import torch


def predict(models, X):
    preds = {}

    for name, model in models.items():
        print(name)
        if name == "LR.pkl":
            with torch.no_grad():
                pred = model(torch.Tensor(X))
        else:
            pred = model.predict(X)

        preds[name] = pred
    return preds
