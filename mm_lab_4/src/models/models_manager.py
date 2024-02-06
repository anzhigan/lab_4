import os
import joblib
import conf
import torch
from models.LR import LinearRegressionModel
from models.NN import NN


def save_model(models):
    for name, model in models.items():
        model_path = os.path.join(conf.PATH_TO_MODEL, f"{name}.pkl")
        if name == "LR":
            torch.save(
                model.state_dict(), model_path, _use_new_zipfile_serialization=False
            )
        if name == "NN":
            torch.save(
                model.state_dict(), model_path, _use_new_zipfile_serialization=False
            )
        else:
            joblib.dump(model, model_path)


def load_model(X):
    models = {}
    for filename in os.listdir(conf.PATH_TO_MODEL):
        model_path = os.path.join(conf.PATH_TO_MODEL, filename)
        print(model_path)
        if filename.endswith(".pkl"):
            if filename == "LR.pkl":
                X = torch.Tensor(X)
                input_size = X.shape[1]
                model = LinearRegressionModel(input_size)

                model.load_state_dict(torch.load(model_path))
                model.eval()

            if filename == "NN.pkl":
                X = torch.Tensor(X)
                input_size = X.shape[1]
                model = NN(input_size)

                model.load_state_dict(torch.load(model_path))
                model.eval()

            else:
                model = joblib.load(model_path)

            models[filename] = model
            print(f"Модель {filename} успешно загружена.")

    return models
