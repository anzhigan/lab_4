import os
import joblib
import conf


def save_model(models):
    for name, model in models.items():
        model_path = os.path.join(conf.PATH_TO_MODEL, f"{name}.pkl")
        joblib.dump(model, model_path)


def load_model(path):
    models = {}
    for filename in os.listdir(path):
        if filename.endswith(".pkl"):
            model_path = os.path.join(path, filename)
            loaded_model = joblib.load(model_path)
            models[filename] = loaded_model
            print(f"Модель {filename} успешно загружена.")

    return models
