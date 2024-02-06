from data.make_dataset import make_ds
from visualization.visualize import visual
from models.train_model import train
from models.train_model_LR import train_LR
from models.models_manager import save_model, load_model
from models.metrics import score
from models.predict_model import predict
import conf


def main():
    data = make_ds(conf.PATH_TO_DATA)
    model_ml_based = train(data[0], data[2])
    save_model(model_ml_based)
    models = load_model(data[0])

    is_train = True
    y_train_preds_train = predict(models, data[0])
    score(y_train_preds_train, data[2], is_train)
    visual(y_train_preds_train, data[2], is_train)

    is_train = False
    y_train_preds_test = predict(models, data[1])
    score(y_train_preds_test, data[3], is_train)
    visual(y_train_preds_test, data[3], is_train)


main()
