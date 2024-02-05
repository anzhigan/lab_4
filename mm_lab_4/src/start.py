from data.make_dataset import make_ds
from models.train_model import train
from models.models_manager import save_model, load_model
import conf
from models.metrics import score


def main():
    data = make_ds(conf.PATH_TO_DATA)
    train_res = train(data[0], data[2])
    save_model(train_res)
    models = load_model(conf.PATH_TO_MODEL)
    score(models, data[0], data[2])


main()
