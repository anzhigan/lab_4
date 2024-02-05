from matplotlib import pyplot as plt
import os
import conf


def visual(preds, y, train):
    for name, pred in preds.items():
        fig = plt.figure()
        plt.plot(pred, y, "o")

        if train:
            name = f"{name}_train.png"
            img_path = os.path.join(conf.PATH_TO_IMG, name)
        else:
            name = f"{name}_test.png"
            img_path = os.path.join(conf.PATH_TO_IMG, name)

        fig.suptitle(name)
        fig.savefig(img_path)

        plt.show()
