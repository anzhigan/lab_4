from sklearn.metrics import mean_squared_error, r2_score


def score(preds, y, is_train):
    if is_train:
        print("TRAIN")
    else:
        print("TEST")

    for name, pred in preds.items():
        mse = mean_squared_error(y, pred)
        r2 = r2_score(y, pred)

        # Вывод результатов
        print(f"Model: {name}")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R-squared: {r2:.4f}")
        print("\n" + "=" * 50 + "\n")
