from sklearn.metrics import mean_squared_error, r2_score


def score(models, X, y):
    for name, model in models.items():
        # Оценка производительности
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)

        # Вывод результатов
        print(f"Model: {name}")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R-squared: {r2:.4f}")
        print("\n" + "=" * 50 + "\n")
