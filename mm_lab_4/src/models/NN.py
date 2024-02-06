import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score


class NN(nn.Module):
    def __init__(self, input_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x


def train(X, y):
    X_train_tensor = torch.tensor(X, dtype=torch.float32)
    y_train_tensor = torch.tensor(y.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X, dtype=torch.float32)
    y_test_tensor = torch.tensor(y.values, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_size = X.shape[1]
    model = NN(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    mse_values = []
    r2_values = []

    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions = model(X_test_tensor).squeeze().numpy()

        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)

        # Сохранение значений MSE и R-squared
        mse_values.append(mse)
        r2_values.append(r2)

    return model

    # print("Neural Network:")
    # print(f"Final Mean Squared Error: {mse:.4f}")
    # print(f"Final R-squared: {r2:.4f}")

    # # Визуализация MSE и R-squared по эпохам
    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(range(1, num_epochs + 1), mse_values, label='MSE')
    # plt.xlabel('Epochs')
    # plt.ylabel('MSE')
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(range(1, num_epochs + 1), r2_values, label='R-squared')
    # plt.xlabel('Epochs')
    # plt.ylabel('R-squared')
    # plt.legend()

    # plt.show()
