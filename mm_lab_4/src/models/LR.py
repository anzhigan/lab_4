import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


def train_LR(X, y):
    X = torch.Tensor(X)
    y = torch.Tensor(y)

    input_size = X.shape[1]
    model = LinearRegressionModel(input_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adamax(model.parameters(), lr=0.1)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.05)

    num_epochs = 1000

    for epoch in range(num_epochs):
        outputs = model(X)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Learning Rate: {scheduler.get_last_lr()[0]}, Loss: {loss.item():.4f}"
            )

    return model
