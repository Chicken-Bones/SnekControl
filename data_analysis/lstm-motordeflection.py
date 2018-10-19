from __future__ import print_function
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#####################
# Load data
#####################
data = np.genfromtxt('2018-10-05 14-07.csv', delimiter=',', skip_header=1)
data = data[300000:700100, :]

#reduce data
data = data[:100100, :]

data_x = data[:, 1:4]
data_y = data[:, 4:7]
baseline_err = data[:, 7:10]
baseline_err -= np.mean(baseline_err, axis=0).reshape(1, -1)
baseline_y = data_y - baseline_err

num_datapoints, input_dim = data_x.shape
output_dim = data_y.shape[1]

# data parameters
seq_len = 100
test_size = 0.2
batch_size = 2000

# test train split and batching
indices = np.arange(seq_len, num_datapoints)
num_datapoints = len(indices)
num_train = int((1-test_size) * num_datapoints)
num_test = num_datapoints - num_train

np.random.seed(0)
test_indices = np.sort(np.random.choice(indices, num_test, replace=False))
train_indices = np.delete(indices, test_indices - seq_len)


def make_data_tensors(indices):
    X = np.concatenate([data_x[i-seq_len:i, None, :] for i in indices], 1)
    X = torch.from_numpy(X).type(torch.Tensor).to(device)
    y = torch.from_numpy(data_y[indices, :]).type(torch.Tensor).to(device)
    return X, y


X_test, y_test = make_data_tensors(test_indices)
print("data loaded ", num_datapoints, "x", input_dim, ", batch size: ", batch_size)
print("train ", num_train, ", test", num_test)
print("baseline MSE ", np.mean(np.square(baseline_err)))


# Network params
hidden_dim = 128
num_layers = 1
learning_rate = 2e-3
num_epochs = 10000

#####################
# Build model
#####################


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1, num_layers=2):
        super(LSTM, self).__init__()

        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        lstm_out, _ = self.lstm(input, None)
        y_pred = self.linear(lstm_out[-1])
        return y_pred


model = LSTM(input_dim, hidden_dim, batch_size, output_dim, num_layers)
model = model.to(device)

loss_fn = torch.nn.MSELoss()
#optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate / 2)
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=500, gamma=0.5)

#####################
# Train model
#####################

hist = np.zeros(num_epochs)

for t in range(num_epochs):
    scheduler.step()
    for batch_start in range(0, num_train, batch_size):
        X_train, y_train = make_data_tensors(train_indices[batch_start:batch_start+batch_size])

        model.train()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        del X_train
        del y_train

    if t % 10 == 0:
        model.eval()
        y_preds = [model(X_test[:, j:j+batch_size, :]) for j in range(0, num_test, batch_size)]
        y_pred = torch.cat(y_preds)

        loss_test = loss_fn(y_pred, y_test)
        print("Epoch ", t, "MSE Train: ", loss.item(), "MSE Test: ", loss_test.item())
        plot_preds_vs_data(y_pred, y_test, "test%d.png" % t)

    hist[t] = loss.item()


plt.plot(hist, label="Training loss")
plt.legend()
plt.savefig("loss.png"); plt.close()


def plot_preds_vs_data(y_pred, y_data, filename, len = 10000):
        plt.figure(figsize=(100, 10 * output_dim))
        y_pred = y_pred.cpu().detach().numpy()[:len]
        y_data = y_data.cpu().detach().numpy()[:len]
        y_baseline = baseline_y[test_indices][:len]
        for p in range(3):
            plt.subplot(output_dim, 1, p + 1)
            plt.plot(y_pred[:, p], label="Preds")
            plt.plot(y_data[:, p], label="Data")
            plt.plot(y_baseline[:, p], label="Baseline")
            plt.legend()
        plt.savefig(filename, dpi=100)
        plt.close()


def export_model(filename):
    dummy_input = torch.linspace(0, 2, steps=seq_len*output_dim, device=device).reshape(seq_len, 1, output_dim)
    torch.onnx.export(model, dummy_input, filename, verbose=True)
    model.eval()
    print(model(dummy_input))