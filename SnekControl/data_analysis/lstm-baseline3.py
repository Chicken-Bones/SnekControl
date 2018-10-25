from __future__ import print_function
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

#####################
# Generate data
#####################
np.random.seed(0)

seq_len = 20
vars = 3
noise = 0
coeffs = np.random.randn(seq_len * vars)

num_datapoints = 4000
# (n, vars)
data_x = np.random.randn(num_datapoints + seq_len, vars)
# (n_seq, batch_size, 3)
data_X = np.concatenate([data_x[i:i+seq_len, None, :] for i in range(num_datapoints)], 1)
data_y = np.array([np.dot(data_X[:, i].reshape(-1), coeffs) for i in range(num_datapoints)])
data_y += noise*np.random.randn(*data_y.shape)

#####################
# Set parameters
#####################

test_size = 0.2
num_train = int((1-test_size) * num_datapoints)

# Network params
h1 = 32
output_dim = 1
num_layers = 1
learning_rate = 1e-2
num_epochs = 6001

#####################
# Generate data
#####################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# make training and test sets in torch
X_train = torch.from_numpy(data_X[:, :num_train, :]).type(torch.Tensor).to(device)
X_test = torch.from_numpy(data_X[:, num_train:, :]).type(torch.Tensor).to(device)
y_train = torch.from_numpy(data_y[:num_train]).type(torch.Tensor).to(device)
y_test = torch.from_numpy(data_y[num_train:]).type(torch.Tensor).to(device)

#####################
# Build model
#####################


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1, num_layers=2):
        super(LSTM, self).__init__()

        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.linear = nn.Linear(hidden_dim, output_dim)
        #self.linear = nn.Linear(seq_len, output_dim)

    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        y_pred = self.linear(lstm_out[-1])
        #y_pred = self.linear(input.transpose(0, 1).squeeze(2))
        return y_pred.view(-1)


model = LSTM(input_dim=vars, hidden_dim=h1, batch_size=num_train, output_dim=output_dim, num_layers=num_layers)
model = model.to(device)

loss_fn = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate / 2)
#optimiser = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)

#####################
# Train model
#####################

hist = np.zeros(num_epochs)

for t in range(num_epochs):
    # Forward pass
    y_pred = model(X_train)

    loss = loss_fn(y_pred, y_train)
    if t % 100 == 0:
        loss_test = loss_fn(model(X_test), y_test)
        print("Epoch ", t, "MSE: ", loss.item(), "Test: ", loss_test.item())
    hist[t] = loss.item()

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

#####################
# Plot preds and performance
#####################

plt.plot(y_pred.cpu().detach().numpy(), label="Preds")
plt.plot(y_train.cpu().detach().numpy(), label="Data")
plt.legend()
plt.rcParams["figure.figsize"] = [80, 10]
plt.savefig("fig1.png"); plt.close()

plt.plot(hist, label="Training loss")
plt.legend()
plt.savefig("fig2.png"); plt.close()

