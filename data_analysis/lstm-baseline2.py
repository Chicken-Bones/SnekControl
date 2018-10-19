import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

#####################
# Generate data
#####################
np.random.seed(0)

seq_len = 20
noise = 1
coeffs = np.random.randn(seq_len)

num_datapoints = 2000
data_x = np.random.randn(num_datapoints + seq_len)
data_X = np.reshape([data_x[i:i+seq_len] for i in range(num_datapoints)], (-1, seq_len))
data_y = np.dot(data_X, coeffs)
data_y += noise*np.random.randn(*data_y.shape)

#####################
# Set parameters
#####################

#test_size = 0.2
#num_train = int((1-test_size) * num_datapoints)
num_train = num_datapoints

# Network params
h1 = 32
output_dim = 1
num_layers = 1
learning_rate = 1e-2
num_epochs = 3001

#####################
# Generate data
#####################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# make training and test sets in torch
X_train = torch.from_numpy(data_X).type(torch.Tensor)
#X_test = torch.from_numpy(data.X_test).type(torch.Tensor)
y_train = torch.from_numpy(data_y).type(torch.Tensor).view(-1)
#y_test = torch.from_numpy(data.y_test).type(torch.Tensor).view(-1)

X_train = X_train.transpose(0, 1).view([seq_len, -1, 1])
#X_test = X_test.view([input_size, -1, 1])

X_train = X_train.to(device)
y_train = y_train.to(device)
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

        self.hidden = self.init_hidden()

    def init_hidden(self):
        if not hasattr(self, 'lstm'):
            return

        size = (self.lstm.num_layers, self.batch_size, self.lstm.hidden_size)
        return (torch.zeros(*size, device=device),
                torch.zeros(*size, device=device))

    def forward(self, input):
        lstm_out, self.hidden = self.lstm(input, self.hidden)
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1])
        #y_pred = self.linear(input.transpose(0, 1).squeeze(2))
        return y_pred.view(-1)


model = LSTM(1, h1, batch_size=num_train, output_dim=output_dim, num_layers=num_layers)
model = model.to(device)

loss_fn = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

#####################
# Train model
#####################

hist = np.zeros(num_epochs)

for t in range(num_epochs):
    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    model.hidden = model.init_hidden()
    
    # Forward pass
    y_pred = model(X_train)

    loss = loss_fn(y_pred, y_train)
    if t % 100 == 0:
        print("Epoch ", t, "MSE: ", loss.item())
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
plt.show()

plt.semilogy(hist, label="Training loss")
plt.legend()
plt.show()
