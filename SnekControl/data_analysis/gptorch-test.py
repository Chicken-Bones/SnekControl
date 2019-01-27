import math
import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np

# load some test data
num_train = 10000
num_test = 5000
num_total = num_train+num_test
seq_len = 20
vars_in = 2
vars_out = 2
noise = 1

np.random.seed(1)
coeffs = [np.random.randn(seq_len * vars_in) for _ in range(vars_out)]

data = np.genfromtxt('2018-10-24 15-32.csv', dtype=np.float32, delimiter=',', skip_header=1)
data_x = data[:, 1:(vars_in+1)]
data_X = np.concatenate([data_x[i:i+seq_len, None, :] for i in range(num_total)], 1)
data_y = [np.array([np.dot(data_X[:, i].reshape(-1), c) for i in range(num_total)]) for c in coeffs]
data_y = [y + noise*np.random.randn(*y.shape) for y in data_y]

torch_x = torch.from_numpy(data_X[-1].squeeze()).type(torch.Tensor).cuda()
torch_y = torch.stack([torch.from_numpy(y) for y in data_y], -1).type(torch.Tensor).cuda()

# gaussian processes don't work well without rescaling
torch_x /= 20
torch_y /= 20

train_x = torch_x[:num_train]
train_y = torch_y[:num_train, :]
test_x = torch_x[num_train:]
test_y = torch_y[num_train:, :]

#plot test data
plt.subplot(2, 1, 1)
plt.plot(train_x.cpu().numpy())
plt.subplot(2, 1, 2)
for i in range(vars_out):
    plt.plot(train_y[:, i].cpu().numpy(), label="y"+str(i))
plt.legend()
plt.savefig("test_data.png"); plt.close()



# We will use the simplest form of GP model, exact inference
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)

        # SKI requires a grid size hyperparameter. This util can help with that
        grid_size = gpytorch.utils.grid.choose_grid_size(train_x)

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=vars_out
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(), grid_size=grid_size, num_dims=vars_in,
                ), grid_size=grid_size, num_dims=vars_in
            ), num_tasks=vars_out, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=vars_out).cuda()
model = MultitaskGPModel(train_x, train_y, likelihood).cuda()

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

n_iter = 50
for i in range(n_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))
    optimizer.step()

    if loss.item() < -1000:
        break

# Set into eval mode
model.eval()
likelihood.eval()

# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictions = likelihood(model(test_x))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()

# Initialize plots
plt.figure(figsize=(4*vars_out, 4))
for i in range(vars_out):
    plt.subplot(1, vars_out, i+1)
    # Plot training data as black stars
    plt.plot(test_y[:, i].cpu().detach().numpy(), 'k*')
    # Predictive mean as blue line
    plt.plot(mean[:, i].cpu().detach().numpy(), 'b')
    # Shade in confidence
    plt.fill_between(np.arange(num_test),
                     lower[:, i].cpu().detach().numpy(),
                     upper[:, i].cpu().detach().numpy(),
                     alpha=0.5)
    plt.legend(['Observed Data', 'Mean', 'Confidence'])
    plt.title('Observed Values (Likelihood)')

plt.savefig('gptest.png')
plt.close()
None