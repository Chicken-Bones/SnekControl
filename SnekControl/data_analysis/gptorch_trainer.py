import torch
import gpytorch
import numpy as np

# Predominantly https://gpytorch.readthedocs.io/en/latest/examples/05_Scalable_GP_Regression_Multidimensional/KISSGP_Kronecker_Regression.html
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, input_dim, output_dim):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)

        # SKI requires a grid size hyperparameter. This util can help with that
        grid_size = gpytorch.utils.grid.choose_grid_size(train_x)

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=output_dim
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(), grid_size=grid_size, num_dims=input_dim,
                ), grid_size=grid_size, num_dims=input_dim
            ), num_tasks=output_dim, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


def train(logger, dir, device,
          data_x, data_y, baseline_y, input_dim, output_dim,
          train_batches, test_batches, batch_indices, batch_size,
          plot_preds_vs_data):

    # BIG PROBLEM, not enough memory for even a 5000x3 series input on gpu,
    device = torch.device("cpu")
    # BIG PROBLEM, cannot train on whole dataset
    train_batches = train_batches[:1]
    # in fact, even the batch size of 10000 is too much for it, hack it down to 5000
    batch_indices[train_batches[0]] = batch_indices[train_batches[0]][:5000]

    # guassian processes like data to be normalised?
    def normalize(data, range):
        return (data - range[0]) / (range[1] - range[0])
    def unnormalize(data, range):
        return data * (range[1] - range[0]) + range[0]

    range_x = [np.amin(data_x, axis=0), np.amax(data_x, axis=0)]
    range_y = [np.amin(data_y, axis=0), np.amax(data_y, axis=0)]
    data_x = normalize(data_x, range_x)
    data_y = normalize(data_y, range_y)

    # some loss of accuracy due to concatenation across batch boundaries
    train_indices = np.concatenate([batch_indices[b] for b in train_batches])
    test_indices = np.concatenate([batch_indices[b] for b in test_batches])

    train_x = torch.from_numpy(data_x[train_indices]).to(device)
    train_y = torch.from_numpy(data_y[train_indices]).to(device)
    test_x = torch.from_numpy(data_x[test_indices]).to(device)
    # test data needs to be un-normalized for MSE comparison
    test_y = torch.from_numpy(unnormalize(data_y[test_indices], range_y)).to(device)

    # setup the model, and begin training
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=output_dim).to(device)
    model = MultitaskGPModel(train_x, train_y, likelihood, input_dim, output_dim).to(device)

    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # training loop
    n_iter = 50
    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        # it'd be nice to print MSE here but I don't know how to extract it
        logger.info('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))
        optimizer.step()

        # I don't actually know what a loglikelihood of -1000 is (can't be the exponent because that's beyond precision)
        # this condition is here because it's roughly the time that some internal exponents start going to infinity
        # due to lack of noise in the data
        if loss.item() < -1000:
            break

    # Set into eval mode
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        def eval(x, y, log_key):
            preds = likelihood(model(x))
            pred = preds.mean.cpu().detach().numpy()
            pred = unnormalize(pred, range_y)
            mse = np.mean(np.square(pred - y), axis=0)
            mse_total = np.mean(mse)
            logger.info("MSE %s %.4f %s", log_key, mse_total, mse)

            return torch.from_numpy(pred)

        # don't know how to get MSE during iterations (from loss function, so just doing a forward prediction here
        eval(train_x, unnormalize(train_y.cpu().detach().numpy(), range_y), 'Train')

        # test data comparison
        y_pred = eval(test_x, test_y.cpu().detach().numpy(), 'Test')
        y_base = baseline_y[batch_indices[test_batches[0]]]
        plot_preds_vs_data(
            y_pred[:batch_size],
            test_y[:batch_size],
            y_base, dir + "/test.png")