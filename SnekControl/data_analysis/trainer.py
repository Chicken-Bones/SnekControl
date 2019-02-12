from __future__ import print_function, division
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import logging
import sys
import argparse

parser = argparse.ArgumentParser(description='Tension Estimator Trainer.')
parser.add_argument('--seq_len', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=20000)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--lstm', dest='model', action='store_const', const='LSTM')
parser.add_argument('--gru', dest='model', action='store_const', const='GRU')
parser.add_argument('--sigmoid', dest='model', action='store_const', const='Sigmoid')
parser.add_argument('--encoder', dest='model', action='store_const', const='Encoder')


args = parser.parse_args()

#####################
# Logging and dir
#####################

while True:
    try:
        dir = 'run'+datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        os.mkdir(dir)
        break
    except FileExistsError:
        pass

logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] %(message)s', '%H:%M:%S')

handler = logging.FileHandler(dir+'/log.txt')
handler.setFormatter(formatter)
logger.addHandler(handler)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info('Args: %s', args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info('Device: %s', device)

#####################
# Load data
#####################
seq_len = args.seq_len
test_size = 0.2
batch_size = args.batch_size

data = None
indices = None


def add_dataset(name):
    global data, indices
    set_data = np.genfromtxt(name, dtype=np.float32, delimiter=',', skip_header=1)
    set_inds = np.arange(seq_len, set_data.shape[0])
    if data is None:
        data = set_data
        indices = set_inds
    else:
        indices = np.append(indices, data.shape[0] + set_inds)
        data = np.concatenate((data, set_data))

    logger.info("loaded %s (%d points)", name, set_data.shape[0])


add_dataset('2019-01-27 11-55_clean.csv')

data_x = data[:, 1:4]
data_y = data[:, 4:7]
#baseline_err = data[:, 7:10]
#baseline_err -= np.mean(baseline_err, axis=0).reshape(1, -1)
#baseline_y = data_y - baseline_err

input_dim = data_x.shape[1]
output_dim = data_y.shape[1]

# test train split and batching
num_datapoints = len(indices)
num_batches = num_datapoints // batch_size
num_train_batches = int((1-test_size) * num_batches)
num_test_batches = num_batches - num_train_batches

np.random.seed(0)
test_batches = np.sort(np.random.choice(range(num_batches), num_test_batches, replace=False))
train_batches = np.delete(range(num_batches), test_batches)
batch_indices = [indices[batch*batch_size:(batch+1)*batch_size] for batch in range(num_batches)]

logger.info("train %d test %d", num_train_batches * batch_size, num_test_batches * batch_size)
#logger.info("baseline MSE %.4f", np.mean(np.square(baseline_err)))


def plot_preds_vs_data(y_pred, y_data, filename):
    plt.figure(figsize=(batch_size / 100, 10 * output_dim))
    y_pred = y_pred.cpu().detach().numpy()
    y_data = y_data.cpu().detach().numpy()
    for p in range(3):
        plt.subplot(output_dim, 1, p + 1)
        plt.plot(y_pred[:, p], label="Preds")
        plt.plot(y_data[:, p], label="Data")
        plt.legend()
    plt.savefig(filename, dpi=100)
    plt.close()


#####################
# Gaussian Process
#####################
if args.model == 'GP':
    import gptorch_trainer
    gptorch_trainer.train(
        logger, dir, device,
        data_x, data_y, baseline_y, input_dim, output_dim,
        train_batches, test_batches, batch_indices, batch_size,
        plot_preds_vs_data)
    exit()

#####################
# Define model
#####################


class TakeLastHidden(nn.Module):
    def forward(self, input: torch.Tensor):
        h, c = input
        return h[-1]


class GLU(nn.Module):
    def forward(self, input: torch.Tensor):
        return nn.functional.glu(input, dim=1)


layers = []
if args.model == 'LSTM' or args.model == 'GRU':
    prepare_shape = lambda X: X.transpose([1, 0, 2])
    step_size = 500

    layers = [
        nn.LSTM(input_dim, args.hidden_dim) if args.model == 'LSTM' else
        nn.GRU(input_dim, args.hidden_dim),
        TakeLastHidden(),
        nn.Linear(args.hidden_dim, output_dim)
    ]

if args.model == 'Sigmoid':
    prepare_shape = lambda X: X.reshape(X.shape[0], -1)
    step_size = 200

    layers = [
        nn.Linear(input_dim*seq_len, args.hidden_dim),
        nn.Sigmoid(),
        nn.Linear(args.hidden_dim, output_dim)
    ]

if args.model == 'Encoder':
    prepare_shape = lambda X: X.transpose([0, 2, 1])
    step_size = 200

    features = args.hidden_dim
    stack_size = 10
    kernel_size = 3
    features_in = input_dim
    for i in range(stack_size):
        layers.extend([
            nn.Conv1d(features_in, features * 2, 3, padding=1),
            GLU()
        ])
        features_in = features

    layers.append(nn.Linear(features, output_dim))

if args.dropout != 0:
    layers.insert(-1, nn.Dropout(p=args.dropout))

model = nn.Sequential(*layers)
model = model.to(device)

loss_fn = torch.nn.MSELoss()
optimiser = torch.optim.SGD(model.parameters(), lr=5e-3, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiser, gamma=0.5, milestones=
        [step_size, 2*step_size, 3*step_size, 3.2*step_size, 3.4*step_size, 3.6*step_size, 3.8*step_size, 4.0*step_size])
num_epochs = int(step_size*4.2)

logger.info(model)
logger.info(optimiser)
logger.info("MultiStepLR(gamma=%.2f)", scheduler.gamma)


# if not flatten (seq_len, batch, input_dim)
# if flatten (batch, seq_len*input_dim)
# linear input is repeated input_dim for each timestep, [[t_0] [t_1] ...]
def make_batch_tensors(batch):
    inds = batch_indices[batch]
    X = np.stack([data_x[i-seq_len:i, :] for i in inds])
    X = prepare_shape(X)  # from (batch_size, seq_len, input_dim) -> network suitable
    X = torch.from_numpy(X).to(device)
    y = torch.from_numpy(data_y[inds, :]).to(device)
    return X, y


def export_model(filename, verbose=False):
    dummy_input = np.linspace(0, 2, num=seq_len*output_dim, dtype=np.float32)\
        .reshape(1, seq_len, output_dim)
    dummy_input = prepare_shape(dummy_input)
    dummy_input = torch.from_numpy(dummy_input).to(device)

    torch.onnx.export(model, dummy_input, filename, verbose=verbose)
    model.eval()
    result = model(dummy_input).cpu().detach().numpy()

    # check the model
    dummy_input = dummy_input.cpu().detach().numpy()
    import onnx
    import caffe2.python.onnx.backend as caffe_backend
    onnx_model = onnx.load(filename)
    result2 = caffe_backend.run_model(onnx_model, [dummy_input])[0]
    if not np.isclose(result, result2).all():
        raise Exception("model is not consistent: {} {}".format(result, result2))

    onnx_model = onnx.utils.polish_model(onnx_model)
    onnx.save(onnx_model, filename.replace(".onnx", "_opt.onnx"))


#####################
# Train model
#####################

hist_train = []
hist_test = []
def plot_loss_hist():
    plt.semilogy(hist_train, label="Training loss")
    plt.semilogy(hist_test, label="Test loss")
    plt.legend()
    plt.savefig(dir + "/loss.png")
    plt.close()


def dump_preds():
    dump = np.zeros((num_batches * batch_size, 1 + input_dim + 2 * output_dim))
    for batch in range(0, num_batches):
        inds = batch_indices[batch]
        X, _ = make_batch_tensors(batch)
        y_pred = model(X).cpu().detach().numpy()

        j = batch * batch_size
        dump[j:j+batch_size, 0] = data[inds, 0]
        dump[j:j+batch_size, 1:4] = data_x[inds, :]
        dump[j:j+batch_size, 4:7] = data_y[inds, :]
        dump[j:j+batch_size, 7:10] = y_pred

    logger.info("saving eval.csv")
    np.savetxt(dir+'/eval.csv', dump, fmt='%.4f', delimiter=', ')


for t in range(num_epochs+1):
    scheduler.step()
    model.train()
    for batch in train_batches:
        optimiser.zero_grad()
        X_train, y_train = make_batch_tensors(batch)
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimiser.step()

    if t % 10 != 0:
        continue

    model.eval()
    with torch.no_grad():
        def eval(batches):
            batch_mse = []
            for batch in batches:
                X, y = make_batch_tensors(batch)
                y_pred = model(X)
                batch_mse.append(torch.mean((y - y_pred)**2, 0))
            mse = torch.mean(torch.stack(batch_mse), 0)
            return mse.cpu().detach().numpy(), float(torch.mean(mse))

        _, mse_train_total = eval(train_batches)
        mse_test, mse_test_total = eval(test_batches)

        np.set_printoptions(precision=4)
        logger.info("Epoch %d MSE Train: %.4f, Test %.4f %s", t, mse_train_total, mse_test_total, mse_test)
        export_model(dir+'/model.onnx')

        # update histogram
        hist_train.append(mse_train_total)
        hist_test.append(mse_test_total)
        plot_loss_hist()

        if t < 100 or t % (step_size//5) == 0:
            X_test, y_test = make_batch_tensors(test_batches[0])
            y_pred = model(X_test)
            plot_preds_vs_data(y_pred, y_test, dir+"/test%d.png" % t)

        if t % (step_size//5) == 0:
            dump_preds()


