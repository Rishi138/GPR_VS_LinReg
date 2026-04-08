import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np


# Function used, -0.5e^(sin(3x))
def fn(x):
    return -0.5 * np.exp(x) + np.sin(3 * x)


# Data tensors for input and output, input is 26 evenly spaced points form -0.5 to 2
train_x = torch.linspace(-0.5, 2, 26)
train_y = torch.tensor([fn(x) for x in train_x.numpy()])

# GP model
class GPModel(gpytorch.models.ExactGP):
    # likelihood is likelihood output reading is perfect
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        # baseline function for data assuming no inputs
        self.mean_module = gpytorch.means.ConstantMean()
        # scales variance beyond just 0 and 1
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    # Forward pass function
    def forward(self, x):
        # runs x through mean module to create baseline function
        mean_x = self.mean_module(x)
        # runs input through kernel builds covariance matrix
        covar_x = self.covar_module(x)
        # returns distribution, mean (weighted average) and variance (based on variance from other points)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Likelihood function for gaussian noise in GPR
likelihood = gpytorch.likelihoods.GaussianLikelihood()
# Perfect function so set noise to basically 0
likelihood.noise = 1e-4

model = GPModel(train_x, train_y, likelihood)


# training iterations
training_iter = 50


# set model into "training mode" (internal variable adjustments)
model.train()
likelihood.train()

# finetune hyperparameters using adam optimizer, lr: learning rate
# adam vs crude trial, adam uses gradients/momentum from prev cycle
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# "Loss" function for GPR
# Likelihood of set hyperparams for training data
# specifically for gaussian noise
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    # reset hyperparameters
    optimizer.zero_grad()

    # output from model
    output = model(train_x)

    # calculate loss and backprop
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))

    optimizer.step()

# evaluation mode, update to posterior
model.eval()
likelihood.eval()


def test_point(x):
    with torch.no_grad():
        pred = likelihood(model(torch.tensor([x])))
        mu = pred.mean.item()
    return mu


test_x = [-1.0, -0.8, -0.35, -0.1, 0.15, 0.7, 1.1, 1.4, 2.5]
test_y = [fn(x) for x in test_x]
preds = []
for x in test_x:
    preds.append(test_point(x))

percent_errors = [abs(test_y[i] - preds[i]) / abs(test_y[i]) * 100 for i in range(len(test_y))]
avg = int(np.mean(percent_errors))
print("Average percent error (%):", avg)


with torch.no_grad(), gpytorch.settings.fast_pred_var():
    plot_test_x = torch.linspace(-0.5, 2, 130)
    observed_pred = likelihood(model(plot_test_x))

with torch.no_grad():
    # Initialize graph
    fig, ax = plt.subplots(figsize=(10, 6))
    # Get variance
    std = observed_pred.variance.sqrt().numpy()
    mean = observed_pred.mean.numpy()
    # Plot test points posterior mean
    ax.plot(plot_test_x.numpy(), mean, color="red", linewidth=2, label="Predicted (posterior mean)")
    # Plot variance as envelope lines (±σ)
    ax.plot(plot_test_x.numpy(), mean + 1 * std, "b--", alpha=0.8, linewidth=1.5, label=None)
    ax.plot(plot_test_x.numpy(), mean - 1 * std, "b--", alpha=0.8, linewidth=1.5, label=None)
    # Y-limits
    ax.set_ylim([-5, 0.5])
    # Plot actual function
    actual_y = np.array([fn(x) for x in plot_test_x.numpy()])
    ax.plot(plot_test_x.numpy(), actual_y, color="green", linewidth=2, label="Actual function")
    plt.tight_layout()
    plt.legend()
    plt.show()
