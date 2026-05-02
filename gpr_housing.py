import pandas as pd
import torch
import gpytorch
import random
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("Housing.csv")

# price, area, bedrooms
# two feature analysis


def scale(column):
    scaled = []
    for item in df[column].values:
        scaled.append(item/max(df[column].values))
    return scaled



prices = scale("price")
areas = scale("area")
bedrooms = scale("bedrooms")

combined = []
for indx in range(len(prices)):
    indv = [prices[indx],areas[indx],bedrooms[indx]]
    combined.append(indv)

random.shuffle(combined)

split = int(0.2 * len(combined))
test_set = combined[:split]
train_set = combined[split:]

test_x = [[indv[1], indv[2]] for indv in test_set]
test_y = [indv[0] for indv in test_set]

train_x = [[indv[1], indv[2]] for indv in train_set]
train_y = [indv[0] for indv in train_set]


train_x = torch.tensor(train_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32)
test_x = torch.tensor(test_x, dtype=torch.float32)
test_y = torch.tensor(test_y, dtype=torch.float32)


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x,train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
likelihood.noise = 0.1
model = GPModel(train_x, train_y, likelihood)
training_iter = 300

model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.15)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

last_five = []
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))

    if len(last_five) < 5:
        last_five.append(model.covar_module.base_kernel.lengthscale.item())
    else:
        avg = 0
        for ls in last_five:
            avg+=ls
        avg = avg/5
        avg_dev = 0
        for ls in last_five:
            dev = abs(avg-ls)
            avg_dev += dev
        avg_dev = avg_dev/5
        last_five.pop(0)
        last_five.append(model.covar_module.base_kernel.lengthscale.item())
        if avg_dev <= 0.001:
            print("Convergence detected, early stopping.")
            break


    optimizer.step()

model.eval()
likelihood.eval()


def test_point(x):
    with torch.no_grad():
        pred = likelihood(model(x.unsqueeze(0)))
        mu = pred.mean.item()
        variance = pred.variance.item()
    return mu, variance


print(test_point(test_x[0]))
print(test_y[0])

avg_error_perc = 0
avg_var = 0
avg_var_perc = 0
not_in_var = 0
for indx in range(len(test_x)):
    x = test_x[indx]
    y = test_y[indx]
    pred, var = test_point(x)
    error = abs(pred-y)/y
    avg_error_perc += error
    avg_var += var

    var_perc = var/pred
    avg_var_perc += var_perc

    not_in = abs(error-var_perc)
    not_in_var += not_in


avg_var = avg_var/len(test_x)
avg_error_perc = avg_error_perc/len(test_x)
avg_var_perc = avg_var_perc/len(test_x)
not_in_var = not_in_var/len(test_x)

print(f"Average Percent Error: {int(avg_error_perc*100)}%")
print(f"Average Percent Variance: {int(avg_var_perc*100)}%")
print(f"Average Percent Error Not In Variance: {int(not_in_var*100)}%")
print(f"Average Variance: ${int(avg_var*max(df['price'].values))}")


# build grid
resolution = 150
area_range = np.linspace(0, 1, resolution)
bedroom_range = np.linspace(0, 1, resolution)

var_grid = np.zeros((resolution, resolution))

for i, bed in enumerate(bedroom_range):
    for j, area in enumerate(area_range):
        x = torch.tensor([[area, bed]], dtype=torch.float32)
        with torch.no_grad():
            pred = likelihood(model(x))
            var_grid[i, j] = pred.variance.item()

fig, ax = plt.subplots(figsize=(7, 5))

im = ax.imshow(var_grid, origin='lower', aspect='auto',
               extent=[0, 1, 0, 1], cmap='RdYlGn_r')
ax.scatter(train_x[:,0].numpy(), train_x[:,1].numpy(),
           c='white', s=10, alpha=0.5)
plt.colorbar(im, ax=ax)
ax.set_title('Uncertainty (σ²)')
ax.set_xlabel('Area')
ax.set_ylabel('Bedrooms')

plt.savefig("heatmap.png")
plt.show()
