import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def fn(x):
    return -0.5 * np.exp(x) + np.sin(3*x) # Simplified to pure exponential


# Data (unchanged)
inp = [[-0.5], [-0.4], [-0.3], [-0.2], [-0.1], [0.0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [1.0], [1.1], [1.2], [1.3], [1.4], [1.5], [1.6], [1.7], [1.8], [1.9], [2.0]]
y = [fn(x[0]) for x in inp]

x_test = [[-1],[-0.8], [-0.35], [-0.1], [0.15],[0.7], [1.1], [1.4], [2.5]]
y_test = [fn(x[0]) for x in x_test]

X = np.array(inp)
y = np.array(y)

poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

# Test predictions
x_test_array = np.array(x_test)
x_test_poly = poly.transform(x_test_array)
preds = model.predict(x_test_poly)

# Error calculation
percent_errors = [abs(y_test[i] - preds[i]) / abs(y_test[i]) * 100 for i in range(len(y_test))]
avg = int(np.mean(percent_errors))

print("Average percent error (%):", avg)
print(f"Quadratic model: {model.coef_[0]: .3f}x² + {model.coef_[1]: .3f}x + {model.intercept_: .3f}")

# Plotting
x_plot = np.linspace(-1, 2.5, 1000).reshape(-1, 1)
x_plot_poly = poly.transform(x_plot)
y_model = model.predict(x_plot_poly)
y_true = fn(x_plot.ravel())

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_true, 'g-', linewidth=3, label='True: -e^x')
plt.plot(x_plot, y_model, 'r--', linewidth=2.5, label='Quadratic fit (degree=2)')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(f'Exponential vs Quadratic Polynomial\nAvg Test Error: {avg}%')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-5, 0.5)
plt.tight_layout()
plt.show()
