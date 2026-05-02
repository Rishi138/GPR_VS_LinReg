import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Housing.csv")

def scale(column):
    return [item / max(df[column].values) for item in df[column].values]

prices = scale("price")
areas = scale("area")
bedrooms = scale("bedrooms")

combined = []
for i in range(len(prices)):
    combined.append([prices[i], areas[i], bedrooms[i]])

import random
random.shuffle(combined)

split = int(0.2 * len(combined))
test_set = combined[:split]
train_set = combined[split:]

X_train = np.array([[p[1], p[2]] for p in train_set])
y_train = np.array([p[0] for p in train_set])
X_test = np.array([[p[1], p[2]] for p in test_set])
y_test = np.array([p[0] for p in test_set])

reg = LinearRegression()
reg.fit(X_train, y_train)
preds = reg.predict(X_test)

avg_error_perc = 0
for i in range(len(X_test)):
    error = abs(preds[i] - y_test[i]) / y_test[i]
    avg_error_perc += error
avg_error_perc /= len(X_test)

print(f"Average Percent Error: {int(avg_error_perc*100)}%")