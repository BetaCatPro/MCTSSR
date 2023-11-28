import copy

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from neupy import algorithms

data, meta = arff.loadarff('../data/space_ga.arff')
data = pd.DataFrame(data)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.95)

nw = algorithms.GRNN(std=0.1, verbose=False)
nw.train(x_train, y_train)

y_predicted = nw.predict(x_test)
r_mse = mean_squared_error(y_predicted, y_test, squared=False)

print(r_mse)
