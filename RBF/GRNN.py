import numpy as np
from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from pyGRNN import GRNN

data, meta = arff.loadarff('../data/space_ga.arff')
data = pd.DataFrame(data)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.95)

# Example 1: use Isotropic GRNN with a Grid Search Cross validation to select the optimal bandwidth
# IGRNN = GRNN()
# params_IGRNN = {'kernel': ["RBF"],
#                 'sigma': list(np.arange(0.1, 4, 0.01)),
#                 'calibration': ['None']
#                 }
# grid_IGRNN = GridSearchCV(estimator=IGRNN,
#                           param_grid=params_IGRNN,
#                           scoring='neg_mean_squared_error',
#                           cv=5,
#                           verbose=1
#                           )
# grid_IGRNN.fit(x_train, y_train.ravel())
# best_model = grid_IGRNN.best_estimator_
# y_pred = best_model.predict(x_test)
# mse_IGRNN = mean_squared_error(y_test, y_pred, squared=False)
# print(mse_IGRNN)

# Example 2: use Anisotropic GRNN with Limited-Memory BFGS algorithm to select the optimal bandwidths
AGRNN = GRNN(calibration="gradient_search")
AGRNN.fit(x_train, y_train.ravel())
sigma = AGRNN.sigma
y_pred = AGRNN.predict(x_test)
mse_IGRNN = mean_squared_error(y_test, y_pred, squared=False)
print(mse_IGRNN)
