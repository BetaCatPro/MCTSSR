import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from scipy.io import arff
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from tensorflow.keras import backend
from tensorflow.keras.layers import Layer, Dense, Activation
from tensorflow.keras.initializers import RandomUniform, Initializer, Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential


class InitCentersKMeans(Initializer):
    def __init__(self, X, max_iter=100):
        self.X = X
        self.max_iter = max_iter

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]

        n_centers = shape[0]
        km = KMeans(n_clusters=n_centers, max_iter=self.max_iter, verbose=0)
        km.fit(self.X)
        return km.cluster_centers_


class InitCentersRandom(Initializer):
    def __init__(self, X):
        self.X = X

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]
        idx = np.random.randint(self.X.shape[0], size=shape[0])
        return self.X[idx, :]


class RBFLayer(Layer):
    def __init__(self, output_dim, initializer=None, betas=1.0, **kwargs):
        self.output_dim = output_dim
        self.init_betas = betas
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
        else:
            self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer=Constant(value=self.init_betas),
                                     trainable=True)

        super(RBFLayer, self).build(input_shape)

    def call(self, x):
        C = backend.expand_dims(self.centers)
        H = backend.transpose(C - backend.transpose(x))
        return backend.exp(-self.betas * backend.sum(H ** 2, axis=1))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def init_rbf(X_train):
    model = Sequential()
    rbflayer = RBFLayer(30,
                        initializer=InitCentersKMeans(X_train),
                        betas=2.0,
                        input_shape=(X_train.shape[1],))
    model.add(rbflayer)
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['RootMeanSquaredError'])

    return model


if __name__ == '__main__':
    data, meta = arff.loadarff('../data/space_ga.arff')
    data = pd.DataFrame(data)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=0)
    model = init_rbf(X_train)
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    print(mean_squared_error(model.predict(X_test), y_test.to_numpy().reshape(-1, 1), squared=False))
