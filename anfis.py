import os
import pickle
import numpy as np
import pandas as pd
import json
import argparse
from skcmeans import algorithms
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.linear_model import HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from neupy import environment
from neupy import algorithms as neual
from neupy import layers
from neupy import plots
from datetime import datetime
from data_engineering import load_dataset, normalize, load_gendata, load_genlabels, load_filtered_dataset

# THEANO_FLAGS="device=cuda0,exception_verbosity=high" MKL_THREADING_LAYER=GNU python generate.py


def build_net(n_input, activation=layers.Sigmoid, sizes=[3, 3]):
    net = layers.Input(n_input)
    for size in sizes:
        net = net > activation(size)
    net = net > layers.Linear(1)

    conj = neual.ConjugateGradient(
        connection=net,
        step=0.005,
        addons=[neual.LinearSearch],
        show_epoch=25
    )
    return conj


class Anfis(object):
    def __init__(self, name, model1, model2, *args, **kwargs):
        self.name = name
        self.model1 = model1
        self.model2 = model2

    def fit(self, data, target, *args, **kwargs):
        target = target.reshape((-1, 1))
        self.data_scale = MinMaxScaler()
        self.target_scale = MinMaxScaler()
        self.new_input_scale = MinMaxScaler()
        self.data_cluster = algorithms.Probabilistic(n_clusters=4)
        self.target_cluster = algorithms.Probabilistic(n_clusters=6)
        self.data_scale.fit(data)
        self.target_scale.fit(target.reshape(-1, 1))
        self.data_cluster.fit(data)
        self.target_cluster.fit(target.reshape(-1, 1))

        data = self.data_scale.transform(data)
        target = np.log10(target)

        feature_from_data = np.argmin(
            self.data_cluster.distances(data), axis=1).reshape((-1, 1))
        feature_from_target = np.argmin(
            self.target_cluster.distances(target), axis=1).reshape((-1, 1))
        new_input = np.concatenate((self.data_scale.inverse_transform(
            data), feature_from_data, feature_from_target), axis=1)
        self.new_input_scale.fit(new_input)
        new_input = self.new_input_scale.transform(new_input)

        self.model1.fit(data, target)

        x_train, x_test, y_train, y_test = train_test_split(
            new_input, target,
            test_size=0.5,
            shuffle=False
        )
        self.model2.fit(
            x_train, y_train,
            x_test, y_test,
            epochs=1000
        )
        # plots.error_plot(self.model2)

    def predict(self, x, *args, **kwargs):
        x = self.data_scale.transform(x)
        y_pseudo = self.model1.predict(x)
        y_pseudo = y_pseudo.reshape((-1, 1))
        feature_from_data = np.argmin(
            self.data_cluster.distances(x), axis=1).reshape((-1, 1))
        feature_from_target = np.argmin(
            self.target_cluster.distances(y_pseudo), axis=1).reshape((-1, 1))
        new_input = np.concatenate((self.data_scale.inverse_transform(
            x), feature_from_data, feature_from_target), axis=1)
        new_input = self.new_input_scale.transform(new_input)
        y_pred = self.model2.predict(new_input)

        return (10**y_pred).reshape(-1)

    def eval(self, x_test, y_test, *args, **kwargs):
        p = self.predict(x_test)
        e = mse(y_test, p)
        return e


def Main(*args, **kwargs):
    dataset1 = load_dataset("./data/dataset1.csv")
    dataset2 = load_dataset("./data/dataset2.csv")
    dataset3 = load_dataset("./data/dataset3.csv")
    dataset4 = load_dataset("./data/dataset4.csv")
    dataset5 = load_dataset("./data/dataset5.csv")
    dataset6 = load_dataset("./data/dataset6.csv")

    df_train = np.concatenate((dataset1, dataset2, dataset3, dataset4, dataset5, dataset6))[1:,1:]
    # df_train = pd.read_csv('temp/train.csv')
    x_train, x_test, y_train, y_test = train_test_split(
        df_train.values[:, :-1], df_train.values[:, -1],
        test_size=0.2
    )
    net = build_net(n_input=8)

    # models = [SVR, DecisionTreeRegressor, RandomForestRegressor,
              # Lasso, LinearRegression, HuberRegressor]
    models = [build_net(n_input=8, sizes=[5,5,5])]

    result = []
    for model in models:
        name = 'ConjugateGradient'
        anfis = Anfis(
            name=name,
            model1=RandomForestRegressor(),
            model2=model
        )
        anfis.fit(x_train, y_train)
        e = anfis.eval(x_test, y_test)
        filename = '{}.pickle'.format(name)
        print('Complete {} - {}'.format(name, e))
        result.append({
            'name': name,
            'datetime': datetime.now().__str__().split('.')[0],
            'filename': filename,
            'mse': e
        })
        with open(os.path.join('temp', filename), 'wb') as f:
            pickle.dump(anfis, f)
    with open('models.json', 'w') as f:
        json.dump({'anfis': result}, f)

    print('Complete')


if __name__ == '__main__':
    Main()