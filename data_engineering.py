import numpy as np
import pandas as pd
from sklearn import preprocessing


def load_dataset(path):
    return np.array(pd.read_csv(path))


def load_filtered_dataset(path):
    dataset = np.array(pd.read_csv(path))
    ret = []
    for data in dataset:
        if data[-1] > 2:
            ret.append(data)
    return np.array(ret)


def normalize(data):
    return preprocessing.scale(data)


def load_gendata(path):
    gendata = []
    with open(path, "r") as myfile:
        for line in myfile:
            strs = line.split("\t")
            temp = []
            for s in strs:
                temp.append(float(s))
            temp = np.array(temp)
            gendata.append(temp)
    return np.array(gendata)


def load_genlabels(path):
    genlabel = []
    with open(path, "r") as myfile:
        for line in myfile:
            genlabel.append(float(line))
    genlabel = np.array(genlabel)
    genlabel = np.reshape(genlabel, (genlabel.shape[0], 1))
    return genlabel
