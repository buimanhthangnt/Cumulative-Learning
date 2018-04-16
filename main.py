import numpy as np
from gan import CL_Gan
from regression import CL_Regression
from sklearn.preprocessing import MinMaxScaler
from data_engineering import load_dataset, normalize, load_gendata, load_genlabels, load_filtered_dataset


dataset1 = load_dataset("./data/dataset1.csv")
dataset2 = load_dataset("./data/dataset2.csv")
dataset3 = load_dataset("./data/dataset3.csv")
dataset4 = load_dataset("./data/dataset4.csv")
dataset5 = load_dataset("./data/dataset5.csv")
dataset6 = load_dataset("./data/dataset6.csv")

dataset = np.concatenate((dataset1, dataset2, dataset3, dataset4, dataset5, dataset6))

scaler = MinMaxScaler(feature_range=(0,1))
filtered_data = dataset[1:, 1:-1]
data = scaler.fit_transform(filtered_data)
filtered_labels = dataset[1:, -1]
labels = np.reshape(np.log(filtered_labels), (-1, 1))


# train Generative Adversarial Network
# GAN = CL_Gan(data)
# GAN.train()
# # generate date using GAN
# GAN.generate_data(size=300)



print("\n\nTrain Regression Network with real data")
Regression = CL_Regression(data, labels, 8000)
Regression.train()


# # load generated data
# gendata = load_gendata("./gen_data/data.txt")
# # generate label for generated data using trained Regression
# Regression.generate_labels(gendata)
# # load generated labels for generated data
# genlabels = load_genlabels("./gen_data/labels.txt")



# print("\n\nTrain another Regression Network using generated data and labels")
# Regression2 = CL_Regression(gendata, genlabels, 800)
# Regression2.train()


# print("\n\nTest model using real data")
# Regression2.test(data, labels)
