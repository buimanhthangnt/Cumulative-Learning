import numpy as np
from gan import CL_Gan
from regression import CL_Regression
from data_engineering import load_dataset, normalize, load_gendata, load_genlabels, load_filtered_dataset


dataset = load_dataset("./data/dataset1.csv")
filtered_data = dataset[1:, 1:-1]
data = normalize(filtered_data)
filtered_labels = dataset[1:, -1]
labels = np.reshape(np.log(filtered_labels), (filtered_labels.shape[0], 1))

GAN = CL_Gan(data)
GAN.train()
GAN.generate_data(size=300)

Regression = CL_Regression(data, labels, 5000)
Regression.train()

gendata = load_gendata("./gen_data/data.txt")
Regression.generate_labels(gendata)
genlabels = load_genlabels("./gen_data/labels.txt")

Regression2 = CL_Regression(gendata, genlabels, 800)
Regression2.train()
Regression2.test(data, labels)
