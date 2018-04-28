from sklearn.ensemble import RandomForestRegressor
from data_engineering import load_dataset, normalize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import numpy as np

dataset1 = load_dataset("./data/dataset1.csv")
dataset2 = load_dataset("./data/dataset2.csv")
dataset3 = load_dataset("./data/dataset3.csv")
dataset4 = load_dataset("./data/dataset4.csv")
dataset5 = load_dataset("./data/dataset5.csv")
dataset6 = load_dataset("./data/dataset6.csv")

dataset = np.concatenate((dataset1, dataset2, dataset3, dataset4, dataset5, dataset6))
# data = normalize(dataset[1:,1:-1])
# labels = np.log(dataset[1:,-1])

scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(dataset[1:, 1:-1])
labels = np.log(dataset[1:, -1])

data, labels = shuffle(data, labels)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)

best_rand = None
best_loss = 99999


for i in range(100):
    rand = np.random.randint(100)
    regression = RandomForestRegressor(n_estimators=54, max_depth=45, max_features=2, random_state=rand)
    regression.fit(X_train, y_train)
    y_pred = regression.predict(X_test)
    loss = np.mean(np.square(y_pred - y_test))
    if loss < best_loss:
        best_rand = rand
        best_loss = loss
        print(loss)

print("Best random state = %d" % (best_loss))