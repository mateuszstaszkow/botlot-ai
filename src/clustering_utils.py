import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets._samples_generator import make_blobs

def cluster_data(data):
    X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.40, random_state=0)
    plt.scatter(X[:, 0], X[:, 1], s=50)
    plt.show()

    data.apply(lambda item: [item.name + '_' + item.arrival.city, item.summary])

    print(X)
    print(y_true)

    print(data)
