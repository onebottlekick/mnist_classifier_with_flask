import pickle
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)

with open('./datasets/mnist.pickle', 'wb') as f:
    pickle.dump(mnist, f, pickle.HIGHEST_PROTOCOL)