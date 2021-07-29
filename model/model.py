from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np

with open('datasets/mnist.pickle', 'rb') as f:
    mnist = pickle.load(f)

X = mnist.data.values
y =  mnist.target.to_numpy().astype(int)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)

with open('model/model.pkl', 'wb') as f:
    pickle.dump(rf_clf, f, pickle.HIGHEST_PROTOCOL)