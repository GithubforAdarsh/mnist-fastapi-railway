from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
import pickle

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]

clf = LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=-1)
clf.fit(X, y)

with open("mnist_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Model saved successfully!")
