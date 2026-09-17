from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


mnist = fetch_openml('mnist_784')

d = mnist.data.iloc[3]

print(d)