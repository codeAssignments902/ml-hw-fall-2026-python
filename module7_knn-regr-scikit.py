import numpy as np
from sklearn.neighbors import KNeighborsRegressor


class KNNRegression:
    def __init__(self, n):
        # Data initialization (NumPy): N rows, column 0 = x, column 1 = y
        self.points = np.zeros((n, 2), dtype=float)
        self.count = 0

    def insert(self, x, y):
        # Data insertion into the preallocated NumPy array
        self.points[self.count] = [x, y]
        self.count += 1

    def features(self):
        # Scikit-learn expects a 2D feature matrix of shape (n_samples, n_features)
        return self.points[:, 0].reshape(-1, 1)

    def labels(self):
        return self.points[:, 1]

    def predict(self, x, k):
        # ML computation (Scikit-learn): fit k-NN regressor and predict Y for X
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(self.features(), self.labels())
        return float(model.predict(np.array([[x]]))[0])

    def label_variance(self):
        # Variance of the training labels (population variance, mean of squared deviations)
        return float(np.var(self.labels()))


def read_positive_int(prompt):
    while True:
        try:
            value = int(input(prompt))
            if value > 0:
                return value
        except ValueError:
            pass
        print("Please enter a positive integer.")


def read_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a real number.")


def main():
    n = read_positive_int("Enter N: ")
    k = read_positive_int("Enter k: ")

    model = KNNRegression(n)
    for i in range(n):
        x = read_float(f"Enter x value for point {i + 1}: ")
        y = read_float(f"Enter y value for point {i + 1}: ")
        model.insert(x, y)

    x = read_float("Enter X: ")

    if k <= n:
        print(f"Result (Y): {model.predict(x, k)}")
    else:
        print(f"Error: k ({k}) must be less than or equal to N ({n}).")

    print(f"Variance of labels in the training dataset: {model.label_variance()}")


main()
