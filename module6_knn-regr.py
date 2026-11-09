import numpy as np


class KNNRegression:
    def __init__(self, n):
        # Data initialization: N rows, column 0 = x, column 1 = y
        self.points = np.zeros((n, 2), dtype=float)
        self.count = 0

    def insert(self, x, y):
        # Data insertion into the preallocated NumPy array
        self.points[self.count] = [x, y]
        self.count += 1

    def predict(self, x, k):
        # Data calculation: distance from X to every training x (1D, so |x - xi|)
        distances = np.abs(self.points[:, 0] - x)
        # Indices of the k smallest distances (stable sort keeps ties in input order)
        nearest = np.argsort(distances, kind="stable")[:k]
        # k-NN regression result is the mean y of the k nearest neighbors
        return np.mean(self.points[nearest, 1])


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


main()
