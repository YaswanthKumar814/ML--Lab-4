import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from random import seed, randint
from typing import Tuple, List


class DataPoint:
    def __init__(self, x: float, y: float, label: int):
        self.x = x
        self.y = y
        self.label = label

    def to_features(self):
        return [self.x, self.y]


class SyntheticDataGenerator:
    def __init__(self, num_points: int = 20, min_val: int = 1, max_val: int = 10, seed_val: int = 42):
        self.num_points = num_points
        self.min_val = min_val
        self.max_val = max_val
        self.seed_val = seed_val
        self.data_points: List[DataPoint] = []

    def generate_training_data(self):
        np.random.seed(self.seed_val)
        seed(self.seed_val)
        for _ in range(self.num_points):
            x = np.random.uniform(self.min_val, self.max_val)
            y = np.random.uniform(self.min_val, self.max_val)
            label = randint(0, 1)
            self.data_points.append(DataPoint(x, y, label))

    def get_features_and_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        X = np.array([point.to_features() for point in self.data_points])
        y = np.array([point.label for point in self.data_points])
        return X, y


class TestDataGenerator:
    def __init__(self, min_val: float = 0, max_val: float = 10, step: float = 0.1):
        self.min_val = min_val
        self.max_val = max_val
        self.step = step

    def generate_test_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_vals = np.arange(self.min_val, self.max_val + self.step, self.step)
        y_vals = np.arange(self.min_val, self.max_val + self.step, self.step)
        xx, yy = np.meshgrid(x_vals, y_vals)
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        return grid_points, xx, yy


class KNNVisualizer:
    def __init__(self, knn_model: KNeighborsClassifier, test_points: np.ndarray, xx: np.ndarray, yy: np.ndarray):
        self.knn_model = knn_model
        self.test_points = test_points
        self.xx = xx
        self.yy = yy

    def predict_and_plot(self):
        print("🔍 Classifying test data using kNN (k=3)...")
        predictions = self.knn_model.predict(self.test_points)
        zz = predictions.reshape(self.xx.shape)

        plt.figure(figsize=(10, 8))
        plt.contourf(self.xx, self.yy, zz, cmap=plt.cm.RdBu, alpha=0.5)
        plt.colorbar(label='Predicted Class')

        plt.xlabel("Feature X")
        plt.ylabel("Feature Y")
        plt.title("Test Data Classification using kNN (k=3)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ========== Main Execution ========== #
if __name__ == "__main__":
    # Step 1: Generate training data
    training_data_gen = SyntheticDataGenerator()
    training_data_gen.generate_training_data()
    X_train, y_train = training_data_gen.get_features_and_labels()

    # Step 2: Train kNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Step 3: Generate test grid
    test_data_gen = TestDataGenerator()
    test_points, xx, yy = test_data_gen.generate_test_grid()

    # Step 4: Visualize classification results
    visualizer = KNNVisualizer(knn, test_points, xx, yy)
    visualizer.predict_and_plot()
