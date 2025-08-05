import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from random import seed, randint


class DataPoint:
    def __init__(self, x: float, y: float, label: int):
        self.x = x
        self.y = y
        self.label = label  # 0 for Blue, 1 for Red

    def __repr__(self):
        return f"({self.x:.2f}, {self.y:.2f}) -> Class {self.label}"


class SyntheticDataGenerator:
    def __init__(self, num_points: int = 20, min_val: int = 1, max_val: int = 10, seed_val: int = 42):
        self.num_points = num_points
        self.min_val = min_val
        self.max_val = max_val
        self.seed_val = seed_val
        self.data_points = []

    def generate_data(self) -> None:
        np.random.seed(self.seed_val)
        seed(self.seed_val)
        for _ in range(self.num_points):
            x = np.random.uniform(self.min_val, self.max_val)
            y = np.random.uniform(self.min_val, self.max_val)
            label = randint(0, 1)  # Randomly assign class 0 or 1
            self.data_points.append(DataPoint(x, y, label))

    def get_data_by_class(self) -> Tuple[np.ndarray, np.ndarray]:
        class0 = [(dp.x, dp.y) for dp in self.data_points if dp.label == 0]
        class1 = [(dp.x, dp.y) for dp in self.data_points if dp.label == 1]

        class0 = np.array(class0)
        class1 = np.array(class1)

        return class0, class1


class DataPlotter:
    def __init__(self, class0: np.ndarray, class1: np.ndarray):
        self.class0 = class0
        self.class1 = class1

    def plot(self):
        plt.figure(figsize=(8, 6))
        if self.class0.size > 0:
            plt.scatter(self.class0[:, 0], self.class0[:, 1], color='blue', label='Class 0 (Blue)')
        if self.class1.size > 0:
            plt.scatter(self.class1[:, 0], self.class1[:, 1], color='red', label='Class 1 (Red)')

        plt.xlabel("Feature X")
        plt.ylabel("Feature Y")
        plt.title("Scatter Plot of Synthetic Training Data")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ========= Main Execution ========= #
if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    generator.generate_data()

    print("Generated Data Points:")
    for point in generator.data_points:
        print(point)

    class0_data, class1_data = generator.get_data_by_class()
    plotter = DataPlotter(class0_data, class1_data)
    plotter.plot()
