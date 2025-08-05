# Import required libraries
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs
from typing import Tuple  # For returning multiple types from a function
from random import seed, randint  # For generating random labels reproducibly


# Class to represent a single data point with X, Y coordinates and a class label
class DataPoint:
    def __init__(self, x: float, y: float, label: int):
        self.x = x  # Feature X
        self.y = y  # Feature Y
        self.label = label  # Class label: 0 (Blue), 1 (Red)

    def __repr__(self):
        # This defines how the object will be printed when using print()
        return f"({self.x:.2f}, {self.y:.2f}) -> Class {self.label}"


# Class to generate synthetic 2D data points for 2 different classes
class SyntheticDataGenerator:
    def __init__(self, num_points: int = 20, min_val: int = 1, max_val: int = 10, seed_val: int = 42):
        """
        Initialize the generator with:
        - num_points: total number of data points to generate
        - min_val, max_val: range for X and Y values
        - seed_val: to make results reproducible (same every time)
        """
        self.num_points = num_points
        self.min_val = min_val
        self.max_val = max_val
        self.seed_val = seed_val
        self.data_points = []  # This will hold generated DataPoint objects

    def generate_data(self) -> None:
        """
        Generate 'num_points' synthetic data points with random X, Y values and random class labels.
        """
        np.random.seed(self.seed_val)  # Ensures same random numbers every time
        seed(self.seed_val)

        for _ in range(self.num_points):
            x = np.random.uniform(self.min_val, self.max_val)  # Random X in range
            y = np.random.uniform(self.min_val, self.max_val)  # Random Y in range
            label = randint(0, 1)  # Randomly assign class 0 (Blue) or 1 (Red)
            self.data_points.append(DataPoint(x, y, label))  # Store the generated point

    def get_data_by_class(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separates the generated data points into two arrays by class (0 and 1).
        Returns:
            class0: Numpy array of class 0 points
            class1: Numpy array of class 1 points
        """
        class0 = [(dp.x, dp.y) for dp in self.data_points if dp.label == 0]
        class1 = [(dp.x, dp.y) for dp in self.data_points if dp.label == 1]

        class0 = np.array(class0)
        class1 = np.array(class1)

        return class0, class1


# Class to handle plotting the 2D data points
class DataPlotter:
    def __init__(self, class0: np.ndarray, class1: np.ndarray):
        self.class0 = class0  # Class 0 (Blue) points
        self.class1 = class1  # Class 1 (Red) points

    def plot(self):
        """
        Plots a scatter plot of the data points with colors based on their class.
        """
        plt.figure(figsize=(8, 6))

        # Plot class 0 points (Blue)
        if self.class0.size > 0:
            plt.scatter(self.class0[:, 0], self.class0[:, 1], color='blue', label='Class 0 (Blue)')

        # Plot class 1 points (Red)
        if self.class1.size > 0:
            plt.scatter(self.class1[:, 0], self.class1[:, 1], color='red', label='Class 1 (Red)')

        # Set axis labels and title
        plt.xlabel("Feature X")
        plt.ylabel("Feature Y")
        plt.title("Scatter Plot of Synthetic Training Data")

        # Show legend and grid
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ========= Main Execution (Entry Point) ========= #
if __name__ == "__main__":
    # Step 1: Create a data generator
    generator = SyntheticDataGenerator()

    # Step 2: Generate synthetic data
    generator.generate_data()

    # Step 3: Print the generated data points
    print("Generated Data Points:")
    for point in generator.data_points:
        print(point)

    # Step 4: Separate data points by class
    class0_data, class1_data = generator.get_data_by_class()

    # Step 5: Plot the points using DataPlotter
    plotter = DataPlotter(class0_data, class1_data)
    plotter.plot()
