import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm


class MiniImageNet2DProcessor:
    def __init__(self, data_path, selected_classes: list, num_samples_per_class=10, device='cpu'):
        self.data_path = data_path
        self.selected_classes = selected_classes
        self.num_samples_per_class = num_samples_per_class
        self.device = torch.device(device)
        self.model = resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # Remove classifier
        self.model.eval().to(self.device)
        self.X = []
        self.y = []
        self.class_to_idx = {}

    def _extract_features(self, loader):
        features = []
        labels = []
        with torch.no_grad():
            for images, targets in tqdm(loader, desc="🔍 Extracting features"):
                images = images.to(self.device)
                output = self.model(images).squeeze(-1).squeeze(-1)
                features.append(output.cpu().numpy())
                labels.extend(targets.numpy())
        return np.vstack(features), np.array(labels)

    def load_and_process_data(self):
        transform = transforms.Compose([
            transforms.Resize((84, 84)),
            transforms.ToTensor()
        ])

        dataset = ImageFolder(self.data_path, transform=transform)
        self.class_to_idx = {v: k for k, v in dataset.class_to_idx.items()}

        # Filter for selected classes only
        selected_indices = [idx for idx, (img_path, label) in enumerate(dataset.samples)
                            if dataset.classes[label] in self.selected_classes]

        # Select N samples per class
        selected_data = {cls: [] for cls in self.selected_classes}
        for idx in selected_indices:
            path, label = dataset.samples[idx]
            cls = dataset.classes[label]
            if len(selected_data[cls]) < self.num_samples_per_class:
                selected_data[cls].append((path, label))

        # Flatten and override dataset
        new_samples = [item for sublist in selected_data.values() for item in sublist]
        dataset.samples = new_samples
        dataset.targets = [label for _, label in new_samples]

        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=16, shuffle=False)

        self.X, self.y = self._extract_features(loader)

        # Reduce to 2D
        pca = PCA(n_components=2)
        self.X = pca.fit_transform(self.X)
        return self.X, self.y


class KNNVisualizer:
    def __init__(self, X, y, class_names):
        self.X = X
        self.y = y
        self.class_names = class_names

    def plot_training_data(self):
        plt.figure(figsize=(8, 6))
        for label in np.unique(self.y):
            plt.scatter(self.X[self.y == label, 0], self.X[self.y == label, 1],
                        label=self.class_names[label],
                        c='blue' if label == 0 else 'red', edgecolor='k')
        plt.title("A3: 2D Feature Scatter Plot of 2 miniImageNet Classes")
        plt.xlabel("PCA Feature 1")
        plt.ylabel("PCA Feature 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_decision_boundary(self, k):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(self.X, self.y)

        # Create mesh grid
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        test_points = np.c_[xx.ravel(), yy.ravel()]
        Z = clf.predict(test_points).reshape(xx.shape)

        # Plot decision boundary
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.5)
        for label in np.unique(self.y):
            plt.scatter(self.X[self.y == label, 0], self.X[self.y == label, 1],
                        label=f"{self.class_names[label]}", edgecolor='k',
                        c='blue' if label == 0 else 'red')
        plt.title(f"A4/A5: Decision Boundary (k = {k})")
        plt.xlabel("PCA Feature 1")
        plt.ylabel("PCA Feature 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# =================== MAIN =================== #
if __name__ == "__main__":
    # Set your dataset path here:
    DATA_PATH =  r'C:\yaswanth\BTECH\Sem 5\Machine Learning\Lab\Dataset' # <-- Update this
    SELECTED_CLASSES = ['n01532829', 'n01749939']  # any 2 classes you want

    processor = MiniImageNet2DProcessor(DATA_PATH, SELECTED_CLASSES)
    X, y = processor.load_and_process_data()

    visualizer = KNNVisualizer(X, y, SELECTED_CLASSES)

    # A3
    visualizer.plot_training_data()

    # A4 (k = 3)
    visualizer.plot_decision_boundary(k=3)

    # A5 (various k values)
    for k in [1, 3, 5, 7, 11]:
        visualizer.plot_decision_boundary(k)
