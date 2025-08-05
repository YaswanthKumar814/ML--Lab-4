import numpy as np
import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class FeatureExtractor:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.model = resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval().to(self.device)

    def extract_features(self, dataloader):
        features = []
        labels = []
        with torch.no_grad():
            for imgs, targets in tqdm(dataloader, desc="🔍 Extracting Features"):
                imgs = imgs.to(self.device)
                outputs = self.model(imgs).squeeze()
                features.append(outputs.cpu().numpy())
                labels.extend(targets.numpy())
        return np.vstack(features), np.array(labels)


class MiniImageNetDataLoader:
    def __init__(self, data_dir, selected_classes, samples_per_class=20):
        self.data_dir = data_dir
        self.selected_classes = selected_classes
        self.samples_per_class = samples_per_class

    def load_data(self):
        transform = transforms.Compose([
            transforms.Resize((84, 84)),
            transforms.ToTensor()
        ])

        dataset = ImageFolder(self.data_dir, transform=transform)
        class_to_idx = {cls: idx for idx, cls in enumerate(dataset.classes)}

        selected = {cls: [] for cls in self.selected_classes}
        for path, label in dataset.samples:
            cls = dataset.classes[label]
            if cls in self.selected_classes and len(selected[cls]) < self.samples_per_class:
                selected[cls].append((path, label))
        
        final_samples = [item for sublist in selected.values() for item in sublist]
        dataset.samples = final_samples
        dataset.targets = [label for _, label in final_samples]

        from torch.utils.data import DataLoader
        return DataLoader(dataset, batch_size=16, shuffle=False)


class KNNHyperparameterTuner:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def tune_k(self, method='grid'):
        param_grid = {'n_neighbors': list(range(1, 21))}
        knn = KNeighborsClassifier()

        if method == 'grid':
            search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
        elif method == 'random':
            search = RandomizedSearchCV(knn, param_grid, cv=5, n_iter=10, scoring='accuracy')
        else:
            raise ValueError("method must be 'grid' or 'random'")

        print(f"🔍 Running {method.capitalize()}SearchCV for kNN...")
        search.fit(self.X, self.y)
        print(f"✅ Best k: {search.best_params_['n_neighbors']}")
        print(f"✅ Best Accuracy: {search.best_score_:.4f}")
        return search.best_params_['n_neighbors'], search.best_score_


# ========== Main Execution ========== #
if __name__ == "__main__":
    # Set your own miniImageNet path
    DATA_PATH = r'C:\yaswanth\BTECH\Sem 5\Machine Learning\Lab\Dataset'  # <-- Change this
    SELECTED_CLASSES = ['n01532829', 'n01749939']

    # Step 1: Load data
    loader = MiniImageNetDataLoader(DATA_PATH, SELECTED_CLASSES, samples_per_class=50)
    dataloader = loader.load_data()

    # Step 2: Extract Features using ResNet18
    extractor = FeatureExtractor(device='cuda' if torch.cuda.is_available() else 'cpu')
    X, y = extractor.extract_features(dataloader)

    # Step 3: Reduce to 2D (for consistency)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Step 4: Hyperparameter tuning for k
    tuner = KNNHyperparameterTuner(X_2d, y)
    best_k, best_score = tuner.tune_k(method='grid')  # You can also try 'random'
