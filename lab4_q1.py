import os
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


class MiniImageNetClassifier:
    def __init__(self, data_path, test_size=0.2, random_state=42, n_neighbors=3):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_pred = None
        self.y_test_pred = None

        self.class_names = []

    def load_and_preprocess_data(self):
        print("🔄 Loading and preprocessing data...")
        transform = transforms.Compose([
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))  # flatten
        ])

        dataset = datasets.ImageFolder(self.data_path, transform=transform)
        self.class_names = dataset.classes

        # tqdm progress bar for data loading
        X = []
        y = []
        for img_tensor, label in tqdm(dataset, desc="📦 Processing images", total=len(dataset)):
            X.append(img_tensor)
            y.append(label)

        X = torch.stack(X).numpy()
        y = np.array(y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )
        print(f"✅ Loaded {len(dataset)} samples. Train size: {len(self.X_train)}, Test size: {len(self.X_test)}")

    def train_model(self):
        print("🚀 Training KNN model...")
        self.model.fit(self.X_train, self.y_train)
        print("✅ Training complete.")

    def evaluate_model(self):
        print("🔍 Evaluating model...")

        self.y_train_pred = self._predict_with_progress(self.X_train, "🔁 Predicting train data")
        self.y_test_pred = self._predict_with_progress(self.X_test, "🔁 Predicting test data")

        print("\n========== 📊 Train Classification Report ==========")
        print(classification_report(self.y_train, self.y_train_pred, target_names=self.class_names))

        print("\n========== 📊 Test Classification Report ==========")
        print(classification_report(self.y_test, self.y_test_pred, target_names=self.class_names))

        cm_train = confusion_matrix(self.y_train, self.y_train_pred)
        cm_test = confusion_matrix(self.y_test, self.y_test_pred)

        self.plot_confusion_matrix(cm_train, "Train Confusion Matrix")
        self.plot_confusion_matrix(cm_test, "Test Confusion Matrix")

        self.infer_model_fit()

    def _predict_with_progress(self, data, desc):
        """Adds a progress bar for prediction (works for large datasets)"""
        batch_size = 500  # KNN is fast, but batching avoids memory issues for large sets
        preds = []
        for i in tqdm(range(0, len(data), batch_size), desc=desc):
            batch = data[i:i + batch_size]
            preds.extend(self.model.predict(batch))
        return np.array(preds)

    def plot_confusion_matrix(self, cm, title):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def infer_model_fit(self):
        train_acc = np.mean(self.y_train == self.y_train_pred)
        test_acc = np.mean(self.y_test == self.y_test_pred)

        print(f"\n✅ Train Accuracy: {train_acc:.2f}")
        print(f"✅ Test Accuracy:  {test_acc:.2f}")

        if train_acc < 0.7 and test_acc < 0.7:
            print("🔴 Model is UNDERFITTING.")
        elif train_acc > 0.9 and test_acc < 0.7:
            print("🟠 Model is OVERFITTING.")
        else:
            print("🟢 Model is REGULARLY FITTING (Generalizing well).")


# ======== Run Classifier ========
if __name__ == "__main__":
    # Replace with your actual path
    data_path = r'C:\yaswanth\BTECH\Sem 5\Machine Learning\Lab\Dataset'

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    classifier = MiniImageNetClassifier(data_path=data_path)
    classifier.load_and_preprocess_data()
    classifier.train_model()
    classifier.evaluate_model()
