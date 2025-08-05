import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from typing import Tuple


class PurchaseDataLoader:
    def __init__(self, excel_path: str, sheet_name: str):
        self.excel_path = excel_path
        self.sheet_name = sheet_name

    def load_data(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name)
        X = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
        y = df['Payment (Rs)'].values
        return df, X, y


class PricePredictor:
    def __init__(self):
        self.model = LinearRegression()

    def train_and_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.model.fit(X, y)
        predictions = self.model.predict(X)
        return predictions


class RegressionEvaluator:
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print("📊 Regression Evaluation Metrics:")
        print(f"MSE  : {mse:.2f}")
        print(f"RMSE : {rmse:.2f}")
        print(f"MAPE : {mape*100:.2f}%")
        print(f"R²    : {r2:.4f}")

        return mse, rmse, mape, r2


# ========== Main Execution ==========
if __name__ == "__main__":
    EXCEL_PATH = r"C:\yaswanth\BTECH\Sem 5\Machine Learning\Lab\Lab Session Data.xlsx"
    SHEET_NAME = "Purchase data"

    # Load data
    loader = PurchaseDataLoader(EXCEL_PATH, SHEET_NAME)
    df, X, y = loader.load_data()

    # Train model and predict
    predictor = PricePredictor()
    y_pred = predictor.train_and_predict(X, y)

    # Evaluate model
    evaluator = RegressionEvaluator()
    evaluator.evaluate(y, y_pred)
