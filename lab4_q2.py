# Import required libraries
import pandas as pd  # for reading Excel data and handling dataframes
import numpy as np  # for numerical operations
from sklearn.linear_model import LinearRegression  # to create a linear regression model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score  # for evaluation metrics
from typing import Tuple  # to annotate return types

# Class to load and prepare data from Excel
class PurchaseDataLoader:
    def __init__(self, excel_path: str, sheet_name: str):
        """
        Initialize the data loader with the path to the Excel file and the sheet name.
        """
        self.excel_path = excel_path
        self.sheet_name = sheet_name

    def load_data(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Loads the Excel sheet and separates input features (X) and target variable (y).
        Returns:
            df: the full dataframe
            X: input features (Candies, Mangoes, Milk Packets)
            y: target (Payment in Rs)
        """
        df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name)  # read Excel sheet
        X = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values  # select input columns
        y = df['Payment (Rs)'].values  # select target column
        return df, X, y

# Class to train a linear regression model and make predictions
class PricePredictor:
    def __init__(self):
        """
        Initialize a Linear Regression model from scikit-learn.
        """
        self.model = LinearRegression()

    def train_and_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Train the model on given features and labels, then predict.
        Args:
            X: input features
            y: true labels
        Returns:
            predictions: predicted prices
        """
        self.model.fit(X, y)  # train the model
        predictions = self.model.predict(X)  # predict on the same input data
        return predictions

# Class to evaluate model predictions using standard regression metrics
class RegressionEvaluator:
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Evaluate the predictions against true values using regression metrics.
        Args:
            y_true: actual target values
            y_pred: predicted target values
        Prints:
            MSE, RMSE, MAPE, R² score
        """
        mse = mean_squared_error(y_true, y_pred)  # mean squared error
        rmse = np.sqrt(mse)  # root mean squared error
        mape = mean_absolute_percentage_error(y_true, y_pred)  # mean absolute percentage error
        r2 = r2_score(y_true, y_pred)  # R² score

        print("📊 Regression Evaluation Metrics:")
        print(f"MSE  : {mse:.2f}")
        print(f"RMSE : {rmse:.2f}")
        print(f"MAPE : {mape*100:.2f}%")
        print(f"R²    : {r2:.4f}")

        return mse, rmse, mape, r2


# ========== Main Execution ==========
if __name__ == "__main__":
    # Specify the path to the Excel file and the sheet name
    EXCEL_PATH = r"C:\yaswanth\BTECH\Sem 5\Machine Learning\Lab\Lab Session Data.xlsx"
    SHEET_NAME = "Purchase data"

    # Step 1: Load the data
    loader = PurchaseDataLoader(EXCEL_PATH, SHEET_NAME)
    df, X, y = loader.load_data()

    # Step 2: Train the model and make predictions
    predictor = PricePredictor()
    y_pred = predictor.train_and_predict(X, y)

    # Step 3: Evaluate the model's performance
    evaluator = RegressionEvaluator()
    evaluator.evaluate(y, y_pred)
