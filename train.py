# train.py
import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Enable MLflow autologging
mlflow.sklearn.autolog()

def train_model():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_metric("accuracy", acc)
        print(f"Model accuracy: {acc}")

        # Log model
        mlflow.sklearn.log_model(model, "model")





if __name__ == "__main__":
    train_model()
