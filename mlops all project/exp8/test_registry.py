import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

mlflow.set_tracking_uri("http://127.0.0.1:5000") 

X, y = load_iris(return_X_y=True)

with mlflow.start_run():
    model = RandomForestClassifier()
    model.fit(X, y)

    mlflow.sklearn.log_model(
        model,
        name="rf_model",
        registered_model_name="MyModel"
    )

print("Model registered successfully")