import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"
mlflow.set_tracking_uri("file:./mlruns")


mlflow.set_tracking_uri("file:./mlruns")
# Load the iris dataset
iris = load_iris()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the model on the training data
rf_classifier.fit(X_train, y_train)
# Log the model with MLflow
mlflow.sklearn.log_model(rf_classifier, name="random_forest_model")
# Log the model's parameters and metrics
mlflow.log_param("n_estimators", 100)
mlflow.log_param("random_state", 42)
accuracy = rf_classifier.score(X_test, y_test)
mlflow.log_metric("accuracy", accuracy)
print(f"Model trained and logged with MLflow. Accuracy: {accuracy:.2f}")
