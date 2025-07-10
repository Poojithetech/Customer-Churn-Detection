import pandas as pd
import joblib
from data_loading import load_data
from data_preprocessing import preprocess_data
from sklearn.linear_model import LogisticRegression

MODEL_PATH = "churn_model.pkl"

def save_model(model):
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def load_model():
    return joblib.load(MODEL_PATH)

if __name__ == "__main__":
    # Load and preprocess data
    train_df, test_df = load_data()
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)

    # Train a new model
    X_train = train_df.drop("Churn", axis=1)
    y_train = train_df["Churn"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Save model
    save_model(model)

    # Load model and make predictions on test set
    model = load_model()
    X_test = test_df.drop("Churn", axis=1)
    y_test = test_df["Churn"]
    y_pred = model.predict(X_test)

    from sklearn.metrics import accuracy_score, classification_report
    print("\n--- Test Set Evaluation ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
