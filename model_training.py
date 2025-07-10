import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from data_loading import load_data
from data_preprocessing import preprocess_data
from visualization import plot_confusion_matrix, plot_roc_curve
import joblib

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=2000)  # Increased max_iter
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    # Load and preprocess data
    train_df, _ = load_data()
    train_df = preprocess_data(train_df)

    # Split into features and target
    X = train_df.drop("Churn", axis=1)
    y = train_df["Churn"]

    # Train-test split from training data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features and retain column names
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_val = pd.DataFrame(scaler.transform(X_val), columns=X.columns)

    # Train model
    model = train_model(X_train, y_train)
    print("✅ Model training complete.")

    # Evaluate the model
    y_pred = model.predict(X_val)
    print("\n--- Model Evaluation ---")
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("\nClassification Report:\n", classification_report(y_val, y_pred))

    # Create visualizations
    plot_confusion_matrix(y_val, y_pred)
    plot_roc_curve(model, X_val, y_val)

    print("\nVisualizations saved as PNG files:")
    print("- confusion_matrix.png")
    print("- roc_curve.png")

    # Save model and scaler
    try:
        joblib.dump(model, 'churn_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        print("\n✅ Model and scaler saved:")
        print("- churn_model.pkl")
        print("- scaler.pkl")
    except Exception as e:
        print(f"\n❌ Error saving model or scaler: {e}")
