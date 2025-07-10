import joblib
import pandas as pd
from data_loading import load_data
from data_preprocessing import preprocess_data
from visualization import plot_confusion_matrix, plot_roc_curve

if __name__ == "__main__":
    # Load data
    _, test_df = load_data()
    test_df = preprocess_data(test_df)

    # Split features and target
    X_test = test_df.drop("Churn", axis=1)
    y_test = test_df["Churn"]

    # Load trained model
    model = joblib.load("churn_model.pkl")

    # Predict
    y_pred = model.predict(X_test)

    # Plot and save visualizations
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(model, X_test, y_test)

    print("✅ Evaluation visuals saved: 'confusion_matrix.png' & 'roc_curve.png'")
