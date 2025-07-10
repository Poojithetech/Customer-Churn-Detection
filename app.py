import os
from flask import Flask, render_template, request
import joblib
import pandas as pd

# Configure template folder path
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=template_dir)

# Load model and scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")


def preprocess_input(form_data):
    """Replicate the exact preprocessing from training"""
    df = pd.DataFrame([form_data])

    # Convert to numeric
    numeric_cols = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls',
                    'Payment Delay', 'Total Spend', 'Last Interaction']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    # Binary encoding
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=['Subscription Type', 'Contract Length'])

    # Ensure all expected columns exist
    expected_columns = model.feature_names_in_
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    return df[expected_columns]


@app.route("/", methods=["GET", "POST"])
def predict():
    try:
        if request.method == "POST":
            form_data = {
                'Age': request.form.get('Age'),
                'Tenure': request.form.get('Tenure'),
                'Usage Frequency': request.form.get('Usage_Frequency'),
                'Support Calls': request.form.get('Support_Calls'),
                'Payment Delay': request.form.get('Payment_Delay'),
                'Total Spend': request.form.get('Total_Spend'),
                'Last Interaction': request.form.get('Last_Interaction'),
                'Gender': request.form.get('Gender'),
                'Subscription Type': request.form.get('Subscription_Type', 'Standard'),
                'Contract Length': request.form.get('Contract_Length', 'Monthly')
            }

            processed_data = preprocess_input(form_data)
            scaled_data = scaler.transform(processed_data)
            prediction = model.predict(scaled_data)[0]

            result = "Churn Risk: HIGH" if prediction == 1 else "Churn Risk: LOW"
            return render_template("index.html", result=result)

        return render_template("index.html", result=None)

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)  # Log the error
        return render_template("index.html", result=error_msg)


if __name__ == "__main__":
    # Verify template path
    print(f"Looking for templates in: {template_dir}")
    print(f"Template exists: {os.path.exists(os.path.join(template_dir, 'index.html'))}")
    app.run(debug=True)