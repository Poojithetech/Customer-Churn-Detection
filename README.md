# 🧠 Customer Churn Prediction App

This project predicts customer churn using a machine learning model and provides a web interface built with Flask for user interaction.

---

## 📁 Project Files

- `app.py`: Flask web app to input customer data and get churn prediction.
- `config.py`: Stores paths to training and test datasets.
- `data_loading.py`: Loads training and test data.
- `data_preprocessing.py`: Cleans and encodes data for modeling.
- `model_training.py`: Trains the logistic regression model and saves it.
- `model_inference.py`: Loads model and predicts churn on new data.
- `model_evaluation.py`: Evaluates the model with confusion matrix and ROC curve.
- `visualization.py`: Draws plots to visualize model performance.

---

## 🚀 How to Run

1. Install required packages:
   ```
   pip install pandas scikit-learn flask matplotlib seaborn joblib
   ```

2. Add your training and test CSV files to the `data/` folder.

3. Train the model:
   ```
   python model_training.py
   ```

4. Run the web app:
   ```
   python app.py
   ```

5. Open your browser and go to `http://127.0.0.1:5000`

---

## 📊 Output

- Model will create:
  - `churn_model.pkl`
  - `scaler.pkl`
  - `confusion_matrix.png`
  - `roc_curve.png`

These files will help with predictions and evaluation.

---

## ✅ Requirements

- Python 3.7+
- Flask
- pandas
- scikit-learn
- matplotlib
- seaborn
- joblib

---

## 📌 Notes

- Ensure `index.html` is placed in the `templates/` folder.
- Do not forget to upload `churn_model.pkl` and `scaler.pkl` before running the app.

---

## 📃 License

Open for learning and academic use.
