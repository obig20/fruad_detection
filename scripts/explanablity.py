import shap
import joblib
import pandas as pd

def explain_model(model, X_test):
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)

if __name__ == "__main__":
    model = joblib.load("models/random_forest.pkl")
    data = pd.read_csv("data/processed_creditcard_data.csv")
    X_test = data.drop("Class", axis=1)
    explain_model(model, X_test)