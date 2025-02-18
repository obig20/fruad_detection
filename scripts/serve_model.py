# %%
# Import libraries
import pandas as pd
import shap
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def explain_with_shap(model, X_train, X_test, feature_names):
    """Explain model predictions using SHAP"""
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test)
    
    # Summary plot (global feature importance)
    print("SHAP Summary Plot:")
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)
    plt.show()
    
    # Force plot for a single prediction (local explanation)
    print("SHAP Force Plot for the first prediction:")
    shap.force_plot(
        explainer.expected_value,  # Use the expected value
        shap_values[1][0, :],         # SHAP values for the positive class
        X_test.iloc[0, :],            # Use .iloc to access the first row
        feature_names=feature_names
    )
    plt.show()
    
    # Dependence plot (example: first feature)
    print("SHAP Dependence Plot for the first feature:")
    shap.dependence_plot(
        feature_names[0], 
        shap_values[1], 
        X_test, 
        feature_names=feature_names
    )
    plt.show()
def explain_with_lime(model, X_train, X_test, feature_names, class_names):
    """Explain model predictions using LIME"""
    # Initialize LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train, 
        feature_names=feature_names, 
        class_names=class_names, 
        mode='classification'
    )
    
    # Explain a single prediction
    exp = explainer.explain_instance(
        X_test.iloc[0],  # Use .iloc to access the first row
        model.predict_proba, 
        num_features=5
    )
    
    # Plot feature importance for the explanation
    print("LIME Explanation for the first prediction:")
    exp.as_pyplot_figure()
    plt.show()


def prepare_data(data, target_column):
    """Prepare data for training and testing"""
    from sklearn.model_selection import train_test_split
    
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
def main():
    # Load datasets
    fraud_data = pd.read_csv('processed_fraud_data.csv')
    creditcard_data=pd.read_csv('../Data/creditcard.csv')
    
    # Prepare data
    X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = prepare_data(
        fraud_data, target_column='class'
    )
    X_train_cc, X_test_cc, y_train_cc, y_test_cc = prepare_data(
        creditcard_data, target_column='Class'
    )
    
    # Train a Random Forest model (example using fraud_data)
    model = RandomForestClassifier()
    model.fit(X_train_fraud, y_train_fraud)
    
    # Feature names (replace with actual column names)
    feature_names = fraud_data.drop('class', axis=1).columns.tolist()
    
    # Explain with SHAP
    print("Explaining Fraud Data Model with SHAP...")
    explain_with_shap(model, X_train_fraud, X_test_fraud, feature_names)
    
    # Explain with LIME
    print("Explaining Fraud Data Model with LIME...")
    explain_with_lime(
        model, 
        X_train_fraud, 
        X_test_fraud, 
        feature_names, 
        class_names=['Non-Fraud', 'Fraud']
    )
if __name__ == "__main__":
    main()
