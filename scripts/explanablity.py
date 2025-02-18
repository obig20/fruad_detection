# %%
# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import mlflow
import mlflow.sklearn

# %%
def prepare_data(df, target_column):
    """Prepare data for modeling"""
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# %%
def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a model"""
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate the model
    print(f"Classification Report for {model_name}:")
    print(classification_report(y_test, y_pred))
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC Score for {model_name}: {roc_auc:.4f}")
    
    # Log metrics with MLflow
    with mlflow.start_run():
        mlflow.log_param("model", model_name)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.sklearn.log_model(model, model_name)

# %%
def main():
    # Load datasets
    fraud_data = pd.read_csv('processed_fraud_data.csv')
    creditcard_data = pd.read_csv('creditcard.csv')
    
    # Prepare data
    X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = prepare_data(
        fraud_data, target_column='class'
    )
    X_train_cc, X_test_cc, y_train_cc, y_test_cc = prepare_data(
        creditcard_data, target_column='Class'
    )
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
    }
    
    # Train and evaluate models
    for model_name, model in models.items():
        print(f"\nTraining {model_name} on Fraud Data...")
        train_and_evaluate(model, X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud, model_name)
        
        print(f"\nTraining {model_name} on Credit Card Data...")
        train_and_evaluate(model, X_train_cc, X_test_cc, y_train_cc, y_test_cc, model_name)

# %%
if __name__ == "__main__":
    # Initialize MLflow
    mlflow.set_experiment("Fraud Detection")
    
    # Run the main function
    main()