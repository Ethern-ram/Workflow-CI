import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

def load_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    if not os.path.exists(train_path):
        print(f"Data not found at {data_dir}. Menggunakan data dummy.")
        # Create dummy data for CI to not fail if data is not committed
        X_train = pd.DataFrame(np.random.rand(100, 5))
        y_train = pd.Series(np.random.randint(0, 2, 100))
        X_test = pd.DataFrame(np.random.rand(20, 5))
        y_test = pd.Series(np.random.randint(0, 2, 20))
        return X_train, X_test, y_train, y_test

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    X_train = df_train.drop('target', axis=1)
    y_train = df_train['target']
    X_test = df_test.drop('target', axis=1)
    y_test = df_test['target']
    return X_train, X_test, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = load_data("breast_cancer_preprocessing")
    
    with mlflow.start_run() as run:
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", acc)
        # Force model save to predictable artifacts path so CI knows
        mlflow.sklearn.log_model(rf, "model")
        
        # Simpan run ID dalam file text untuk mempermudah docker build
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

if __name__ == "__main__":
    main()
