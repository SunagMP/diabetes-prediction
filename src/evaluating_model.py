import pickle
from sklearn.metrics import classification_report
import logging 
import os
import pandas as pd
import json

# ---------------------------------------------- logging -----------------------------------
import logging
logger = logging.getLogger("feature-engineering-logger")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'feature_engineering.log')

file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formate = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
console_handler.setFormatter(formate)
file_handler.setFormatter(formate)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ------------------------------ testing and evaluating the data -------------------------
def test_trained_model(model, x_test: pd.DataFrame):
    try:
        y_pred = model.predict(x_test)
        logger.debug("Model tested against test data")
        return y_pred
    except Exception as e:
        logger.error("Failed testing due to : %s", e)
        raise

def evaluate_the_model(y_test, y_pred):
    try:
        test_result = classification_report(y_test, y_pred)
        logger.debug("model tested against the test data")
        logger.info("the test result :\n %s", test_result)
        return test_result
    except Exception as e:
        logger.error("failed to evaluate, %s", e)
        raise

def save_report(test_result):
    try:
        os.makedirs('reports', exist_ok=True)
        with open('reports/metrics.json' , 'w') as f:
            json.dump(test_result, f)
        logger.debug("report saved successfully.")
    except Exception as e:
        logger.error("failed due to , %s", e)


def main():
    try:
        test_df = pd.read_csv('data/processed_data/processed_test_data.csv')
        X_test = test_df.drop(columns=['Outcome'])
        y_test = test_df['Outcome']

        with open('models/model.pkl', 'rb') as f:
            trained_model = pickle.load(f)

        y_pred = test_trained_model(trained_model, X_test)
        test_result = evaluate_the_model(y_test, y_pred)
        save_report(test_result)
        logger.info("Model evaluation successful")

    except Exception as e:
        logger.debug("Failed due to %s", e)

if __name__ == '__main__':
    main()