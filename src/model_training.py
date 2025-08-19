import pickle
from imblearn.ensemble import BalancedRandomForestClassifier
import logging 
import os
import pandas as pd

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

def train_model(x_train:pd.DataFrame, y_train:pd.DataFrame):
    try:
        model = BalancedRandomForestClassifier()
        model.fit(x_train, y_train)
        logger.debug("The model trained successfully.")
        return model
    except Exception as e:
        logger.error("couldn't train the model, because %s", e)

def save_trained_model(trained_model):
    try:
        os.makedirs('models', exist_ok=True)   # only create directory 
        with open('models/model.pkl', 'wb') as f:
            pickle.dump(trained_model, f)      # direct pickle dump
        logger.debug("trained model saved successfully")
    except Exception as e:
        logger.error("Failed to save model, due to %s", e)
        raise

def main():
    try:
        df = pd.read_csv('data/processed_data/processed_train_data.csv')
        X = df.drop(columns=['Outcome'])
        y = df['Outcome']
        trained_model = train_model(x_train= X, y_train=y)
        save_trained_model(trained_model)
    except Exception as e:
        logger.error("Failed to train and save model, %s", e)

if __name__ == '__main__':
    main()