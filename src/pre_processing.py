import pandas as pd
import os
from sklearn.model_selection import train_test_split

# ------------------------ logging -------------------------------------
import logging
logger = logging.getLogger("pre-processing-logger")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'pre_processing.log')

file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formate = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
console_handler.setFormatter(formate)
file_handler.setFormatter(formate)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ----------------------------------------- yaml : params reading -----------------
import yaml

def load_params(params_path:str):
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.debug("params loaded successful")
        return params
    except Exception as e:
        logger.error("failed due to , %s", e)

def splitData(df:pd.DataFrame, size : int):
    try:
        train_data, test_data = train_test_split(df, test_size=size, random_state=2)
        logger.debug("The data split successful")
        return train_data, test_data
    except Exception as e:
        logger.error("failed to split the dataframe , due to %s", e)
        raise

def saveSplitData(train_data:pd.DataFrame, test_data:pd.DataFrame):
    try:
        os.makedirs('data', exist_ok=True)
        os.makedirs('./data/split_raw_data', exist_ok=True)

        train_path = os.path.join('data', 'split_raw_data', 'train_data.csv')
        train_data.to_csv(train_path, index=False)
        logger.debug("train data saved.")

        test_path = os.path.join('data', 'split_raw_data', 'test_data.csv')
        test_data.to_csv(test_path, index=False)
        logger.debug("test data saved.")
    except Exception as e:
        logger.error("failed to save %s", e)
        raise

def main():
    try:
        params = load_params('params.yaml')
        test_size = params['pre_processing']['test_size']
        df = pd.read_csv('data/raw/raw_data.csv')
        train_data, test_data = splitData(df, test_size)
        saveSplitData(train_data, test_data)
    except Exception as e:
        logger.error("failed to split and fail %s", e)

if __name__ == '__main__':
    main()