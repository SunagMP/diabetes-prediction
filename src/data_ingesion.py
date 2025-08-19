import pandas as pd
import os 
 
import logging

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data_ingesion_logger")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingesion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formate = logging.Formatter('%(asctime)s  %(name)s %(levelname)s  %(message)s')
console_handler.setFormatter(formate)
file_handler.setFormatter(formate)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def getData(url:str):
    """ This function gathers the data from the given url"""
    try:
        df = pd.read_csv(url)
        logger.debug("The data successfully gathered from %s", url)
        return df
    except Exception as e:
        logger.error("The following unexpected error happend %s", e)
        raise

def saveDataframe(df:pd.DataFrame):
    try:
        data_dir = 'data'
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs('./data/raw', exist_ok=True)
        raw_data_path = os.path.join(data_dir, 'raw','raw_data.csv')
        df.to_csv(raw_data_path, index= False)
        logger.debug("Data saved successfully at %s", raw_data_path)

    except Exception as e:
        logger.error("The following unexpected error happend %s", e)
        raise

def main():
    try:
        
        data_url = "https://raw.githubusercontent.com/SunagMP/diabetes-dataset/refs/heads/main/diabetes.csv"
        url_df = getData(url= data_url)
        saveDataframe(df= url_df)
        logger.debug("Data Ingesion successfull")
    except Exception as e:
        logger.error("failed to perform data ingesion %s", e)

if __name__ == '__main__':
    main()