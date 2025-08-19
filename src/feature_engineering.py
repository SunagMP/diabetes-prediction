import pandas as pd
import os


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

#---------------------------------------------- Utility functions -----------------------
def ageGroup(age):
    if age <= 30:
        return 0           # age group 0 young
    elif age <= 60:
        return 1           # age group 1 elders
    return 2               # age group 2 old

def bmiCategory(bmi):
    if bmi <= 18.5:
        return 0         # underweight
    elif bmi <= 24.9:
        return 1         # normal
    elif bmi <= 29.9:
        return 2         # overweight
    return 3             # obbessed

def create_features(df:pd.DataFrame):
    try:
        df['AgeGroup'] = df['Age'].map(ageGroup)
        df['BMIGroup'] = df['BMI'].map(bmiCategory)
        # glucose to insuline ration -> GTIratio
        df['GTIratio'] = df['Glucose'] / (df['Insulin'] +1)
        df['isInsulinFlag'] =((df['Insulin'] > 100) & (df['Glucose'] > 120)).astype(int)
        logger.debug("All features created successfully")
        return df
    except Exception as e:
        logger.error("Failed to create the features, due to : %s", e)
        raise

def saveProccedDF(df: pd.DataFrame, file_path:str):
    try:
        os.makedirs('data', exist_ok=True)
        os.makedirs('./data/processed_data', exist_ok=True)

        dest = os.path.join('data', 'processed_data', f"{file_path}.csv")
        df = df.drop(columns=['Age', 'BMI', 'SkinThickness'])
        df.to_csv(dest, index=False)
        logger.debug("%s saved successfully", file_path)

    except Exception as e:
        logger.error("failed to save %s", e)

def main():
    try:
        train_df = pd.read_csv('data/split_raw_data/train_data.csv')
        processed_df = create_features(train_df)
        saveProccedDF(processed_df, "processed_train_data")

        test_df = pd.read_csv('data/split_raw_data/test_data.csv')
        processed_df = create_features(test_df)
        saveProccedDF(processed_df, "processed_test_data")

        logger.debug("Clean data saved .")
    except Exception as e:
        logger.error("couldn't clean data , due to : ", e)

    
if __name__ == '__main__':
    main()