import os
from pandas import read_csv
from .base_dataset import Dataset

FILENAME = os.path.join(os.path.dirname(__file__), "diabetes", "diabetes.csv")

class DiabetesDataset(Dataset):
    @staticmethod
    def get_df_data():
        df = read_csv(FILENAME, sep=",")
        return df
    
    @staticmethod
    def get_label_column():
        return "Outcome"
    
    @staticmethod
    def remove_percentile(df, pct):
        return NotImplemented