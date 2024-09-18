import os
import pandas as pd
from .base_dataset import Dataset

FILENAME = os.path.join(os.path.dirname(__file__), "adult", "adult.csv")

class AdultDataset(Dataset):
    @staticmethod
    def get_df_data() -> pd.DataFrame:
        df = pd.read_csv(FILENAME, sep=",")
        return df
    
    @staticmethod
    def get_label_column() -> str:
        return "income"
    
    @staticmethod
    def remove_percentile(df: pd.DataFrame, pct: float) -> pd.DataFrame:
        return NotImplemented
    
    @staticmethod
    def get_encoding_sizes() -> dict[str, int]:
        return {
            "workclass": 3,
            "education": 4,
            "marital-status": 2,
            "occupation": 3,
            "relationship": 2,
            "race": 2,
            "native-country": 5,
            # "age": ,
            # "fnlwgt": ,
            # "education-num": ,
            # "capital-gain": ,
            # "capital-loss": ,
            # "hours-per-week": ,
        }