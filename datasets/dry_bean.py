import os
import pandas as pd 
from .base_dataset import Dataset

FILENAME = os.path.join(os.path.dirname(__file__), "dry_bean", "Dry_Bean_Dataset.csv")

class DryBeanDataset(Dataset):
    @staticmethod
    def get_df_data() -> pd.DataFrame:
        df = pd.read_csv(FILENAME, sep=",")
        return df
    
    @staticmethod
    def get_label_column() -> str:
        return "Class"

    @staticmethod
    def remove_percentile(df: pd.DataFrame, pct: float) -> pd.DataFrame:
        return NotImplemented