import os
import pandas as pd
from .base_dataset import Dataset

FILENAME = os.path.join(os.path.dirname(__file__), "contraceptive", "cmc.data.csv")

class ContraceptiveDataset(Dataset):
    @staticmethod
    def get_df_data() -> pd.DataFrame:
        df = pd.read_csv(FILENAME, sep=",")
        df.replace({"contraceptive_method": 1}, 0, inplace=True)
        df.replace({"contraceptive_method": 2}, 1, inplace=True)
        df.replace({"contraceptive_method": 3}, 1, inplace=True)
        return df
    
    @staticmethod
    def get_label_column() -> str:
        return "contraceptive_method"
    
    @staticmethod
    def remove_percentile(df: pd.DataFrame, pct: float) -> pd.DataFrame:
        return NotImplemented