import os
import pandas as pd
from .base_dataset import Dataset

FILENAME = os.path.join(os.path.dirname(__file__), "bank_marketing", "bank-full.csv")

class BankMarketingDataset(Dataset):
    @staticmethod
    def get_df_data() -> pd.DataFrame:
        df = pd.read_csv(FILENAME, sep=";")
        # for col, series in df.select_dtypes(exclude="number").items():
        #     print(col.upper(), series.value_counts(), sep="\n", end="\n\n\n")
        return df
    
    @staticmethod
    def get_label_column() -> str:
        return "y"
    
    @staticmethod
    def remove_percentile(df: pd.DataFrame, pct: float) -> pd.DataFrame:
        return NotImplemented