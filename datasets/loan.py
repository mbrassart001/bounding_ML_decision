import os
import pandas as pd
from .base_dataset import Dataset

from typing import Hashable, Mapping

FILENAME = os.path.join(os.path.dirname(__file__), "loan", "loan_data_set.csv")

class LoanDataset(Dataset):
    @staticmethod
    def get_df_data() -> pd.DataFrame:
        df = pd.read_csv(FILENAME, sep=",")
        df = df.drop(columns=["Loan_ID"])
        return df
    
    @staticmethod
    def get_label_column() -> str:
        return "Loan_Status"
    
    @staticmethod
    def remove_percentile(df: pd.DataFrame, pct: float) -> pd.DataFrame:
        df_rank = df[["ApplicantIncome", "CoapplicantIncome"]]
        df_rank["rankA"] = df_rank[["ApplicantIncome"]].rank(pct=True)
        df_rank["rankCo"] = df_rank["CoapplicantIncome"].rank(pct=True)

        df_res = df.loc[(df_rank["rankA"]<=pct) & (df_rank["rankCo"]<=pct)]
        df_res.index = range(len(df_res))

        return df_res

    @staticmethod
    def na_fill_values(df: pd.DataFrame) -> Hashable | Mapping | pd.Series | pd.DataFrame:
        return {
            'Gender': 'Male',
            'Married': 'No',
            'Dependents': None,
            'Education': 'Not Graduate',
            'Self_Employed': None,
            # 'ApplicantIncome': 0,
            # 'CoapplicantIncome': 0,
            'LoanAmount': 0,
            'Loan_Amount_Term': df['LoanAmount'],
            'Credit_History': 0,
            'Property_Area': None,
        }