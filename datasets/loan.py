import os
from pandas import read_csv, get_dummies
from .base_dataset import Dataset

FILENAME = os.path.join(os.path.dirname(__file__), "loan", "loan_data_set.csv")

class LoanDataset(Dataset):
    @staticmethod
    def get_df_data():
        df = read_csv(FILENAME, sep=",")
        df = df.drop(columns=["Loan_ID"])
        return df
    
    @staticmethod
    def get_label_column():
        return "Loan_Status"
    
    @staticmethod
    def remove_percentile(df, pct):
        df_rank = df[["ApplicantIncome", "CoapplicantIncome"]]
        df_rank["rankA"] = df_rank[["ApplicantIncome"]].rank(pct=True)
        df_rank["rankCo"] = df_rank["CoapplicantIncome"].rank(pct=True)

        df_res = df.loc[(df_rank["rankA"]<=pct) & (df_rank["rankCo"]<=pct)]
        df_res.index = range(len(df_res))

        return df_res

    @staticmethod
    def na_fill_values(df):
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