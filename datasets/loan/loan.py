import os
import random
import pandas as pd

def get_df_data():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "loan_data_set.csv"), sep=",")
    df = df.drop(columns=["Loan_ID"])

    return df

def remove_percentile(df, pct):
    df_rank = df[["ApplicantIncome", "CoapplicantIncome"]]
    df_rank["rankA"] = df_rank[["ApplicantIncome"]].rank(pct=True)
    df_rank["rankCo"] = df_rank["CoapplicantIncome"].rank(pct=True)

    df_res = df.loc[(df_rank["rankA"]<=pct) & (df_rank["rankCo"]<=pct)]
    df_res.index = range(len(df_res))

    return df_res

def data_label_separation(df):
    df_y = pd.get_dummies(df[["Loan_Status"]], drop_first=True)
    df_x = df.drop(columns=["Loan_Status"])

    return df_x, df_y

def discretize_numeric(df_x):
    nunique = df_x.nunique(axis=0)
    df_x_mean = df_x.mean(axis=0, numeric_only=True)

    for col, n in nunique.items():
        if n > 4:
            df_x[col] = df_x[col].apply(lambda x : min(4, x//(.5*df_x_mean[col])))
    
    return df_x

def hot_encode(df_x, columns):
    df_x = pd.get_dummies(df_x, columns=columns, drop_first=True)
    return df_x

def balance_dataset(df_x, df_y):
    itrue = df_y.index[df_y["Loan_Status_Y"]==1].tolist()
    ifalse = df_y.index[df_y["Loan_Status_Y"]==0].tolist()

    swap = len(itrue) > len(ifalse)
    if swap:
        itrue,ifalse=ifalse,itrue

    ifalse = random.choices(ifalse, k=len(itrue))

    if swap:
        itrue,ifalse=ifalse,itrue

    df_x = df_x.iloc[itrue+ifalse]
    df_y = df_y.iloc[itrue+ifalse]

    return df_x, df_y

def get_loan_dataset(balancing=True, discretizing=True, hot_encoding=True, rmv_pct=False):
    df = get_df_data()
    if rmv_pct:
        df = remove_percentile(df, rmv_pct)
    df_x, df_y = data_label_separation(df)
    if discretizing:
        df_x = discretize_numeric(df_x)
        hot_encode_columns = df_x.columns
    else:
        hot_encode_columns = [col for col, n in df_x.nunique(axis=0).items() if n < 5]
    if hot_encoding:
        df_x = hot_encode(df_x, hot_encode_columns)
    if balancing:
        df_x, df_y = balance_dataset(df_x, df_y)
    
    x = df_x.to_numpy(dtype=float)
    y = df_y.to_numpy(dtype=float)
    
    return x, y