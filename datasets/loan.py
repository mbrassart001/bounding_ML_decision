import os
import random
import pandas as pd

def get_df_data():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "loan", "loan_data_set.csv"), sep=",")
    df = df.drop(columns=["Loan_ID"])

    return df

def remove_na(df, method):      
    if method == 'fill':
        val_default = {
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
    
        for k, v in val_default.items():
            if v is not None:
                df[k] = df[k].fillna(v)

    df = df.dropna().reset_index(drop=True)
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

def normalize_numeric(df_x):
    df_x_mean = df_x.mean(axis=0, numeric_only=True)
    df_x_std = df_x.std(axis=0, numeric_only=True)

    df_x[df_x_mean.index] = (df_x[df_x_mean.index] - df_x_mean) / df_x_std

    return df_x

def hot_encode(df_x, columns):
    df_x = pd.get_dummies(df_x, columns=columns, drop_first=True)
    return df_x

def balance_dataset(df_x, df_y):
    itrue = df_y.index[df_y["Loan_Status_Y"]==1].tolist()
    ifalse = df_y.index[df_y["Loan_Status_Y"]==0].tolist()

    if len(itrue) > len(ifalse):
        itrue = random.choices(itrue, k=len(ifalse))
    else:
        ifalse = random.choices(ifalse, k=len(itrue))

    df_x = df_x.iloc[itrue+ifalse]
    df_y = df_y.iloc[itrue+ifalse]

    return df_x, df_y

def get_loan_dataset(balancing=True, discretizing=True, hot_encoding=True, na_handling='drop', rmv_pct=False):
    df = get_df_data()
    df = remove_na(df, na_handling)
    if rmv_pct:
        df = remove_percentile(df, rmv_pct)
    df_x, df_y = data_label_separation(df)
    if discretizing:
        df_x = discretize_numeric(df_x)
        hot_encode_columns = df_x.columns
    else:
        df_x = normalize_numeric(df_x)
        hot_encode_columns = df_x.select_dtypes(exclude='number').columns
    if hot_encoding:
        df_x = hot_encode(df_x, hot_encode_columns)
    if balancing:
        df_x, df_y = balance_dataset(df_x, df_y)
    
    x = df_x.to_numpy(dtype=float)
    y = df_y.to_numpy(dtype=float)
    
    return x, y