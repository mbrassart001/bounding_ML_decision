import random
import pandas as pd

class Dataset:
    @staticmethod
    def get_df_data():
        return NotImplemented

    @staticmethod
    def get_label_column():
        return NotImplemented

    @staticmethod
    def remove_percentile(df, pct):
        return NotImplemented

    @staticmethod
    def na_fill_values(df):
        return NotImplemented

    @staticmethod
    def remove_na(cls, df, na_handling="drop"):
        if na_handling == "fill":
            val_default = cls.na_fill_values(df)

            for k, v in val_default.items():
                if v is not None:
                    df[k] = df[k].fillna(v)
        else:
            df = df.dropna().reset_index(drop=True)
        return df

    @staticmethod
    def data_label_separation(df, label_column):
        df_y = df[[label_column]]
        df_x = df.drop(columns=[label_column])

        return df_x, df_y

    @staticmethod
    def discretize_numeric(df_x):
        nunique = df_x.nunique(axis=0)
        df_x_mean = df_x.mean(axis=0, numeric_only=True)

        for col, n in nunique.items():
            if n > 4:
                df_x[col] = df_x[col].apply(lambda x : min(4, x//(.5*df_x_mean[col])))
        
        return df_x

    @staticmethod
    def normalize_numeric(df_x):
        df_x_mean = df_x.mean(axis=0, numeric_only=True)
        df_x_std = df_x.std(axis=0, numeric_only=True)

        df_x[df_x_mean.index] = (df_x[df_x_mean.index] - df_x_mean) / df_x_std

        return df_x

    @staticmethod
    def hot_encode(df_x, columns):
        df_x = pd.get_dummies(df_x, columns=columns, drop_first=True)
        return df_x

    # TODO multi-class
    @staticmethod
    def balance_dataset(df_x, df_y, label_column):
        itrue = df_y.index[df_y[label_column]==1].tolist()
        ifalse = df_y.index[df_y[label_column]==0].tolist()

        if len(itrue) > len(ifalse):
            itrue = random.choices(itrue, k=len(ifalse))
        else:
            ifalse = random.choices(ifalse, k=len(itrue))

        df_x = df_x.iloc[itrue+ifalse]
        df_y = df_y.iloc[itrue+ifalse]

        return df_x, df_y

    @staticmethod
    def label_to_numeric(df_y, label_column):
        numeric_df_y = df_y.apply(pd.to_numeric, errors="coerce")

        if numeric_df_y.isnull().values.any():
            df_y[label_column] = pd.Categorical(df_y[label_column]).codes
        else:
            df_y = numeric_df_y

        return df_y

    @classmethod
    def get_dataset(cls, balancing=True, discretizing=True, hot_encoding=True, na_handling="drop", rmv_pct=False):
        label_column = cls.get_label_column()
        df = cls.get_df_data()
        df = cls.remove_na(cls, df, na_handling)

        if rmv_pct:
            df = cls.remove_percentile(df, rmv_pct)
            
        df_x, df_y = cls.data_label_separation(df, label_column)
        df_y = cls.label_to_numeric(df_y, label_column)

        if balancing:
            df_x, df_y = cls.balance_dataset(df_x, df_y, label_column)

        if discretizing:
            df_x = cls.discretize_numeric(df_x)
            hot_encode_columns = df_x.columns
        else:
            df_x = cls.normalize_numeric(df_x)
            hot_encode_columns = df_x.select_dtypes(exclude="number").columns
        
        if hot_encoding:
            df_x = cls.hot_encode(df_x, hot_encode_columns)
        
        x = df_x.to_numpy(dtype=float)
        y = df_y.to_numpy(dtype=float)
        
        return x, y