import random
import pandas as pd

from numpy import ndarray
from typing import List, Tuple, Union, Dict, Any

class Dataset:
    @staticmethod
    def get_df_data() -> pd.DataFrame:
        return NotImplemented

    @staticmethod
    def get_label_column() -> str:
        return NotImplemented

    @staticmethod
    def remove_percentile(df: pd.DataFrame, pct: float) -> pd.DataFrame:
        return NotImplemented

    @staticmethod
    def na_fill_values(df: pd.DataFrame) -> Dict[str: Any]:
        return NotImplemented

    @staticmethod
    def remove_na(cls, df: pd.DataFrame, na_handling: str="drop") -> pd.DataFrame:
        if na_handling == "fill":
            val_default = cls.na_fill_values(df)

            for k, v in val_default.items():
                if v is not None:
                    df[k] = df[k].fillna(v)
        else:
            df = df.dropna().reset_index(drop=True)
        return df

    @staticmethod
    def data_label_separation(df: pd.DataFrame, label_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_y = df[[label_column]]
        df_x = df.drop(columns=[label_column])

        return df_x, df_y

    @staticmethod
    def discretize_numeric(df_x: pd.DataFrame) -> pd.DataFrame:
        nunique = df_x.nunique(axis=0)
        df_x_mean = df_x.mean(axis=0, numeric_only=True)

        for col, n in nunique.items():
            if n > 4:
                df_x[col] = df_x[col].apply(lambda x : min(4, x//(.5*df_x_mean[col])))
        
        return df_x

    @staticmethod
    def normalize_numeric(df_x: pd.DataFrame) -> pd.DataFrame:
        df_x_mean = df_x.mean(axis=0, numeric_only=True)
        df_x_std = df_x.std(axis=0, numeric_only=True)

        df_x[df_x_mean.index] = (df_x[df_x_mean.index] - df_x_mean) / df_x_std

        return df_x

    @staticmethod
    def hot_encode(df_x: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df_x = pd.get_dummies(df_x, columns=columns, drop_first=True)
        return df_x

    # TODO multi-class
    @staticmethod
    def balance_dataset(df_x: pd.DataFrame, df_y: pd.DataFrame, label_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    def label_to_numeric(df_y: pd.DataFrame, label_column: str) -> pd.DataFrame:
        numeric_df_y = df_y.apply(pd.to_numeric, errors="coerce")

        if numeric_df_y.isnull().values.any():
            df_y[label_column] = pd.Categorical(df_y[label_column]).codes
        else:
            df_y = numeric_df_y

        return df_y

    @classmethod
    def get_dataset(cls, balancing: bool=True, discretizing: bool=True, hot_encoding: bool=True, na_handling: str="drop", rmv_pct: Union[bool|str]=False) -> Tuple[ndarray, ndarray]:
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