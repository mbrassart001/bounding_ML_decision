import random
import pandas as pd
import numpy as np

from numpy import ndarray
from typing import Tuple, Union, Hashable, Mapping, Sequence

pd.options.mode.copy_on_write = True

DATASET_METADATA = {}

class Dataset:
    @classmethod
    def get_metadata(cls) -> dict[str, Tuple[int,int]|int]:
        metadata = DATASET_METADATA.get(cls)
        if metadata is None:
            raise ValueError(cls)
        return metadata 
    
    @staticmethod
    def get_encoding_sizes() -> dict[str, int]:
        return NotImplemented

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
    def na_fill_values(df: pd.DataFrame) -> Hashable | Mapping | pd.Series | pd.DataFrame:
        return NotImplemented

    @staticmethod
    def multiclass_handling(df: pd.DataFrame, label_column: str, keep_label: Union[int, Sequence[str]]=2) -> pd.DataFrame:
        existing_label = df[label_column].value_counts()

        if isinstance(keep_label, int):
            keep_label = existing_label.head(keep_label).index.to_list()
        elif isinstance(keep_label, list|tuple|set):
            if not set(keep_label) <= set(existing_label.index):
                raise ValueError(f"{keep_label} is not a subset of existing labels ({set(existing_label.index)})")
        else:
            raise TypeError("keep_label should be an integer or a sequence of label")
        
        if len(keep_label) < 2:
            raise ValueError("Need at least two labels")

        df = df[df[label_column].str.contains("|".join(keep_label))].reset_index()

        return df

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
        df_x_min = df_x.min(axis=0, numeric_only=True)

        for col, n in nunique.items():
            if n > 4 and col in df_x_mean.index:
                df_x[col] = df_x[col].apply(lambda x : min(4, (x-df_x_min[col])//(.5*df_x_mean[col])))
        
        return df_x

    @staticmethod
    def normalize_numeric(df_x: pd.DataFrame) -> pd.DataFrame:
        df_x_mean = df_x.mean(axis=0, numeric_only=True)
        df_x_std = df_x.std(axis=0, numeric_only=True)

        df_x[df_x_mean.index] = (df_x[df_x_mean.index] - df_x_mean) / df_x_std

        return df_x

    @staticmethod
    def hot_encode(df_x: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        df_x = pd.get_dummies(df_x, columns=columns, drop_first=True)
        intervals = {}
        for col in columns:
            indices = [i for i, s in enumerate(df_x.columns) if s.startswith(col+'_')]
            if indices[0] != indices[-1]:
                intervals[col] = ((indices[0], indices[-1], ))
            else:
                intervals[col] = indices[0]
        return df_x, intervals
    
    @staticmethod
    def numeric_hot_encode(df_x: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        df_exclude = df_x.drop(columns=columns)

        dummies = [df_exclude]
        for col in columns:
            unique_values = sorted(df_x[col].unique())[1:]
            for val in unique_values:
                dummies.append((df_x[col] >= val).astype(int))

        df_x_res = pd.concat(dummies, axis=1)
        return df_x_res

    # TODO multi-class
    @staticmethod
    def balance_dataset(df_x: pd.DataFrame, df_y: pd.DataFrame, label_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # nunique ?
        itrue = df_y.index[df_y[label_column]==1].tolist()
        ifalse = df_y.index[df_y[label_column]==0].tolist()

        # min then for loop ?
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

    @staticmethod
    def _get_dataset(
        cls, 
        balancing: bool = True, 
        discretizing: bool = True, 
        hot_encoding: bool = True, 
        na_handling: str = "drop", 
        rmv_pct: Union[float|bool] = False,
        keep_label: Union[int|Sequence[str]] = 2,
    ) -> Tuple[ndarray, ndarray]:

        label_column = cls.get_label_column()
        df = cls.get_df_data()
        df = cls.remove_na(cls, df, na_handling)

        if df[label_column].nunique() > 2:
            cls.multiclass_handling(df, label_column, keep_label)

        if rmv_pct:
            df = cls.remove_percentile(df, rmv_pct)
            
        df_x, df_y = cls.data_label_separation(df, label_column)
        df_y = cls.label_to_numeric(df_y, label_column)

        txt_columns = df_x.select_dtypes(exclude="number").columns
        num_columns = df_x.select_dtypes(include="number").columns

        if discretizing:
            df_x = cls.discretize_numeric(df_x)
            numeric_hot_encode_columns = num_columns
        else:
            df_x = cls.normalize_numeric(df_x)
            numeric_hot_encode_columns = None
        
        if hot_encoding:
            hot_encode_columns = txt_columns
            df_x, intervals = cls.hot_encode(df_x, hot_encode_columns)
            if numeric_hot_encode_columns is not None:
                df_x = cls.numeric_hot_encode(df_x, numeric_hot_encode_columns) # TODO intervals
            else:
                nunique = df_x[num_columns].nunique(axis=0)
                for i, (col, n) in enumerate(nunique.items()):
                    if n > 2:
                        intervals[col] = ((i,i))
                    else:
                        intervals[col] = i

        if balancing:
            df_x, df_y = cls.balance_dataset(df_x, df_y, label_column)
        
        x = df_x.to_numpy()
        y = df_y.to_numpy()
        DATASET_METADATA[cls] = intervals

        return x, y

    @classmethod
    def get_dataset(
        cls, 
        balancing: bool = True, 
        discretizing: bool = True, 
        hot_encoding: bool = True, 
        na_handling: str = "drop", 
        rmv_pct: Union[float|bool] = False,
        keep_label: Union[int|Sequence[str]] = 2,
    ) -> Tuple[ndarray, ndarray]:
        return Dataset._get_dataset(
            cls=cls,
            balancing=balancing,
            discretizing=discretizing,
            hot_encoding=hot_encoding,
            na_handling=na_handling,
            rmv_pct=rmv_pct,
            keep_label=keep_label,
        )

class ImageDataset(Dataset):
    @classmethod
    def get_dataset(
        cls, 
        balancing: bool = True,
        na_handling: str = "drop", 
        keep_label: int | Sequence[str] = 2
    ) -> Tuple[ndarray]:

        x, y = Dataset._get_dataset(
            cls=cls,
            balancing=balancing, 
            discretizing=False,
            hot_encoding=False,
            na_handling=na_handling,
            rmv_pct=False,
            keep_label=keep_label,
        )

        x = np.array(list(np.ravel(x)))
        return x, y