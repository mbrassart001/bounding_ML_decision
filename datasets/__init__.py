from .diabetes import DiabetesDataset
from .loan import LoanDataset
from .dry_bean import DryBeanDataset
from .obesity import ObesityDataset
from .mnist import MnistDataset
from .adult import AdultDataset
from .bank_marketing import BankMarketingDataset
from .contraceptive import ContraceptiveDataset
from .compas import CompasDataset
from .titanic import TitanicDataset
from .synthetic import SyntheticDataset

from .base_dataset import Dataset

DATASETS = {
    "diabetes": DiabetesDataset,
    "loan": LoanDataset,
    "drybean": DryBeanDataset,
    "obesity": ObesityDataset,
    "mnist": MnistDataset,
    "adult": AdultDataset,
    "bankmarketing": BankMarketingDataset,
    "contraceptive": ContraceptiveDataset,
    "compas": CompasDataset,
    "titanic": TitanicDataset,
    "synthetic": SyntheticDataset,
}

def get(name: str) -> Dataset:
    name = name.lower().replace("_", "")
    dataset = DATASETS.get(name)
    if dataset is None:
        raise ValueError(f"{name} not in available datasets. ({', '.join(DATASETS.keys())}")
    elif dataset is SyntheticDataset:
        dataset = dataset()
    return dataset
    