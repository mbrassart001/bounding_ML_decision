import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import time
import torch
import random
import numpy as np
from typing import Any, Tuple, Dict, Callable
from pyeda.boolalg.bdd import BDDVariable

import datasets
from utils.modules import EncodingLayer
from utils.model import GlobalModel, RobddModel, MultiRobddModel

from exp.net_utils import match_inputs, train, compute_metrics, approx2robdd
from exp.parse import parse_rmv_features, parse_model

class ExpDataGetter:
    def __init__(self, filename: str) -> None:
        with open(filename, 'r') as f: 
            self.EXP_DESC = yaml.load(f, Loader)

    def get_seed(self, default_seed: int=0) -> int:
        return self.get_exp_param("seed", default=default_seed)

    def get_exp_param(self, *path: str, default: Any=None) -> Any:
        param = self.EXP_DESC
        for name in path:
            param = param.get(name, default)
        return param

    def get_dataset_metadata(self) -> Tuple[np.array, np.array, Dict[str, Tuple[int, int] | int]]:
        dataset_class = datasets.get(self.get_exp_param("dataset", "name"))
        dataset_args = self.get_exp_param("dataset", "args")
        rmv_features = parse_rmv_features(dataset_args.get("remove"))
        dataset_args["remove"] = rmv_features
        np_x, np_y = dataset_class.get_dataset(**dataset_args)
        self.metadata = dataset_class.get_metadata(rmv_features)
        return np_x, np_y, self.metadata

    def get_encoding(self, metadata: Dict[str, Tuple[int, int] | int]) -> Tuple[Dict[int|str, int], int, Dict[str, Dict|int]]:
        encoding_sizes = self.get_exp_param("encoding", "sizes")
        default_encoding_size = self.get_exp_param("encoding", "default_encoding_size")
        encoded_total_size = EncodingLayer.total_encoding_size(metadata, encoding_sizes, default_encoding_size)
        kwargs_encoding = {"metadata": metadata, "output_sizes": encoding_sizes, "default_encoding_size": default_encoding_size}
        return encoding_sizes, encoded_total_size, kwargs_encoding

    def get_robdd_rmv(self, robdd_desc: Dict[str, Any], metadata: Dict[str, Tuple[int, int] | int], encoding_sizes: Dict[int|str, int]) -> list[BDDVariable]:
        try:
            robdd_rmv_features = parse_rmv_features(robdd_desc.get("remove"))
            all_inputs = match_inputs(metadata, encoding_sizes)
            remove_inputs = [input for x in robdd_rmv_features for input in all_inputs.get(x)]
        except TypeError:
            print(robdd_rmv_features, all_inputs)
            raise Exception
        return remove_inputs

class IterDataGetter(ExpDataGetter):
    def __init__(self, filename, seed_list: list[int]=None) -> None:
        super().__init__(filename)
        self._seed_iter = iter(seed_list) if seed_list is not None else None
        self._seed = None

    def get_seed(self, default_seed: int=0) -> int:
        return self._seed if self._seed is not None else default_seed

    def __iter__(self) -> None:
        if self._seed_iter is None:
            seeds = super().get_seed()
            if isinstance(seeds, int):
                seeds = [seeds]
            self._seed_iter = iter(seeds)
        return self

    def __next__(self) -> None:
        if self._seed_iter is None:
            raise StopIteration
        self._seed = next(self._seed_iter)
        return self

class ExpBase:
    def __init__(self, datagtr: ExpDataGetter) -> None:
        self.exp = datagtr

    def seed_all(self, default_seed: int=0) -> None:
        seed = self.exp.get_seed(default_seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def train(self, verbose: bool=False) -> None:
        if verbose: print("Model training started ...", end="\t", flush=True); ts=time.perf_counter()
        train(self.model, self.exp.get_exp_param("training"), *self.data.values())
        if verbose: print("Training completed in %0.3fs"%(time.perf_counter()-ts))

    def metrics_train(self, model: torch.nn.Module=None) -> Dict[str, float]:
        model = model or self.model
        return compute_metrics(model, self.data["x_train"], self.data["y_train"])

    def metrics_valid(self, model: torch.nn.Module=None) -> Dict[str, float]:
        model = model or self.model
        return compute_metrics(model, self.data["x_valid"], self.data["y_valid"])

    def robdd_verif(self, bdd_model: RobddModel|MultiRobddModel) -> None:
        net_metrics = self.metrics_valid()
        bdd_metrics = self.metrics_valid(bdd_model)
        assert all(net_metrics[k]==bdd_metrics[k] for k in net_metrics)

    def robdd_remove_methods(self, remove_inputs: list[BDDVariable]) -> Dict[str, Callable]:
        raise NotImplementedError

    def dataprep(self) -> None:
        # get dataset and metadata
        np_x, np_y, self.metadata = self.exp.get_dataset_metadata()
        x_data, y_data = torch.Tensor(np_x), torch.Tensor(np_y)
        data_size = x_data.size(1)

        # split dataset
        train_index, valid_index = torch.utils.data.random_split(range(x_data.size(0)), [0.9, 0.1])
        x_train, y_train = x_data[train_index], y_data[train_index]
        x_valid, y_valid = x_data[valid_index], y_data[valid_index]
        self.data = {
            "x_train": x_train, "y_train": y_train, 
            "x_valid": x_valid, "y_valid": y_valid,
        }

        # get encoding
        encoding_sizes, encoded_total_size, kwargs_encoding = self.exp.get_encoding(self.metadata)
        self.encoding = {
            "sizes": encoding_sizes,
            "total_size": encoded_total_size,
            "kwargs": kwargs_encoding,
        }

        # get model
        self.model = parse_model(self.exp.get_exp_param("model"), data_size, encoded_total_size, kwargs_encoding)

    def robdd_remove(self, verbose: bool=False) -> None:
        robdd_desc = self.exp.get_exp_param("robdd", default=None)
        if robdd_desc is None:
            return
        remove_inputs = self.exp.get_robdd_rmv(robdd_desc, self.metadata, self.encoding["sizes"])
        remove_methods = self.robdd_remove_methods(remove_inputs)

        new_modules = {"big": self.model.get_module("big")}
        labels = remove_methods.keys()
        for label in labels:
            if verbose: print(f"Converting {label} to ROBDD ...", end="\t", flush=True); ts = time.perf_counter()
            module = self.model.get_module(label)
            robdd = approx2robdd(module)
            new_modules.update({label: robdd})
            if verbose: print("%s converted in %0.3fs"%(label.capitalize(), time.perf_counter()-ts))
        
        new_model = GlobalModel(**new_modules)
        self.robdd_verif(new_model)

        for label, method in remove_methods.items():
            method(new_modules[label])
        
        self.model = new_model

    def main(self, verbose: bool=False) -> Dict[str, float]:
        self.seed_all()
        self.dataprep()
        self.train(verbose)
        self.robdd_remove(verbose)
        return self.metrics_valid()
