import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import sys
import os
import random

import torch
import numpy as np
from pyeda.boolalg.bdd import _bdd, BDDNODEZERO, BDDNODEONE

import datasets
from utils.modules import EncodingLayer
from utils.model import GlobalModel
from parse import parse_model, parse_rmv_features
from net_utils import train, matching_inputs, approx2robdd, compute_metrics

SAVE_PATH = os.path.join(os.path.abspath('..'), "backup")
PKL_PATH = os.path.join(SAVE_PATH, "bdd")
PTH_PATH = os.path.join(SAVE_PATH, "net")
METRIC_PATH = os.path.join(SAVE_PATH, "metrics")

def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

EXP_DESC = None
def get_exp_param(*path, default = None):
    param = EXP_DESC
    for name in path:
        param = param.get(name, default)
    return param

def get_dataset_metadata():
    dataset_class = datasets.get(get_exp_param("dataset", "name"))
    dataset_args = get_exp_param("dataset", "args")
    rmv_features = parse_rmv_features(dataset_args.get("remove"))
    dataset_args["remove"] = rmv_features
    np_x, np_y = dataset_class.get_dataset(**dataset_args)
    metadata = dataset_class.get_metadata(rmv_features)
    return np_x, np_y, metadata

def get_encoding(metadata):
    encoding_sizes = get_exp_param("encoding", "sizes")
    default_encoding_size = get_exp_param("encoding", "default_encoding_size")
    encoded_total_size = EncodingLayer.total_encoding_size(metadata, encoding_sizes, default_encoding_size)
    kwargs_encoding = {"metadata": metadata, "output_sizes": encoding_sizes, "default_encoding_size": default_encoding_size}
    return encoding_sizes, encoded_total_size, kwargs_encoding

def get_robdd_rmv(robdd_desc, metadata, encoding_sizes):
    try:
        robdd_rmv_features = parse_rmv_features(robdd_desc.get("remove"))
        all_inputs = matching_inputs(metadata, encoding_sizes)
        remove_inputs = [input for x in robdd_rmv_features for input in all_inputs.get(x)]
    except TypeError:
        print(robdd_rmv_features, all_inputs)
        raise Exception
    return remove_inputs

def main(filename):
    global EXP_DESC
    
    with open(filename, 'r') as f: 
        EXP_DESC = yaml.load(f, Loader)

    # seed for reproductibility
    seed = get_exp_param("seed", default=0)
    seed_all(seed)

    # get dataset and metadata
    np_x, np_y, metadata = get_dataset_metadata()
    x_data, y_data = torch.Tensor(np_x), torch.Tensor(np_y)
    data_size = x_data.size(1)

    # split dataset
    train_index, valid_index = torch.utils.data.random_split(range(x_data.size(0)), [0.9, 0.1])
    x_train, y_train = x_data[train_index], y_data[train_index]
    x_valid, y_valid = x_data[valid_index], y_data[valid_index]

    # get encoding
    encoding_sizes, encoded_total_size, kwargs_encoding = get_encoding(metadata)

    # get model and training it
    model = parse_model(get_exp_param("model"), data_size, encoded_total_size, kwargs_encoding)
    print("Model training started ...", end="\t", flush=True)
    train(model, get_exp_param("training"), x_train, y_train, x_valid, y_valid)
    print("Training completed")

    # remove features using robdd
    robdd_desc = get_exp_param("robdd", default=None) 
    if robdd_desc is not None:
        remove_inputs = get_robdd_rmv(robdd_desc, metadata, encoding_sizes)
        #TODO union/inter
        compose_zero = {k: _bdd(BDDNODEZERO) for k in remove_inputs}
        compose_one = {k: _bdd(BDDNODEONE) for k in remove_inputs}
        
        new_modules = {"big": model.get_module("big")}
        
        for label in ("up", "down"):
            # construire robdd up/down
            print(f"Converting {label} to ROBDD ...", end="\t", flush=True)
            module = model.get_module(label)
            robdd = approx2robdd(module)
            print(f"{label.capitalize()} converted")

            # retirer colonnes dans robdd
            if label == "up":
                robdd.compose(compose_zero)
            elif label == "down":
                robdd.compose(compose_one)
    
            # modifier model en remplaçant up/down par robdd
            new_modules.update({label: robdd})

        model = GlobalModel(**new_modules)

    return compute_metrics(model, x_valid, y_valid)

if __name__ == "__main__":
    print(main(sys.argv[1]))