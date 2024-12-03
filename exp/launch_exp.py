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
from sklearn import metrics
from pyeda.boolalg.bdd import _bdd, BDDNODEZERO, BDDNODEONE
from pyeda.inter import bddvar

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import datasets
from utils.modules import EncodingLayer
from utils.custom_loss import AsymBCELoss
from utils.misc import train_model
from utils.model import MultiApprox, GlobalModel, ApproxModel, BaseModel, RobddModel
from compiling_nn.build_odd import compile_nn

SAVE_PATH = os.path.join(os.path.abspath('..'), "backup")
PKL_PATH = os.path.join(SAVE_PATH, "bdd")
PTH_PATH = os.path.join(SAVE_PATH, "net")
METRIC_PATH = os.path.join(SAVE_PATH, "metrics")

EXP_DESC = None
def get_exp_param(*path, default = None):
    param = EXP_DESC
    for name in path:
        param = param.get(name, default)
    return param

def parse_approx(model_desc, encoded_data_size, kwargs_enc, agg_method="max", backward_method="all"):
    if model_desc is None:
        return None
    if not isinstance(model_desc, list):
        if not model_desc.get("repetition"):
            model = ApproxModel(
                encoded_data_size,
                *model_desc.get("hidden_layers"),
                **kwargs_enc
            )
            return model
        else:
            model_desc = [model_desc]
    model = MultiApprox(agg_method, backward_method)
    for apx_desc in model_desc:
        for _ in range(apx_desc.get("repetition", 1)):
            module = ApproxModel(
                encoded_data_size,
                *apx_desc.get("hidden_layers"),
                **kwargs_enc
            )
            model.add_apx(module)
    return model

def parse_net(model_desc, data_size):
    model = BaseModel(data_size, *model_desc.get("hidden_layers", list()))
    return model

def parse_model(model_desc, data_size, encoded_data_size, kwargs_enc):
    up_model = parse_approx(model_desc.get("up"), encoded_data_size, kwargs_enc, agg_method="max")
    down_model = parse_approx(model_desc.get("down"), encoded_data_size, kwargs_enc, agg_method="min")
    big_model = parse_net(model_desc.get("big"), data_size)

    if up_model is None or down_model is None:
        return big_model
    model = GlobalModel(up=up_model, down=down_model, big=big_model)
    return model

def parse_criterion(crit_desc):
    if crit_desc is None:
        raise AttributeError("criterion needs to be provided")

    crit_name = crit_desc.get("name")
    crit_args = crit_desc.get("args", None)
    if crit_name == "AsymBCELoss":
        criterion = AsymBCELoss
    else:
        criterion = getattr(torch.nn, crit_name)

    if isinstance(crit_args, dict):
        return criterion(**crit_args)
    else:
        return criterion()

def parse_optimizer(opti_desc, model):
    if opti_desc is None:
        raise AttributeError("optimizer needs to be provided")

    opti_name = opti_desc.get("name")
    opti_args = opti_desc.get("args")
    optimizer = getattr(torch.optim, opti_name)

    if isinstance(opti_args, dict):
        return optimizer(params=model.parameters(), **opti_args)
    else:
        return optimizer(model.parameters())

def parse_rmv_features(rmv_features):
    if rmv_features is None:
        return []
    elif isinstance(rmv_features, list):
        rmv_features = [str(x) for x in rmv_features]
    elif isinstance(rmv_features, str):
        rmv_features = [rmv_features]
    elif isinstance(rmv_features, int):
        rmv_features = [str(rmv_features)]
    else:
        raise ValueError()
    return rmv_features

def train(model, train_desc, x_train, y_train, x_valid, y_valid):
    default_desc = train_desc.get("default", None)
    if default_desc is not None:
        default_criterion = default_desc.get("criterion")
        default_optimizer = default_desc.get("optimizer")
        default_epochs = default_desc.get("epochs", 1)
    else:
        default_criterion = None
        default_optimizer = None
        default_epochs = 1

    for module_name in ["up", "down", "big"]:
        sub_desc = train_desc.get(module_name)
        if sub_desc is None:
            continue
        module = model.get_module(module_name)
        criterion = parse_criterion(sub_desc.get("criterion", default_criterion))
        optimizer = parse_optimizer(sub_desc.get("optimizer", default_optimizer), module)

        train_model(x_train, y_train, module, criterion, optimizer, sub_desc.get("epochs", default_epochs), x_valid, y_valid)

def compute_metrics(model: torch.nn.Module, x_valid, y_valid):
    model.eval()
    scores = {}

    glob_pred = model.forward(x_valid).detach()
    scores["glob_precision"] = metrics.precision_score(glob_pred, y_valid, average="macro", zero_division=0)
    scores["glob_coverage1"] = metrics.recall_score(glob_pred, y_valid, pos_label=1, average="binary", zero_division=0)
    scores["glob_coverage0"] = metrics.recall_score(glob_pred, y_valid, pos_label=0, average="binary", zero_division=0)

    try:
        up_pred   = model.forward_up_only(x_valid).detach()
        _down_pred = model.forward_down_only(x_valid).detach()
        down_pred = torch.where(up_pred > 0.5, up_pred, _down_pred)
        scores["up_coverage"] = metrics.recall_score(up_pred, y_valid, pos_label=1, average="binary", zero_division=0)
        scores["down_coverage"] = metrics.recall_score(down_pred, y_valid, pos_label=0, average="binary", zero_division=0)
        scores["up_rel_coverage"] = scores["up_coverage"]/scores["glob_coverage1"] if scores["glob_coverage1"] != 0 else 0
        scores["down_rel_coverage"] = scores["down_coverage"]/scores["glob_coverage0"] if scores["glob_coverage0"] != 0 else 0
    except AttributeError:
        pass

    return scores

def approx2robdd(module: ApproxModel|MultiApprox):
    if isinstance(module, ApproxModel):
        robdd = compile_nn(module.net)
    elif isinstance(module, MultiApprox):
        #TODO
        raise NotImplementedError()
    else:
        raise NotImplementedError()
    return robdd

def matching_inputs(metadata: dict[int|str, int|tuple[int,int]], encoding_sizes: dict[int|str, int]):
    inputs = {}
    current_size = 0
    for k in metadata.keys():
        encoding_size = encoding_sizes.get(k)
        inputs.update({str(k): [bddvar(f"i{i}") for i in range(current_size, current_size + encoding_size)]})
        current_size+=(encoding_size)
    return inputs

def main(filename):
    global EXP_DESC

    with open(filename) as f: 
        EXP_DESC = yaml.load(f, Loader)

    seed = get_exp_param("seed", default=0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset = datasets.get(get_exp_param("dataset", "name"))
    dataset_args = get_exp_param("dataset", "args")
    rmv_features = parse_rmv_features(dataset_args.get("remove"))
    dataset_args["remove"] = rmv_features
    np_x, np_y = dataset.get_dataset(**dataset_args)
    x_data, y_data = torch.Tensor(np_x), torch.Tensor(np_y)
    data_size = x_data.size(1)

    metadata = dataset.get_metadata(rmv_features)

    encoding_sizes = get_exp_param("encoding", "sizes")
    default_encoding_size = get_exp_param("encoding", "default_encoding_size")
    encoded_data_size = EncodingLayer.total_encoding_size(metadata, encoding_sizes, default_encoding_size)

    kwargs_enc = {"metadata": metadata, "output_sizes": encoding_sizes, "default_encoding_size": default_encoding_size}

    train_index, valid_index = torch.utils.data.random_split(range(x_data.size(0)), [0.9, 0.1])

    x_train, y_train = x_data[train_index], y_data[train_index]
    x_valid, y_valid = x_data[valid_index], y_data[valid_index]

    model = parse_model(get_exp_param("model"), data_size, encoded_data_size, kwargs_enc)

    train(model, get_exp_param("training"), x_train, y_train, x_valid, y_valid)
    # print("Model metrics", compute_metrics(model, x_valid, y_valid))

    robdd_desc = get_exp_param("robdd", default=None) 
    if robdd_desc is not None:
        try:
            robdd_rmv_features = parse_rmv_features(robdd_desc.get("remove"))
            all_inputs = matching_inputs(metadata, encoding_sizes)
            remove_inputs = [input for x in robdd_rmv_features for input in all_inputs.get(x)]
        except TypeError:
            print(robdd_rmv_features, all_inputs)
            raise Exception

        compose_zero = {k: _bdd(BDDNODEZERO) for k in remove_inputs}
        compose_one = {k: _bdd(BDDNODEONE) for k in remove_inputs}
        
        new_modules = {"big": model.get_module("big")}
        
        for label in ("up", "down"):
            # construire robdd up/down
            module = model.get_module(label)
            robdd = approx2robdd(module)

            # retirer colones dans robdd
            if label == "up":
                robdd = robdd.compose(compose_zero)
            elif label == "down":
                robdd = robdd.compose(compose_one)

            # modifier model en remplaçant up/down par robdd
            new_modules.update({label: RobddModel(robdd, module.enc)})

        model = GlobalModel(**new_modules)

    # save model

    return compute_metrics(model, x_valid, y_valid)


if __name__ == "__main__":
    print(main(sys.argv[1]))