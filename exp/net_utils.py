import torch
from sklearn import metrics
from pyeda.inter import bddvar
from parse import parse_criterion, parse_optimizer
from utils.misc import train_model
from utils.model import MultiApprox, ApproxModel, RobddModel, MultiRobddModel
from compiling_nn.build_odd import compile_nn

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
        epochs = sub_desc.get("epochs", default_epochs)

        train_model(x_train, y_train, module, criterion, optimizer, epochs, x_valid, y_valid)

def compute_metrics(model: torch.nn.Module, x_valid, y_valid):
    model.eval()
    scores = {}

    glob_pred = model.forward(x_valid).detach()
    scores["glob_precision"] = metrics.precision_score(glob_pred, y_valid, average="macro", zero_division=0)
    scores["glob_coverage0"], scores["glob_coverage1"] = metrics.recall_score(y_valid, glob_pred, average=None, zero_division=0).tolist()
    try:
        up_pred   = model.forward_up_only(x_valid).detach()
        _down_pred = model.forward_down_only(x_valid).detach()
        down_pred = torch.where(up_pred > 0.5, up_pred, _down_pred) # up takes priority over down when positive
        scores["up_coverage"] = metrics.recall_score(y_valid, up_pred, pos_label=1, average="binary", zero_division=0)
        scores["down_coverage"] = metrics.recall_score(y_valid, down_pred, pos_label=0, average="binary", zero_division=0)
        scores["up_rel_coverage"] = scores["up_coverage"]/scores["glob_coverage1"] if scores["glob_coverage1"] != 0 else 0
        scores["down_rel_coverage"] = scores["down_coverage"]/scores["glob_coverage0"] if scores["glob_coverage0"] != 0 else 0
    except AttributeError:
        pass

    return scores

def approx2robdd(module: ApproxModel|MultiApprox) -> RobddModel|MultiRobddModel:
    if isinstance(module, ApproxModel):
        robdd = RobddModel(compile_nn(module.net), module.enc)
    elif isinstance(module, MultiApprox):
        robdd_dict = {name: approx2robdd(submodule) for name, submodule in module.named_apx()}
        robdd = MultiRobddModel(robdd_dict, module.net.logic)
    else:
        raise TypeError()
    return robdd

def matching_inputs(metadata: dict[int|str, int|tuple[int,int]], encoding_sizes: dict[int|str, int]):
    inputs = {}
    current_size = 0
    for k in metadata.keys():
        encoding_size = encoding_sizes.get(k)
        inputs.update({str(k): [bddvar(f"i{i}") for i in range(current_size, current_size + encoding_size)]})
        current_size+=(encoding_size)
    return inputs