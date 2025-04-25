import torch
from utils.custom_loss import AsymBCELoss
from utils.model import ApproxModel, MultiApprox, BaseModel, GlobalModel

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