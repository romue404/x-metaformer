def exclude_from_wt_decay(named_params, weight_decay, skip_list=("temp", "temperature", "scale", "norm", "bias")):
    params = []
    excluded_params = []

    for name, param in named_params:
        if not param.requires_grad:
            continue
        if any(layer_name in name for layer_name in skip_list):
            excluded_params.append(param)
            # print(f"skipped param {name}")
        else:
            params.append(param)
    return [
        {"params": params, "weight_decay": weight_decay},
        {"params": excluded_params, "weight_decay": 0.0},
    ]