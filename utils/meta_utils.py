def get_parameters(model):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        params = []
        for name, param in model._parameters.items():
            if param is not None and param.requires_grad:
                params.append(param)
        return params
        # return [child for child in model._parameters.values() if child.requires_grad]
    else:
        # look for children from children... to the last child!
        for child in children:
            try:
                flatt_children.extend(get_parameters(child))
            except TypeError:
                flatt_children.append(get_parameters(child))
    return flatt_children


def detach_parameters(model):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        for name, param in model._parameters.items():
            if param is not None and param.requires_grad:
                model._parameters[name] = model._parameters[name].detach()
                model._parameters[name].requires_grad = True
                # return [child for child in model._parameters.values() if child.requires_grad]
    else:
        # look for children from children... to the last child!
        for child in children:
            try:
                flatt_children.extend(detach_parameters(child))
            except TypeError:
                flatt_children.append(detach_parameters(child))
    return flatt_children


def set_parameters(model, params):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        for name, param in model._parameters.items():
            if param is not None and param.requires_grad:
                model._parameters[name] = next(params)
    else:
        # look for children from children... to the last child!
        for child in children:
            try:
                flatt_children.extend(set_parameters(child, params))
            except TypeError:
                flatt_children.append(set_parameters(child, params))
    return flatt_children


def clone_parameters(model, params):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        for name, param in model._parameters.items():
            if param is not None and param.requires_grad:
                model._parameters[name] = next(params).clone()
    else:
        # look for children from children... to the last child!
        for child in children:
            try:
                flatt_children.extend(clone_parameters(child, params))
            except TypeError:
                flatt_children.append(clone_parameters(child, params))
    return flatt_children


def sgd_step(model, grads, lr=0.001):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        for name, param in model._parameters.items():
            if param is not None and param.requires_grad:
                grad = next(grads)
                if grad is not None:
                    model._parameters[name] = model._parameters[name] - lr * grad
    else:
        # look for children from children... to the last child!
        for child in children:
            try:
                flatt_children.extend(sgd_step(child, grads, lr=lr))
            except TypeError:
                flatt_children.append(sgd_step(child, grads, lr=lr))
    return flatt_children

