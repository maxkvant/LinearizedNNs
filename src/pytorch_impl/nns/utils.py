import torch
import torch.nn as nn


def set_bn_eval(m: nn.Module):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        print(f"eval {classname}")
        m.eval()


def set_bn_train(m: nn.Module):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        print(f"train {classname}")
        m.train()


def warm_up_batch_norm(model: nn.Module, dataloader, device, limit=None):
    for batch_id, (X, y) in enumerate(dataloader):
        if (limit is not None) and (batch_id >= limit):
            break

        X = X.to(device)
        model.forward(X)
    model.apply(set_bn_eval)
    return model


def to_one_hot(y, num_classes):
    return torch.eye(num_classes)[y] * 2. - 1.


def to_normal(layer):
    sigma = torch.abs(layer.weight.data).mean() * 2
    layer.weight.data.normal_(sigma)
    return layer
