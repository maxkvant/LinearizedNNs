import torch
import torch.nn as nn


def set_bn_eval(m: nn.Module):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        print(classname)
        m.eval()


def warm_up_batch_norm(model: nn.Module, dataloader, device):
    for batch_id, (X, y) in enumerate(dataloader):
        X = X.to(device)
        model.forward(X)
    model.apply(set_bn_eval)
    return model


def to_one_hot(y, num_classes):
    return torch.eye(num_classes)[y] * 2. - 1.
