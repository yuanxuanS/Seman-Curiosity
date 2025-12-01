
import torch.nn as nn


def get_loss_fn(loss_type):
    assert loss_type in ["bce", "l2", "l1", "xent"]
    loss_fn = None
    if loss_type == "bce":
        loss_fn = nn.BCELoss(reduction="none")
    elif loss_type == "l2":
        loss_fn = nn.MSELoss(reduction="none")
    elif loss_type == "l1":
        loss_fn = nn.L1Loss(reduction="none")
    elif loss_type == "xent":
        loss_fn = nn.CrossEntropyLoss(reduction="none")
    return loss_fn

def get_activation_fn(activation_type):
    assert activation_type in ["none", "sigmoid", "relu"]
    activation = nn.Identity()
    if activation_type == "sigmoid":
        activation = nn.Sigmoid()
    elif activation_type == "relu":
        activation = nn.ReLU()
    return activation