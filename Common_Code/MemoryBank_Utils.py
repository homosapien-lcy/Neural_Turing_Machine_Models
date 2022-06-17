from collections import OrderedDict

import torch

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.nn.init import xavier_uniform
from torch.nn.init import xavier_normal
from torch.nn.init import kaiming_uniform
from torch.nn.init import kaiming_normal

from inspect import signature
import numpy as np
import random
import scipy.optimize as skopt

eps = 1e-5
revise_first_key = True
softmax_normalize = lambda x: F.softmax(x, dim=1)
sum_normalize = lambda x: (x / torch.sum(x, dim=1))
requires_grad_filter = lambda params: filter(lambda p: p.requires_grad, params)


def normalize(mat, p, d=1, eps=eps):
    # p=2 norm, sq sum=1, p=1 norm, sum=1
    norm_factor = torch.norm(mat, p=p, dim=d).view(-1, 1)
    return torch.div(mat, norm_factor + eps)


def normalize_vec(vec, p=1, eps=eps):
    factor = torch.norm(vec, p=p)
    return (vec / (factor.item() + eps))


def standardize(mat):
    mat_dim_1 = mat.size(1)
    mean = mat.mean(dim=1)
    std = mat.std(dim=1)
    centered = mat - torch.stack([mean[:]] * mat_dim_1, dim=1)
    standardized = centered / torch.stack([std[:]] * mat_dim_1, dim=1)

    return standardized


def make_one_hot(labels, C=2, device="gpu"):
    labels_reshape = labels.view(-1, 1)
    if device == "gpu":
        one_hot = torch.cuda.FloatTensor(labels_reshape.size(0), C).zero_()
    else:
        one_hot = torch.FloatTensor(labels_reshape.size(0), C).zero_()
    return one_hot.scatter_(1, labels_reshape.data, 1)


# check nan in mat
def check_mat_nan(mat):
    return (torch.sum(torch.isnan(mat)) > 0)


def nan_to_zero_helper(mat, regenerator):
    if check_mat_nan(mat):
        replacement = regenerator(mat.size())
        # wrap replacement in nan_to_zero, prevent from regen
        # nans! use cpu as device allows
        replacement = nan_to_zero(mat=replacement,
                                  regenerator=regenerator)

        return replacement

    return mat


# modify nan mats
def nan_to_zero(mat, device="gpu", regenerator=torch.zeros):
    if check_mat_nan(mat):
        print("observe an nan in the gradient!!!!!")

        replacement = nan_to_zero_helper(mat=replacement,
                                         regenerator=regenerator)

        if device == "gpu":
            replacement = replacement.cuda()
        else:
            replacement = replacement.cpu()

        return replacement

    return mat
