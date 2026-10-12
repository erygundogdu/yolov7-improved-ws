import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss, ComputeLossOTA
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

# ###load the model
# device = torch.device('cuda')
# ckpt = torch.load("model.pt", map_location=device, weights_only=False)
# model = ckpt["model"].float().to(device).eval()
# img = torch.randn(1,3,640,640, device=device)

# # for p in model.named_parameters():
# #     print(p[0], "num of params:", p[1].numel())

# pred,x = model(img)
# for i,f in enumerate(x):
#     print(i, f.shape)

def get_teacher(model_pth, device):
    ckpt = torch.load(model_pth, map_location=device, weights_only=False)
    model = ckpt["model"].float().to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def loss_kd(pred_s, pred_t,weights = torch.ones(3,device='cuda'),ce = True): ##list of tensors 
    ###input target labels from fucntion input
    lobj = torch.zeros(1,device=weights.device)
    lbbox = torch.zeros(1,device=weights.device)
    lcls = torch.zeros(1,device=weights.device)
    l_total = torch.zeros(1,device=weights.device)

    for i in range(len(pred_s)):
        obj_s = pred_s[i][...,4]
        obj_t = pred_t[i][...,4].sigmoid()

        bbox_s = pred_s[i][...,0:4]
        bbox_t = pred_t[i][...,0:4]

        cls_s = pred_s[i][...,5:]
        cls_t = pred_t[i][...,5:].sigmoid()

        ###check if the current input has object
        ###
        msk = obj_t > 0.20
        if ce:
            lobj = lobj + F.binary_cross_entropy_with_logits(obj_s,obj_t,reduction="mean")
            if msk.any():
                lbbox = lbbox + F.smooth_l1_loss(bbox_s[msk],bbox_t[msk], reduction="mean")
                lcls = lcls + F.binary_cross_entropy_with_logits(cls_s[msk],cls_t[msk],reduction="mean")
        else:
            # tau = 1.0
            # obj_t_p = F.log_softmax(pred_t[i][...,4] / tau,dim=-1)
            # obj_s_p = F.log_softmax(pred_s[i][...,4] / tau,dim=-1)
            # cls_s_p = F.log_softmax(pred_s[i][...,5:] / tau, dim=-1)
            # cls_t_p = F.log_softmax(pred_t[i][...,5:] / tau, dim=-1)
            lobj = lobj + F.mse_loss(pred_s[i][...,4],pred_t[i][...,4],reduction="mean")
            if msk.any():
                lbbox = lbbox + F.smooth_l1_loss(bbox_s[msk],bbox_t[msk], reduction="mean")
                lcls = lcls + F.mse_loss(pred_s[i][...,5:],pred_t[i][...,5:],reduction="mean")

    lobj = lobj / len(pred_s)
    lbbox = lbbox / len(pred_s)
    lcls = lcls / len(pred_s)
    l_list = [lobj,lbbox,lcls]

    l_total = weights[0] * lobj + weights[1] * lbbox + weights[2] * lcls


    return l_list, l_total

# pred_s  = torch.randn(1,3,640,640,10)
# pred_s_l = []
# pred_t_l = []
# for i in range(3):
#     pred_s_l.append(pred_s)
#     pred_t_l.append(pred_s)
# _, loss = loss_kd(pred_s_l,pred_t_l)


# def loss_kd(pred_s, pred_t, weights=torch.ones(3, device='cuda')):
#     lobj = torch.zeros(1, device=weights.device)
#     lbbox = torch.zeros(1, device=weights.device)
#     lcls = torch.zeros(1, device=weights.device)

#     bg_floor = 0.1

#     for i in range(len(pred_s)):
#         obj_s = pred_s[i][..., 4]
#         obj_t = pred_t[i][..., 4].sigmoid()

#         bbox_s = pred_s[i][..., 0:4]
#         bbox_t = pred_t[i][..., 0:4]

#         cls_s = pred_s[i][..., 5:]
#         cls_t = pred_t[i][..., 5:].sigmoid()

#         w = obj_t
#         w_exp = w.unsqueeze(-1)

#         obj_map = F.binary_cross_entropy_with_logits(obj_s, obj_t, reduction="none")
#         lobj = lobj + (obj_map * (bg_floor + w)).sum() / (bg_floor + w).sum().clamp_min(1.0)

#         box_map = F.smooth_l1_loss(bbox_s, bbox_t, reduction="none")
#         lbbox = lbbox + (box_map * w_exp).sum() / w_exp.sum().clamp_min(1.0)

#         cls_map = F.binary_cross_entropy_with_logits(cls_s, cls_t, reduction="none")
#         lcls = lcls + (cls_map * w_exp).sum() / w_exp.sum().clamp_min(1.0)

#     lobj /= len(pred_s)
#     lbbox /= len(pred_s)
#     lcls /= len(pred_s)

#     return [lobj, lbbox, lcls], weights[0] * lobj + 0.5 * lbbox + weights[2] * lcls

# lobj = loss_kd(x,x)
# # print(lobj)

# sample = torch.tensor([0.8,0.1, 0.1]).float()
# sample = sample.unsqueeze(0).unsqueeze(0).repeat(1,4,1)
# res = F.binary_cross_entropy(sample, sample,reduction="mean")
# print(res)


   
