import os
import torch
import numpy as np
import random
import os
import yaml
import json

from tools.optimization import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def ranking_lossT(logitsT, labelsT): 

    # Refer: https://github.com/akshitac8/BiAM

    eps = 1e-8
    subset_idxT = torch.sum(torch.abs(labelsT),dim=0) 
    subset_idxT = torch.nonzero(subset_idxT>0).view(-1).long().cuda() 
    sub_labelsT = labelsT[:,subset_idxT] 
    sub_logitsT = logitsT[:,subset_idxT] 
    positive_tagsT = torch.clamp(sub_labelsT,0.,1.) 
    negative_tagsT = torch.clamp(-sub_labelsT,0.,1.) 
    maskT = positive_tagsT.unsqueeze(1) * negative_tagsT.unsqueeze(-1) 
    pos_score_matT = sub_logitsT * positive_tagsT 
    neg_score_matT = sub_logitsT * negative_tagsT 
    IW_pos3T = pos_score_matT.unsqueeze(1) 
    IW_neg3T = neg_score_matT.unsqueeze(-1) 
    OT = 1 + IW_neg3T - IW_pos3T
    O_maskT = maskT * OT
    diffT = torch.clamp(O_maskT, 0) 
    violationT = torch.sign(diffT).sum(1).sum(1) 
    diffT = diffT.sum(1).sum(1) 
    lossT =  torch.mean(diffT / (violationT+eps))

    return lossT

def ranking_loss_sigmoid(predictions,target):
    positive_score = torch.log(torch.sigmoid(predictions))
    negtive_score =  torch.log(1-torch.sigmoid(predictions))

    positive_score = target * positive_score
    negtive_score = (1-target) * negtive_score

    loss = -positive_score-negtive_score

    loss = loss.mean()

    return loss


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_args(filename, args):
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    for key, group in data_loaded.items():
        for key, val in group.items():
            setattr(args, key, val)


def write_json(filename, content):
    with open(filename, 'w') as f:
        json.dump(content, f)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def get_optimizer(model, config):
    if config.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    return optimizer


def get_scheduler(optimizer, config, num_batches=-1):
    if not hasattr(config, 'scheduler') :
        return None
    if config.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    elif config.scheduler == 'linear_w_warmup' or config.scheduler == 'cosine_w_warmup':
        assert num_batches != -1
        num_training_steps = num_batches * config.epochs
        num_warmup_steps = int(config.warmup_proportion * num_training_steps)
        if config.scheduler == 'linear_w_warmup':
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
        if config.scheduler == 'cosine_w_warmup':
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    else:
        return None
    
    
    return scheduler


def step_scheduler(scheduler, config, bid, num_batches):
    if not hasattr(config, 'scheduler') or scheduler==None:
        return scheduler
    elif config.scheduler in ['StepLR']:
        if bid + 1 == num_batches:    # end of the epoch
            scheduler.step()
    elif config.scheduler in ['linear_w_warmup', 'cosine_w_warmup']:
        scheduler.step()

    return scheduler
