import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()



class AsymmetricLossOptimizedMean(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimizedMean, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        batch = x.shape[0]

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        _loss = - self.loss.sum() / x.size(0)
        _loss = _loss / y.size(1)
        return _loss
    
class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()
    

nINF = -100

class TwoWayLogSumExp(nn.Module):
    def __init__(self, Tp=4., Tn=1.):
        super(TwoWayLogSumExp, self).__init__()
        self.Tp = Tp
        self.Tn = Tn

    def forward(self, x, y):
        class_mask = (y > 0).any(dim=0)     # C
        sample_mask = (y > 0).any(dim=1)    # B

        # Calculate hard positive/negative logits
        pmask = y.masked_fill(y <= 0, nINF).masked_fill(y > 0, float(0.0))
        plogit_class = torch.logsumexp(-x/self.Tp + pmask, dim=0).mul(self.Tp)[class_mask]
        plogit_sample = torch.logsumexp(-x/self.Tp + pmask, dim=1).mul(self.Tp)[sample_mask]
    
        nmask = y.masked_fill(y != 0, nINF).masked_fill(y == 0, float(0.0))
        nlogit_class = torch.logsumexp(x/self.Tn + nmask, dim=0).mul(self.Tn)[class_mask]
        nlogit_sample = torch.logsumexp(x/self.Tn + nmask, dim=1).mul(self.Tn)[sample_mask]

        return torch.nn.functional.softplus(nlogit_class + plogit_class).mean() + \
                torch.nn.functional.softplus(nlogit_sample + plogit_sample).mean()


class OneWayLogSumExpLoss(nn.Module):
    def __init__(self, Tp=4., Tn=1.):
        super(OneWayLogSumExpLoss, self).__init__()
        self.Tp = Tp
        self.Tn = Tn

    def forward(self, predict, gt):
        class_mask = (gt > 0).any(dim=0)

        # Calculate hard positive/negative logits; set -100 -> 0 -> caculate only on positive class
        pmask = gt.masked_fill(gt <= 0, nINF).masked_fill(gt > 0, float(0.0))
        plogit_class = torch.logsumexp(-predict/self.Tp + pmask, dim=0).mul(self.Tp)[class_mask]
    
        nmask = gt.masked_fill(gt != 0, nINF).masked_fill(gt == 0, float(0.0))
        nlogit_class = torch.logsumexp(predict/self.Tn + nmask, dim=0).mul(self.Tn)[class_mask]

        return torch.nn.functional.softplus(nlogit_class + plogit_class).mean() 


class SampleWayLogSumExpLoss(nn.Module):
    def __init__(self, Tp=4., Tn=1.):
        super(SampleWayLogSumExpLoss, self).__init__()
        self.Tp = Tp
        self.Tn = Tn

    def forward(self, x, y):
        sample_mask = (y > 0).any(dim=1)    # B

        # Calculate hard positive/negative logits
        pmask = y.masked_fill(y <= 0, nINF).masked_fill(y > 0, float(0.0))
        plogit_sample = torch.logsumexp(-x/self.Tp + pmask, dim=1).mul(self.Tp)[sample_mask]
    
        nmask = y.masked_fill(y != 0, nINF).masked_fill(y == 0, float(0.0))
        nlogit_sample = torch.logsumexp(x/self.Tn + nmask, dim=1).mul(self.Tn)[sample_mask]

        return  torch.nn.functional.softplus(nlogit_sample + plogit_sample).mean()
    

class MultiLabelSoftMaxOut(nn.Module):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self,  reduction='sum'):
        super(MultiLabelSoftMaxOut, self).__init__()
        self.reduction = reduction
        
    def forward(self, logits, target):
        '''
        logits: (B, C) raw scores
        target: (B, C) multi-hot labels
        '''
        # 1. mask for positive and negative
        pos_mask = (target > 0)
        neg_mask = (target <= 0)

        # 2. set negative logits for positive class to -inf, so exp(-inf)=0
        neg_logits = logits.masked_fill(pos_mask, float('-inf'))  # for denominator
        pos_logits = logits.masked_fill(neg_mask, 0.0)  # for numerator

        # 3. use logsumexp for stability
        logsumexp_neg = torch.logsumexp(neg_logits, dim=1)  # (B,)
        
        # 4. compute loss for each positive logit: log(exp(pos) / sum(exp(neg)))
        # => pos_logits - logsumexp_neg (broadcast)
        loss_mat = pos_logits - logsumexp_neg.unsqueeze(1)  # (B, C), -inf for negs

        # 5. only sum over positive positions
        pos_num = pos_mask.sum(dim=1).float() + 1e-8  # avoid div 0
        loss_per_sample = - (loss_mat * pos_mask).sum(dim=1) / pos_num  # (B,)

        if self.reduction == 'sum':
            return loss_per_sample.sum()
        elif self.reduction == 'mean':
            return loss_per_sample.mean()
        else:
            return loss_per_sample  # no reduction
    

def multiple_label_aoc_loss(predict, 
                            target,
                            pair_positive_weight=None,
                            attr_positive_weight=None,
                            multi_loss_func_name='CrossEntropy',
                            attr_loss_weight=1.0,
                            obj_loss_weight=1.0,
                            comp_loss_weight=1.0
                            ):
    
    _, batch_attr, batch_obj, batch_target = target
    attr_logits, obj_logits,comp_logits = predict

    batch_attr = batch_attr.cuda()
    batch_obj = batch_obj.cuda()
    batch_target = batch_target.cuda()
    loss_fn = nn.CrossEntropyLoss()



    if multi_loss_func_name == 'ASL':
        multi_attr_loss_fn = AsymmetricLossOptimizedMean()
    elif multi_loss_func_name == 'CrossEntropy':
        multi_attr_loss_fn = nn.CrossEntropyLoss()
        multi_pair_loss_fn = nn.CrossEntropyLoss()
        batch_attr = batch_attr.float() / batch_attr.sum(dim=-1,keepdim=True)
        batch_target = batch_target.float() /batch_target.sum(dim=-1,keepdim=True)
    elif multi_loss_func_name == 'onewayLogSumExp':
        multi_attr_loss_fn = OneWayLogSumExpLoss()
        multi_pair_loss_fn = OneWayLogSumExpLoss()
    elif multi_loss_func_name == 'twowayLogSumExp':
        multi_attr_loss_fn = TwoWayLogSumExp()
        multi_pair_loss_fn = TwoWayLogSumExp()
    elif  multi_loss_func_name == 'MultiLabelSoftMaxOut':
        multi_attr_loss_fn = MultiLabelSoftMaxOut(reduction='mean')
        multi_pair_loss_fn = MultiLabelSoftMaxOut(reduction='mean')
    elif multi_loss_func_name == 'samplewayLogSumExp':
        multi_attr_loss_fn = SampleWayLogSumExpLoss()
        multi_pair_loss_fn = SampleWayLogSumExpLoss()
    elif multi_loss_func_name == "BCE":
        multi_attr_loss_fn = nn.BCEWithLogitsLoss(pos_weight=attr_positive_weight)
        multi_pair_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pair_positive_weight)

    loss_attr = multi_attr_loss_fn(attr_logits, batch_attr)

    loss_obj = loss_fn(obj_logits, batch_obj)

    loss_comp = multi_pair_loss_fn(comp_logits.float(), batch_target)

    loss = loss_attr * attr_loss_weight +\
            loss_obj * obj_loss_weight +\
            loss_comp * comp_loss_weight
    
    return loss


def multiple_label_ao_loss( predict, 
                            target,
                            pair_positive_weight=None,
                            attr_positive_weight=None,
                            multi_loss_func_name='CrossEntropy',
                            attr_loss_weight=1.0,
                            obj_loss_weight=1.0,
                            comp_loss_weight=1.0
                            ):
    
    _, batch_attr, batch_obj, batch_target = target
    attr_logits, obj_logits = predict[0:3]

    batch_attr = batch_attr.cuda()
    batch_obj = batch_obj.cuda()
    batch_target = batch_target.cuda()
    loss_fn = nn.CrossEntropyLoss()


    if multi_loss_func_name == 'ASL':
        multi_attr_loss_fn = AsymmetricLossOptimizedMean()
    elif multi_loss_func_name == 'CrossEntropy':
        multi_attr_loss_fn = nn.CrossEntropyLoss()
        batch_attr = batch_attr.float() / batch_attr.sum(dim=-1,keepdim=True)
        batch_target = batch_target.float() / batch_attr.sum(dim=-1,keepdim=True)
    elif multi_loss_func_name == 'onewayLogSumExp':
        multi_attr_loss_fn = OneWayLogSumExpLoss()
    elif multi_loss_func_name == 'twowayLogSumExp':
        multi_attr_loss_fn = TwoWayLogSumExp()
    elif multi_loss_func_name == 'MultiLabelSoftMaxOut':
        multi_attr_loss_fn = MultiLabelSoftMaxOut(reduction='mean')
    elif multi_loss_func_name == 'samplewayLogSumExp':
        multi_attr_loss_fn = SampleWayLogSumExpLoss()
    elif multi_loss_func_name == "BCE":
        multi_attr_loss_fn = nn.BCEWithLogitsLoss(pos_weight=attr_positive_weight)


    loss_attr = multi_attr_loss_fn(attr_logits, batch_attr)

    loss_obj = loss_fn(obj_logits, batch_obj)

    loss = loss_attr * attr_loss_weight +\
            loss_obj * obj_loss_weight
    
    return loss


def multiple_label_c_loss(predict, 
                            target,
                            pair_positive_weight=None,
                            attr_positive_weight=None,
                            multi_loss_func_name='CrossEntropy'
                            ):
    
    _, batch_attr, batch_obj, batch_target = target
    comp_logits = predict

    batch_attr = batch_attr.cuda()
    batch_obj = batch_obj.cuda()
    batch_target = batch_target.cuda()


    if multi_loss_func_name == 'ASL':
        multi_pair_loss_fn = AsymmetricLossOptimizedMean()
    elif multi_loss_func_name == 'CrossEntropy':
        multi_pair_loss_fn = nn.CrossEntropyLoss()
        batch_target = batch_target.float() / batch_attr.sum(dim=-1,keepdim=True)
    elif multi_loss_func_name == 'HardMining':
        multi_pair_loss_fn = multiple_hard_mining_topk_loss
        batch_target = batch_target.float() / batch_attr.sum(dim=-1,keepdim=True)
    elif multi_loss_func_name == 'softmaxBCE':
        comp_logits = comp_logits.softmax(dim=-1)*4 - 2
        multi_pair_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pair_positive_weight)
    elif multi_loss_func_name == 'onewayLogSumExp':
        multi_pair_loss_fn = OneWayLogSumExpLoss()
    elif multi_loss_func_name == 'twowayLogSumExp':
        multi_pair_loss_fn = TwoWayLogSumExp()
    elif multi_loss_func_name == 'MultiLabelSoftMaxOut':
        multi_pair_loss_fn = MultiLabelSoftMaxOut()
    elif multi_loss_func_name == 'samplewayLogSumExp':
        multi_pair_loss_fn = SampleWayLogSumExpLoss()
    else:
        multi_pair_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pair_positive_weight)

    loss_comp = multi_pair_loss_fn(comp_logits, batch_target)

    loss = loss_comp 
    
    return loss

'''
Strategy 1: ignore negtive samples with low scores
Strategy 2: weights for hard negative samples
'''
def multiple_hard_mining_loss(   
    logits,             # (B,C), fixed
    targets,            # (B, C), soft label e.g. [0.25, 0, 0.25, ...]
    ):
    """
    image_feats: torch.Tensor of shape (B, D)
    text_feats: torch.Tensor of shape (C, D)
    labels: torch.Tensor of shape (B, C), soft labels (each row sums to 1 or <1)
    tau: temperature
    alpha: scaling for hard negative emphasis
    """


    # Softmax denominator with hardness-aware weighting
    # weights = 1 for positives (label>0), exp(alpha * sim) for negatives
    is_positive = (targets > 0).float()
    is_negative = 1.0 - is_positive
    weights = is_positive + is_negative * torch.exp(logits)  # (B, C)

    # Compute log softmax denominator
    # 这样只是相当于简单地将负例的logits 扩大为两倍，没有意义
    weighted_denominator = torch.log((torch.exp(logits) * weights)+ 1e-9).sum(dim=1)  # (B,)

    # Numerator: use soft label weighted positive similarity
    positive_logits = logits * targets  # element-wise (B, C)
    numerator = (torch.log(positive_logits) * targets).sum(dim=1) # element-wise (B, C)

    # Final loss
    loss = - numerator + weighted_denominator
    return loss.mean()


def multiple_hard_mining_topk_loss(   
    logits,             # (B,C), fixed
    targets,            # (B, C), soft label e.g. [0.25, 0, 0.25, ...]
    ):
    """
    image_feats: torch.Tensor of shape (B, D)
    text_feats: torch.Tensor of shape (C, D)
    labels: torch.Tensor of shape (B, C), soft labels (each row sums to 1 or <1)
    tau: temperature
    alpha: scaling for hard negative emphasis
    """
    targets = targets.float()/targets.sum(dim=-1, keepdim=True)

    B, C = logits.size()

    topk =  min((targets != 0 ).sum(dim=1).long().max().item()  * 5, targets.shape[-1])

    # Get topk indices (along dim=1)
    topk_values, topk_indices = logits.topk(topk, dim=1)

    # Create mask: [B, C] = 1 only where index is in topk
    mask = torch.zeros_like(logits).bool()
    mask.scatter_(1, topk_indices, True)
    mask[targets!=0] = True  # Ensure positive targets are always included


    masked_logits = logits.masked_fill(~mask,-100.0)



    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(masked_logits, targets)

    return loss





if __name__ == "__main__":
    # Example usage
    B, C, D = 3, 6, 128

    logits = torch.randn(B, C) * 14.3

    labels = torch.tensor([
        [0.25, 0, 0.25, 0.25, 0, 0.25],
        [0.5,  0.5, 0,    0,  0, 0],
        [0,    0,   1.0,  0,  0, 0]
    ])



    loss = multiple_hard_mining_topk_loss(logits, labels)
    print("Loss:", loss.item())