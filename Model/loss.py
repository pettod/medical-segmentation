import torch
import torch.nn as nn
import numpy as np
from typing import List

class ImageBasedCrossEntropyLoss2d(nn.Module):

    def __init__(self, classes, weight=None, size_average=True, ignore_index=255,
                 norm=False, upper_bound=1.0):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = False

    def calculateWeights(self, target):
        hist = np.histogram(target.flatten(), range(
            self.num_classes + 1), normed=True)[0]
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def forward(self, inputs, targets):
        target_cpu = targets.data.cpu().numpy()
        if self.batch_weights:
            weights = self.calculateWeights(target_cpu)
            self.nll_loss.weight = torch.Tensor(weights).cuda()

        loss = 0.0
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculateWeights(target_cpu[i])
                self.nll_loss.weight = torch.Tensor(weights).cuda()
            
            loss += self.nll_loss(nn.functional.log_softmax(inputs[i].unsqueeze(0)),
                                          targets[i].unsqueeze(0))
        return loss

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)

def dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if len(true.shape) == 2:
        true = true.unsqueeze(0).unsqueeze(0)
    elif len(true.shape) == 3:
        true = true.unsqueeze(1)
    
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = nn.functional.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)

class LabelSmoothSoftmaxCE(nn.Module):
    def __init__(self,
                 lb_pos=0.9,
                 lb_neg=0.005,
                 reduction='mean',
                 lb_ignore=-1,
                 ):
        super(LabelSmoothSoftmaxCE, self).__init__()
        self.lb_pos = lb_pos
        self.lb_neg = lb_neg
        self.reduction = reduction
        self.lb_ignore = lb_ignore
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, logits, label):
        logs = self.log_softmax(logits)
        ignore = label.data.cpu() == self.lb_ignore
        n_valid = (ignore == 0).sum()
        label = label.clone()
        label[ignore] = 0
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        label = self.lb_pos * lb_one_hot + self.lb_neg * (1-lb_one_hot)

        loss = -torch.sum(logs*label, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            loss = loss
        return loss


class DualLoss(nn.Module):
    def __init__(self, num_classes=4, lmbda=10, epsilon=10e-6, mode="train"):
        super(DualLoss, self).__init__()
        self.epsilon = epsilon
        self.lmbda = lmbda
        self.channels = num_classes 
        self.CELoss = nn.CrossEntropyLoss().cuda()
        # self.CELoss = LabelSmoothSoftmaxCE()
        self.bce2d = nn.BCELoss().cuda()
        self.epoch = 1 
        self.alpha = 1.0
        '''
        if mode == "train":
            self.seg_loss = ImageBasedCrossEntropyLoss2d(
                        classes=num_classes, upper_bound=1.0).cuda()
        else:
            self.seg_loss = self.seg_loss = CrossEntropyLoss2d(size_average=True).cuda()
        '''

    def edge_attention(self, x, target, edge):
        n, c, h, w = x.size()
        filler = torch.ones_like(target) * 255
        return self.seg_loss(x,
                             torch.where(edge.max(1)[0].cpu() > 0.8, target, filler).cuda().long())

    def forward(self, pred, target, epoch=0):
        seg, edge_in = pred
        seg_t, edge_t = target
        #seg
        ce = self.CELoss(seg, seg_t.long().cuda())
        dice = dice_loss(seg_t.long().cuda(), seg)
        #edge
        edge = self.bce2d(edge_in.squeeze(1), edge_t.cuda())
        #att = self.edge_attention(seg, seg_t, edge_in)

        return dice + ce + edge# + 0.1*att

    def index_to_bitmask(self, target):
        assert len(target.shape) == 3 #NHW
        
        target = target.unsqueeze(1) #NCHW
        tmp = []    
        for i in range(self.channels):
            tmp.append((target==i).long())
        
        #print(torch.cat(tmp, 1).shape)
        return torch.cat(tmp, 1)
