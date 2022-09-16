# ====================================================
# Library
# ====================================================
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import CFG



class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, margin=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output



# ====================================================
# MODEL
# ====================================================
class CustomModel(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.cfg = cfg
        self.model = timm.create_model(self.cfg.model_name, pretrained=pretrained, in_chans=3)
        
        if cfg.model_name in ['tf_efficientnetv2_b0', 'tf_efficientnet_b5', 'tf_efficientnet_b2']:
            self.in_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
            self.model.global_pool = nn.Identity()
            
        if cfg.model_name in ['resnext50_32x4d', 'resnet50d', 'resnet34d']:
            self.in_features = self.model.fc.in_features
            #self.model.fc = nn.Linear(self.in_features, self.cfg.fc_dim)
            self.model.fc = nn.Identity()
            self.model.global_pool = nn.Identity()
            
        if cfg.model_name == 'tresnet_m':
            self.in_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Linear(self.in_features, self.cfg.fc_dim)
            
        elif cfg.model_name.split('_')[0] == 'vit':
            self.in_features = self.model.head.in_features
            self.model.head = nn.Linear(self.in_features, self.cfg.fc_dim)
        
        
        #self.pooling = GeM()
        self.pooling =  nn.AdaptiveAvgPool2d(1) # GAP
        self.probs = nn.Linear(self.in_features, self.cfg.target_size)
        self.bn = nn.BatchNorm1d(self.in_features) # BNNeck
        self.final = ArcMarginProduct(
            self.in_features,
            cfg.target_size,
            scale = cfg.scale,
            margin = cfg.margin,
            easy_margin = False,
            ls_eps = 0.0
        )

    def forward(self, x, label):
        batch_size = x.shape[0]
        # model backbone shape: torch.Size([4, 2048, 8, 8])
        features = self.model(x)
        # gap shape: torch.Size([4, 2048])
        features = self.pooling(features).view(batch_size, -1)
        arcface = self.final(features, label)
        bn = self.bn(features)
        probs   = self.probs(bn)
        return probs, arcface