import sys
import torch
import torch.nn.functional as F
from packaging import version
import numpy as np
from collections import OrderedDict

import torch.nn as nn

class ContentLoss(nn.Module):
    
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        
        
        
        
        self.target = target.detach()

    def forward(self, input, alpha=1.):
        self.loss = F.mse_loss(alpha*input, self.target)
        return input    

def gram_matrix_ori(input):
    a, b, c, d = input.size()  
    
    

    features = input.view(a * b, c * d)  

    G = torch.mm(features, features.t())  

    
    
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix_ori(target_feature).detach()

    def forward(self, input):
        G = gram_matrix_ori(input)
        self.loss = F.mse_loss(G, self.target)
        return input
