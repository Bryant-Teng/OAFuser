import torch.nn as nn
import torch

'''class Spatial_mask(nn.Module):
    def __init__(self, dim, reduction=1):
        super(Spatial_mask, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
                    nn.Linear(self.dim * 2, self.dim * 2 // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim * 2 // reduction, self.dim), 
                    nn.Sigmoid())

    def forward(self, x1):
        B, _, H, W = x1.shape
        avg = self.avg_pool(x1).view(B, self.dim)
        max = self.max_pool(x1).view(B, self.dim)
        y = torch.cat((avg, max), dim=1) # B 4C
        y = self.mlp(y).view(B, self.dim, 1)
        Spatial_mask = y.reshape(B, self.dim, 1, 1) # B C 1 1
        return Spatial_mask'''
    

class Ang_mask(nn.Module):
    def __init__(self, dim, reduction=1):
        super(Ang_mask, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=(dim)),
            nn.Conv2d(dim, 1, 1),
            nn.Sigmoid())
    def forward(self, x_center, x_view):
        x_row = x_center
        for items in x_view:
            #Difference = self.mlp(torch.abs(x_center-items))
            Difference = torch.abs(x_center-items)
            Ang_mask = (1 - Difference)**2
            x_row = x_row + items*Ang_mask
        return x_row


#def LF_Fusion(x_center,x_view):
#    Ang_mask = Ang_mask(x_center,x_view)
#    view = x_center + x_view*Ang_mask
#    return view

'''def LF_Fusion(self,x_center,x_view,model_A,model_S):
    Ang_mask = model_A(x_center,x_view)
    view = x_center+x_view*Ang_mask*Spa_mask
    ###这里需要一个正则化'''




    