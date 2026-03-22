import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc_gate1 = nn.Linear(d_model, d_model, bias=False)
        self.fc_gate2 = nn.Linear(d_model, d_model, bias=False)

    def reset_parameters(self):
        self.fc_gate1.reset_parameters()
        self.fc_gate2.reset_parameters()

    def forward(self, x1, x2, fast_weights=None, **kwargs):
        if fast_weights is None:
            g1 = torch.sigmoid(self.fc_gate1(x1))
            x2_ = g1 * x2
            g2 = torch.sigmoid(self.fc_gate2(x2))
            x1_ = g2 * x1
        else:
            g1 = torch.sigmoid(F.linear(x1, fast_weights['fc_gate1.weight']))
            x2_ = g1 * x2
            g2 = torch.sigmoid(F.linear(x2, fast_weights['fc_gate2.weight']))
            x1_ = g2 * x1
        return x1_, x2_
