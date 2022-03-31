import torch
import torch.nn as nn
import torch.nn.functional as F


class TanhAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # self.dropout = nn.Dropout(dropout)
        self.ws1 = nn.Linear(d_model, d_model, bias=True)
        self.ws2 = nn.Linear(d_model, d_model, bias=False)
        self.wst = nn.Linear(d_model, 1, bias=False)

    def reset_parameters(self):
        self.ws1.reset_parameters()
        self.ws2.reset_parameters()
        self.wst.reset_parameters()

    def forward(self, x, memory, memory_mask=None, fast_weights=None, **kwargs):
        if fast_weights is None:
            item1 = self.ws1(x)  # [nb, len1, d]
            item2 = self.ws2(memory)  # [nb, len2, d]
            # print(item1.shape, item2.shape)
            item = item1.unsqueeze(2) + item2.unsqueeze(1)  # [nb, len1, len2, d]
            S = self.wst(torch.tanh(item)).squeeze(-1)  # [nb, len1, len2]
        else:
            item1 = F.linear(x, fast_weights['ws1.weight'], fast_weights['ws1.bias'])  # [nb, len1, d]
            item2 = F.linear(memory, fast_weights['ws2.weight'])  # [nb, len2, d]
            # print(item1.shape, item2.shape)
            item = item1.unsqueeze(2) + item2.unsqueeze(1)  # [nb, len1, len2, d]
            S = F.linear(torch.tanh(item), fast_weights['wst.weight']).squeeze(-1)  # [nb, len1, len2]
        if memory_mask is not None:
            memory_mask = memory_mask.unsqueeze(1)  # [nb, 1, len2]
            S = S.masked_fill(memory_mask == 0, float('-inf'))
        S = F.softmax(S, -1)
        return torch.matmul(S, memory), S  # [nb, len1, d]
