import torch.nn as nn
import torch.nn.functional as F

from models.modules import MultiheadAttention


# TransformerEncoder
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        non_padding_mask = None if mask is None else 1 - mask
        x = x.transpose(0, 1)
        for layer in self.encoder_layers:
            x = layer(x, non_padding_mask)
        x = x.transpose(0, 1)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        d_model = d_model
        num_heads = num_heads
        self.dropout = dropout

        self.self_attn = MultiheadAttention(d_model, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model << 1)
        self.fc2 = nn.Linear(d_model << 1, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        dim = x.size(0)

        attn_mask = None if self.attn_mask is None else self.attn_mask.cuda()[:dim, :dim]
        res = x
        x, weight = self.self_attn(x, x, x, mask, attn_mask=attn_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = res + x
        x = self.self_attn_layer_norm(x)

        res = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = res + x
        x = self.final_layer_norm(x)
        return x
