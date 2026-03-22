import torch.nn as nn

from models.transformer.decoder import TransformerDecoder
from models.transformer.encoder import TransformerEncoder


class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_encoder_layers, num_decoder_layers, dropout=0.0):
        super().__init__()
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, num_heads, dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, dropout)

    def forward(self, src, src_mask, tgt, tgt_mask):
        enc_out = self.encoder(src, src_mask)
        out = self.decoder(enc_out, src_mask, tgt, tgt_mask)
        return out


class DualTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_decoder_layers1, num_decoder_layers2, dropout=0.0):
        super().__init__()
        self.decoder1 = TransformerDecoder(num_decoder_layers1, d_model, num_heads, dropout)
        self.decoder2 = TransformerDecoder(num_decoder_layers2, d_model, num_heads, dropout)

    def forward(self, src1, src_mask1, src2, src_mask2, decoding, enc_out=None, gauss_weight=None, need_weight=False):
        assert decoding in [1, 2]
        if decoding == 1:
            if enc_out is None:
                enc_out, _ = self.decoder2(None, None, src2, src_mask2)
            out, weight = self.decoder1(enc_out, src_mask2, src1, src_mask1)
        elif decoding == 2:
            if enc_out is None:
                enc_out, _ = self.decoder1(None, None, src1, src_mask1, tgt_gauss_weight=gauss_weight)
                # enc_out = self.decoder1(None, None, src1, src_mask1)
            out, weight = self.decoder2(enc_out, src_mask1, src2, src_mask2, src_gauss_weight=gauss_weight)
        
        if need_weight:
            return enc_out, out, weight
        return enc_out, out
