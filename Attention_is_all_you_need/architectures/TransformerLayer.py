import torch.nn as nn
from MHA import MultiheadAttention
import torch.nn.functional as F

class TransformerLayer(nn.Module):
    def __init__(self, is_decoder, d_model, n_heads, d_ffn, p_drop):
        super().__init__()
        self.is_decoder = is_decoder
        self.self_attn = MultiheadAttention(d_model, n_heads)
        self.self_attn_drop = nn.Dropout(p_drop)
        self.self_attn_ln = nn.LayerNorm(d_model)
        if is_decoder:
            self.cross_attn = MultiheadAttention(d_model, n_heads)
            self.cross_attn_drop = nn.Dropout(p_drop)
            self.cross_attn_ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ffn)
        self.fc2 = nn.Linear(d_ffn, d_model)
        self.ffn_drop = nn.Dropout(p_drop)
        self.ffn_ln = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask, encoder_out=None, encoder_padding_mask=None):
      causal = self.is_decoder

      z = self.self_attn(x, x, x, key_padding_mask=padding_mask, causal=causal)
      z = self.self_attn_drop(z)
      z1 = self.self_attn_ln(z + x)

      if self.is_decoder and encoder_out is not None:
          zd = self.cross_attn(z1, encoder_out, encoder_out,
                            key_padding_mask=encoder_padding_mask, causal=False)
          zd = self.cross_attn_drop(zd)
          z1 = self.cross_attn_ln(zd + z1)

      z2 = self.fc2(F.relu(self.fc1(z1)))
      z2 = self.ffn_drop(z2)
      final = self.ffn_ln(z2 + z1)

      return final
