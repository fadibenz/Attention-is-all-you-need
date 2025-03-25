from torch import nn
import einops

from architectures.scaled_dot_product import scaled_dot_product_attention

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.n_heads = n_heads
        self.d_head = d_model // n_heads

    def forward(self, q, k, v, key_padding_mask=None, causal=False):

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = einops.rearrange(q, "n l (d h) -> (n d) l  h", h = self.d_head)
        k = einops.rearrange(k, "n l (d h) -> (n d) l  h", h = self.d_head)
        v = einops.rearrange(v, "n l (d h) -> (n d) l  h", h = self.d_head)

        if key_padding_mask is not None:
          key_padding_mask = key_padding_mask.unsqueeze(-1)
          key_padding_mask = key_padding_mask.expand(-1, -1, self.n_heads)
          key_padding_mask = einops.rearrange(key_padding_mask, "n l d -> (n d) l" )

        attention = scaled_dot_product_attention(q, k, v, key_padding_mask, causal)
        attention = einops.rearrange(attention, "(n d) l  h -> n l (d h)", d = self.n_heads )
        o_projection = self.o_proj(attention)
        return o_projection