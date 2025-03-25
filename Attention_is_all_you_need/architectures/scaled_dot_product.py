import torch

def scaled_dot_product_attention(q, k, v, key_padding_mask=None, causal=False):
    from math import sqrt
    n, _, d = q.size()
    future_mask = torch.triu(torch.zeros([1024, 1024], device=q.device).fill_(float("-inf")), 1)
    similarities = torch.einsum("n i s, n j s -> n i j", q, k) / sqrt(d)
    if key_padding_mask is not None:
      similarities = similarities.masked_fill_(key_padding_mask.unsqueeze(1).bool(), float('-inf'))
    if causal:
      similarities = similarities + future_mask.expand(n, -1, -1)[:, :similarities.size(1), :similarities.size(2)]
    coefficients = torch.softmax(similarities, -1)
    attention =  torch.einsum("n i j, n j k -> n i k", coefficients, v)
    return attention
