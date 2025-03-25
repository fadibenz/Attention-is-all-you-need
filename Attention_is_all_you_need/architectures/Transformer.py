import torch.nn as nn
from architectures.TransformerLayer import TransformerLayer
import torch
from transformers.modeling_outputs import Seq2SeqLMOutput


class Transformer(nn.Module):
    def __init__(self, n_words, pad_id, eos_id, max_len, n_layers, d_model, n_heads, d_ffn, p_drop):
        super().__init__()

        self.pad_id = pad_id  #To find padding in the sequence
        self.eos_id = eos_id

        self.emb_word = nn.Embedding(n_words, d_model) # Fixed-Size id -> embeddings,
                                                       # look-up table,
                                                       # equivalent to a linear Layer.
        self.emb_pos = nn.Embedding(max_len, d_model)

        self.emb_word.weight.data.uniform_(-0.05, 0.05)
        self.emb_pos.weight.data.uniform_(-0.05, 0.05)

        self.emb_ln = nn.LayerNorm(d_model)

        self.encoder_layers = nn.ModuleList([
            TransformerLayer(False, d_model, n_heads, d_ffn, p_drop)
            for _ in range(n_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerLayer(True, d_model, n_heads, d_ffn, p_drop)
            for _ in range(n_layers)
        ])

        self.lm_head = nn.Linear(d_model, n_words)

        self.lm_head.weight = self.emb_word.weight
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    @staticmethod
    def make_positions(input_ids, padding_mask):
          positions = torch.cumsum(1 - padding_mask.long(), 1) - 1
          return positions

    def forward(self, input_ids, attention_mask, labels):

        enc_padding_mask = input_ids.eq(self.pad_id).byte()
        enc_pos = self.make_positions(input_ids, enc_padding_mask)
        enc_state = self.emb_ln(self.emb_word(input_ids) + self.emb_pos(enc_pos))


        for layer in self.encoder_layers:
            enc_state = layer(enc_state, enc_padding_mask)

        decoder_input_ids = labels.new_zeros(labels.shape)
        decoder_input_ids[:, 1:] = labels[:, :-1].clone()
        decoder_input_ids[:, 0] = self.eos_id
        decoder_input_ids.masked_fill_(decoder_input_ids == -100, self.pad_id)

        dec_padding_mask = decoder_input_ids.eq(self.pad_id).byte()
        dec_pos = self.make_positions(decoder_input_ids, dec_padding_mask)
        dec_state = self.emb_ln(self.emb_word(decoder_input_ids) + self.emb_pos(dec_pos))

        for layer in self.decoder_layers:
            dec_state = layer(dec_state, dec_padding_mask, enc_state, enc_padding_mask)

        lm_logits = self.lm_head(dec_state)

        loss = self.criterion(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits
        )