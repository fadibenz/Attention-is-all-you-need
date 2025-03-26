I followed a modular implementation approach, strating with scaled dot product attention, Multi-head attention, transformer layers and finally the full transformer with the encoder-decoder paradigm using cuasal and non-causal attention, you can find each under `Attention_is_all_you_need/architectures`.

I tested each part rigorously againt the already implemented pytorch versions, you can find the full tests under `notebooks`.


For the final test I trained my transformer on the x-sum dataset, you can find the results and the wandb dashboard under `notebooks/Training_final_test`.


My implementation is still inefficient for full scale training and evaluation since I have not implemented the KV caching yet, but I will add it this week.
