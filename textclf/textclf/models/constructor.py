import torch.nn as nn
from textclf.models.poolLSTM import PoolLSTM


def construct_model(opt, pre_word_emb=None):
    loss_fn = nn.CrossEntropyLoss(ignore_index=opt.meta.pad_idx, reduction="sum")
    if opt.meta.model == "poolLSTM":
        model = PoolLSTM(loss_fn, opt)
    else:
        raise NotImplementedError
    if pre_word_emb is not None and opt.meta.use_pre_word_emb:
        print("load pretrained embeddings...")
        model.load_pretrained_embedding(pre_word_emb)
    return model