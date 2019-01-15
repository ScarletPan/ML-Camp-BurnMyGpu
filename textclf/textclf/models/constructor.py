import torch.nn as nn
from textclf.models.RCNN import RCNN, BertRCNN
from textclf.models.poolLSTM import PoolLSTM


def construct_model(opt, pre_word_emb=None):
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    if opt.meta.model == "poolLSTM":
        model = PoolLSTM(loss_fn, opt)
    elif opt.meta.model == "rcnn":
        model = RCNN(loss_fn, opt)
    elif opt.meta.model == "bert_rcnn":
        model = BertRCNN(loss_fn, opt)
    else:
        raise NotImplementedError
    if pre_word_emb is not None and opt.meta.use_pre_word_emb:
        print("load pretrained embeddings...")
        model.load_pretrained_embedding(pre_word_emb)
    return model