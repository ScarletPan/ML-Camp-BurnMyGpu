import torch.nn as nn
from textsum.models.seq2seq import AttnEncoderDecoder
from textsum.models.copyGenerator import CopyEncoderDecoder

def construct_model(opt, pre_word_emb=None):
    loss_fn = nn.NLLLoss(reduction="sum")
    if opt.meta.model == "seq2seq":
        model = AttnEncoderDecoder(loss_fn=loss_fn,
                                   opt=opt)
    elif opt.meta.model == "copynet":
        model = CopyEncoderDecoder(loss_fn=loss_fn,
                                   opt=opt)
    else:
        raise NotImplementedError
    if pre_word_emb is not None and opt.meta.use_pre_word_emb:
        print("load pretrained embeddings...")
        model.load_pretrained_embedding(pre_word_emb)
    return model