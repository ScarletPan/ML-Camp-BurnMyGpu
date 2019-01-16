import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from textclf.models.base import BaseDeepModel
from pytorch_pretrained_bert import BertModel, BertForSequenceClassification


class Bert(BaseDeepModel):
    def __init__(self, loss_fn, opt):
        super(Bert, self).__init__()
        rnn_hidden_size = opt.model.rnn_hidden_size
        word_embed_size = opt.model.word_embed_size

        self.embedding = nn.Embedding(opt.model.word_vocab_size, word_embed_size)
        self.encoder = nn.LSTM(input_size=word_embed_size,
                               hidden_size=rnn_hidden_size,
                               num_layers=opt.model.n_layers,
                               bias=True,
                               batch_first=True,
                               bidirectional=True)
        self.bert = BertForSequenceClassification.from_pretrained(
            'bert-base-chinese', num_label=opt.meta.n_class)

    def flatten_parameters(self):
        pass

    def forward(self, inps, inps_len,
                      bert_inps, bert_inps_len):
        return self.bert()

    def run_batch(self, batch):
        inps, inps_lens = batch.inps
        bert_inps, bert_inps_lens = batch.bert_inps
        labels = batch.labels
        logits = self.forward(
            inps=inps, inps_len=inps_lens,
            bert_inps=bert_inps, bert_inps_len=bert_inps_lens)

        loss = self.loss_fn(logits, labels)
        _, pred = logits.max(1)
        num_correct = pred.eq(labels).sum().item()
        num_words = pred.size(0)
        result_dict = {
            "loss": loss,
            "num_correct": num_correct,
            "num_words": num_words,
            "pred": pred.data.cpu().numpy().tolist()
        }
        return result_dict