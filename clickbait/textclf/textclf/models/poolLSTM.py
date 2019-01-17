import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from textclf.models.base import BaseDeepModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PoolLSTM(BaseDeepModel):
    def __init__(self, loss_fn, opt):
        super(PoolLSTM, self).__init__()
        rnn_hidden_size = opt.model.rnn_hidden_size

        self.encoder_embedding = nn.Embedding(opt.model.word_vocab_size, opt.model.word_embed_size)
        self.encoder = nn.LSTM(input_size=opt.model.word_embed_size,
                               hidden_size=rnn_hidden_size,
                               num_layers=opt.model.n_layers,
                               bias=True,
                               batch_first=True,
                               bidirectional=True)
        self.fc = nn.Linear(rnn_hidden_size, 3)

        self.n_class = opt.meta.n_class
        self.n_directions = 2
        self.n_layers = opt.model.n_layers
        self.hidden_size = opt.model.rnn_hidden_size
        self.use_cuda = opt.meta.use_cuda
        self.loss_fn = loss_fn
        self.rnn_hidden_size = rnn_hidden_size
        self_linear_hidden_size = opt.model.self_linear_hidden_size
        linear_hidden_size = opt.model.linear_hidden_size
        self.self_fc = nn.Sequential(
            nn.Linear(rnn_hidden_size * 2, self_linear_hidden_size),
            nn.BatchNorm1d(self_linear_hidden_size),
            nn.Tanh(),
            nn.Linear(self_linear_hidden_size, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(rnn_hidden_size * 8, linear_hidden_size),
            nn.BatchNorm1d(linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(linear_hidden_size, self.n_class)
        )
        self.dropout = nn.Dropout(opt.model.dropout)

    def load_pretrained_embedding(self, pretrained_embedding_matrix):
        tmp = torch.FloatTensor(pretrained_embedding_matrix)
        if self.use_cuda:
            tmp = tmp.cuda()
        self.embedding.weight.data.copy_(tmp)

    def self_attn_encode(self, encoder_output):
        bsize, seq_len, _ = encoder_output.size()
        flatten_encoder_output = encoder_output.view(-1, self.rnn_hidden_size * 2)
        out = self.self_fc(flatten_encoder_output)
        out = out.view(bsize, seq_len, -1)
        self_attn_weights = F.softmax(out, dim=1).transpose(1, 2)
        encoded = self_attn_weights.bmm(encoder_output)
        return encoded

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.n_layers * self.n_directions,
                                      batch_size, self.hidden_size))
        hidden.data.normal_(std=0.01)
        cell = Variable(torch.zeros(self.n_layers * self.n_directions,
                                    batch_size, self.hidden_size))
        cell.data.normal_(std=0.01)
        if self.use_cuda:
            hidden = hidden.cuda()
            cell = cell.cuda()
        return (hidden, cell)

    def forward(self, inps, inps_len,
                      bert_inps, bert_inps_len):
        bsize = inps.size(0)
        fact_embeds = self.encoder_embedding(inps)
        packed_embeds = pack_padded_sequence(fact_embeds,
                                             inps_len.cpu().numpy(),
                                             batch_first=True)
        init_hidden = self.init_hidden(bsize)
        packed_encoder_output, (last_hidden, _) = self.encoder(packed_embeds, init_hidden)
        encoder_output, _ = pad_packed_sequence(packed_encoder_output)
        encoder_output = encoder_output.transpose(0, 1).contiguous()
        last_hidden = self._fix_hidden(last_hidden)

        last_hidden_out = last_hidden.transpose(0, 1).contiguous().view(bsize, -1)
        max_pool_out = self.kmax_pooling(encoder_output, 1, 1).squeeze(1)
        avg_pool_out = encoder_output.mean(dim=1)
        self_attn_out = self.self_attn_encode(encoder_output).squeeze(1)
        concat_out = torch.cat(
            [last_hidden_out, max_pool_out, avg_pool_out, self_attn_out],
            dim=1)
        logits = self.fc(concat_out).squeeze(1)
        return logits

    @staticmethod
    def kmax_pooling(x, dim, k):
        index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)

    @staticmethod
    def _fix_hidden(hidden):
        # The encoder hidden is  (layers*directions) x batch x dim.
        # We need to convert it to layers x batch x (directions*dim).
        hidden = torch.cat([hidden[0:hidden.size(0):2],
                            hidden[1:hidden.size(0):2]], 2)
        return hidden

    def flatten_parameters(self):
        self.encoder.flatten_parameters()