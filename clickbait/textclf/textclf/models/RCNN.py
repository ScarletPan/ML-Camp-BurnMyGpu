import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from textclf.models.base import BaseDeepModel
from pytorch_pretrained_bert import BertModel

bert_model = BertModel.from_pretrained('bert-base-chinese')


class RCNN(BaseDeepModel):
    def __init__(self, loss_fn, opt):
        super(RCNN, self).__init__()
        rnn_hidden_size = opt.model.rnn_hidden_size
        word_embed_size = opt.model.word_embed_size

        self.embedding = nn.Embedding(opt.model.word_vocab_size, word_embed_size)
        self.encoder = nn.LSTM(input_size=word_embed_size,
                               hidden_size=rnn_hidden_size,
                               num_layers=opt.model.n_layers,
                               bias=True,
                               batch_first=True,
                               bidirectional=True)
        self.bert = None

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=rnn_hidden_size * 2 + word_embed_size,
                      out_channels=opt.model.kernel_dim,
                      kernel_size=opt.model.kernel_size),
            nn.BatchNorm1d(opt.model.kernel_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=opt.model.kernel_dim,
                      out_channels=opt.model.kernel_dim,
                      kernel_size=opt.model.kernel_size),
            nn.BatchNorm1d(opt.model.kernel_dim),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(opt.model.kernel_dim, opt.model.linear_hidden_size),
            nn.BatchNorm1d(opt.model.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(opt.model.linear_hidden_size, opt.meta.n_class)
        )
        self.n_class = opt.meta.n_class
        self.n_directions = 2
        self.n_layers = opt.model.n_layers
        self.hidden_size = opt.model.rnn_hidden_size
        self.use_cuda = opt.meta.use_cuda
        self.loss_fn = loss_fn

    def flatten_parameters(self):
        self.encoder.flatten_parameters()

    def load_pretrained_embedding(self, pretrained_embedding_matrix):
        tmp = torch.FloatTensor(pretrained_embedding_matrix)
        if self.use_cuda:
            tmp = tmp.cuda()
        self.embedding.weight.data.copy_(tmp)

    @staticmethod
    def kmax_pooling(x, dim, k):
        index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)

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
        fact_embeds = self.embedding(inps)
        packed_embeds = pack_padded_sequence(fact_embeds,
                                             inps_len.cpu().numpy(),
                                             batch_first=True)
        init_hidden = self.init_hidden(bsize)
        packed_encoder_output, _ = self.encoder(
            packed_embeds, init_hidden)
        encoder_output, _ = pad_packed_sequence(packed_encoder_output)
        encoder_output = encoder_output.transpose(0, 1)

        out = torch.cat([encoder_output, fact_embeds], dim=2)
        conv_out = self.kmax_pooling(self.conv(
            out.transpose(1, 2)), 2, 1).squeeze(2)

        logits = self.fc(conv_out)
        return logits


class BertRCNN(BaseDeepModel):
    def __init__(self, loss_fn, opt):
        super(BertRCNN, self).__init__()
        rnn_hidden_size = opt.model.rnn_hidden_size
        word_embed_size = opt.model.word_embed_size

        self.embedding = nn.Embedding(opt.model.word_vocab_size, word_embed_size)
        self.encoder = nn.LSTM(input_size=word_embed_size,
                               hidden_size=rnn_hidden_size,
                               num_layers=opt.model.n_layers,
                               bias=True,
                               batch_first=True,
                               bidirectional=True)

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=rnn_hidden_size * 2 + word_embed_size,
                      out_channels=opt.model.kernel_dim,
                      kernel_size=opt.model.kernel_size),
            nn.BatchNorm1d(opt.model.kernel_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=opt.model.kernel_dim,
                      out_channels=opt.model.kernel_dim,
                      kernel_size=opt.model.kernel_size),
            nn.BatchNorm1d(opt.model.kernel_dim),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(opt.model.kernel_dim + opt.model.bert_dim*3,
                      opt.model.linear_hidden_size),
            nn.BatchNorm1d(opt.model.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(opt.model.linear_hidden_size, opt.meta.n_class)
        )
        self.n_class = opt.meta.n_class
        self.n_directions = 2
        self.n_layers = opt.model.n_layers
        self.hidden_size = opt.model.rnn_hidden_size
        self.use_cuda = opt.meta.use_cuda
        self.loss_fn = loss_fn

    def flatten_parameters(self):
        self.encoder.flatten_parameters()

    def load_pretrained_embedding(self, pretrained_embedding_matrix):
        tmp = torch.FloatTensor(pretrained_embedding_matrix)
        if self.use_cuda:
            tmp = tmp.cuda()
        self.embedding.weight.data.copy_(tmp)

    @staticmethod
    def kmax_pooling(x, dim, k):
        index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)

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
        fact_embeds = self.embedding(inps)
        packed_embeds = pack_padded_sequence(fact_embeds,
                                             inps_len.cpu().numpy(),
                                             batch_first=True)
        init_hidden = self.init_hidden(bsize)
        packed_encoder_output, _ = self.encoder(
            packed_embeds, init_hidden)
        encoder_output, _ = pad_packed_sequence(packed_encoder_output)
        encoder_output = encoder_output.transpose(0, 1)

        out = torch.cat([encoder_output, fact_embeds], dim=2)
        conv_out = self.kmax_pooling(self.conv(
            out.transpose(1, 2)), 2, 1).squeeze(2)

        global bert_model
        if self.use_cuda:
            bert_model = bert_model.cuda()
        bert_model.eval()
        bert_outputs, _ = bert_model(bert_inps)
        bert_out = torch.cat((bert_outputs[-3][:, 0],
                              bert_outputs[-2][:, 0],
                              bert_outputs[-1][:, 0]), dim=1)

        logits = self.fc(torch.cat((conv_out, bert_out), dim=1))
        return logits