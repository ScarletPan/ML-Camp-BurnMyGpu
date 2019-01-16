from clickbait.textsum.textsum.toolbox.utils import padding_list
from clickbait.textsum.textsum.toolbox.vocab import Vocabulary, BOS_WORD, EOS_WORD, UNK_WORD
import torch


tag2idx = {
    '详细全文': 0,
    '国内': 1,
    '国际': 2
}

idx2tag = {val: key for key, val in tag2idx.items()}


class SumBatch(object):
    def __init__(self, data, opt):
        self.opt = opt
        self.enc_inps = None
        self.ext_enc_inps = None
        self.dec_start_inps = None
        self.max_ext_vocab_size = None
        self.oov_vocabs = None

        self.build_batch(data)

    def build_batch(self, data):
        pad_idx = self.opt.pad_idx
        bos_idx = self.opt.bos_idx
        eos_idx = self.opt.eos_idx

        contents_lens = [len(t[0]) + 1 for t in data]
        enc_inps = [padding_list(t[0] + [eos_idx], self.opt.max_content_length + 1, pad_idx) for t in data]
        self.enc_inps = (torch.LongTensor(enc_inps),
                         torch.LongTensor(contents_lens))

        self.dec_start_inps = torch.LongTensor([[[bos_idx]] for _ in range(len(data))])

        ext_enc_inps = [padding_list(t[1] + [eos_idx], self.opt.max_content_length + 1, pad_idx) for t in data]
        self.ext_enc_inps = (torch.LongTensor(ext_enc_inps),
                             torch.LongTensor(contents_lens))

        self.max_ext_vocab_size = max([t[2].size for t in data])
        self.oov_vocabs = [t[2] for t in data]

    def cuda(self):
        self.enc_inps = [t.cuda(async=True) for t in self.enc_inps]
        self.dec_start_inps = self.dec_start_inps.cuda()
        self.ext_enc_inps = [t.cuda(async=True) for t in self.ext_enc_inps]


class ClfBatch(object):
    def __init__(self, data, opt):
        self.opt = opt
        self.inps = None
        self.bert_inps = None

        self.build_batch(data)

    def build_batch(self, data):
        pad_idx = self.opt.pad_idx
        data = list(sorted(data, key=lambda x: len(x[0]), reverse=True))

        contents_lens = [len(t[0]) for t in data]
        max_content_len = max(contents_lens)
        inps = [padding_list(t[0], max_content_len, pad_idx) for t in data]
        self.inps = (torch.LongTensor(inps),
                     torch.LongTensor(contents_lens))

        bert_contents_lens = [len(t[1]) for t in data]
        max_bert_content_lens = max(contents_lens)
        bert_inps = [padding_list(t[1], max_bert_content_lens, pad_idx) for t in data]
        self.bert_inps = (torch.LongTensor(bert_inps),
                          torch.LongTensor(bert_contents_lens))

    def cuda(self):
        self.inps = [t.cuda(async=True) for t in self.inps]
        self.bert_inps = [t.cuda(async=True) for t in self.bert_inps]


def make_sum_batch(text_list, vocab, opt):
    contents = []
    ext_contents = []
    extended_vocabs = []
    for content in text_list:
        content_words = content.split()[:opt.max_content_length]
        content_idx = [vocab.to_idx(t) for t in content_words]
        contents.append(content_idx)
        ext_vocab = Vocabulary(special_tokens=[])
        for token in content_words:
            if not vocab.has(token):
                ext_vocab.add_word(token)
        extended_vocabs.append(ext_vocab)

        ext_content_idx = []
        for token in content_words:
            if vocab.has(token):
                ext_content_idx.append(vocab.to_idx(token))
            else:
                ext_content_idx.append(vocab.size + ext_vocab.to_idx(token))
        ext_contents.append(ext_content_idx)

    data = [(contents[i], ext_contents[i], extended_vocabs[i])
            for i in range(len(contents))]
    batch = SumBatch(data, opt)
    return batch


def make_clf_batch(text_list, vocab, opt):
    contents = []
    bert_contents = []
    for content in text_list:
        content_words = content.split()[:opt.max_content_length]
        content_idx = [vocab.to_idx(t) for t in content_words]
        contents.append(content_idx)
        bert_contents.append(content_idx)
    data = [(contents[i], bert_contents[i])
            for i in range(len(contents))]
    batch = ClfBatch(data, opt)
    return batch


def load_model(model_path):
    model_file = torch.load(model_path)
    model_file["model"].flatten_parameters()
    model_file["model"].eval()
    return model_file


def text_summarize(text_list, model_file):
    """

    :param text_list: list of string (space split)
    :param model_path:
    :return:
    """
    train_opt = model_file["train_opt"]
    vocabs = model_file["vocabs"]
    model = model_file["model"]
    # print(model)

    batch = make_sum_batch(text_list, vocabs["word"], train_opt.meta)
    batch.cuda()
    preds_idx, attns, pgns = model.predict_batch(batch, max_len=20, beam_size=5,
                                                 eos_val=vocabs["word"].to_idx(EOS_WORD))
    res_preds = []
    res_attns = []
    res_pgns = []
    for k, pred_idx in enumerate(preds_idx):
        oov_vocab = batch.oov_vocabs[k]
        pred = []
        for t in pred_idx:
            if t < vocabs["word"].size:
                pred.append(vocabs["word"].to_word(t))
            elif t < vocabs["word"].size + oov_vocab.size:
                pred.append(oov_vocab.to_word(t - vocabs["word"].size))
            else:
                pred.append(UNK_WORD)
        res_pred = []
        res_attn = []
        res_pgn = []
        for i in range(len(pred)):
            if pred[i] == BOS_WORD or pred[i] == EOS_WORD:
                continue
            else:
                res_pred.append(pred[i])
                res_attn.append(attns[k][i])
                res_pgn.append(pgns[k][i])
        res_preds.append(res_pred)
        res_attns.append(res_attn)
        res_pgns.append(res_pgn)

    return {
        "preds": res_preds,
        "attns": res_attns,
        "pgns": res_pgns,
        "max_seq_len": train_opt.meta.max_content_length
    }


def text_tag_classification(text_list, model_file):
    train_opt = model_file["train_opt"]
    vocabs = model_file["vocabs"]
    model = model_file["model"]
    # print(model)

    batch = make_clf_batch(text_list, vocabs["word"], train_opt.meta)
    batch.cuda()
    preds = model.predict_batch(batch).data.cpu().tolist()
    preds = [idx2tag[i] for i in preds]
    return preds