# encoding: utf-8
import csv
import os
import torch
import torch.utils.data
from textclf.toolbox.utils import padding_list, chunks
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


tag2idx = {
    '详细全文': 0,
    '国内': 1,
    '国际': 2
}

idx2tag = {val: key for key, val in tag2idx.items()}


def all_len_eq(_list, x):
    for item in _list:
        if len(item) != x:
            return False
    return True


class Batch(object):
    def __init__(self, data, opt):
        self.opt = opt
        self.inps = None
        self.bert_inps = None
        self.labels = None

        self.build_batch(data)

    def build_batch(self, data):
        pad_idx = self.opt.pad_idx
        data = list(sorted(data, key=lambda x: len(x[0]), reverse=True))

        contents_lens = [len(t[0]) for t in data]
        max_content_len = max(contents_lens)
        inps = [padding_list(t[0], max_content_len, pad_idx) for t in data]
        self.inps = (torch.LongTensor(inps),
                     torch.LongTensor(contents_lens))
        assert all_len_eq(inps, max_content_len), print(data)

        bert_contents_lens = [len(t[1]) for t in data]
        max_bert_content_lens = max(contents_lens)
        bert_inps = [padding_list(t[1], max_bert_content_lens, pad_idx) for t in data]
        self.bert_inps = (torch.LongTensor(bert_inps),
                          torch.LongTensor(bert_contents_lens))
        assert all_len_eq(bert_inps, max_bert_content_lens), print(data)

        labels = [t[2] for t in data]
        self.labels = torch.LongTensor(labels)

    def cuda(self):
        self.inps = [t.cuda(async=True) for t in self.inps]
        self.bert_inps = [t.cuda(async=True) for t in self.bert_inps]
        self.labels = self.labels.cuda(async=True)


class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, vocabs, opt, data_cache_path=None):
        self.opt = opt
        self.raw_tags = []
        self.raw_contents = []
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.opt.bert_pad_idx = self.bert_tokenizer.vocab[""]
        raw_lines = []
        with open(data_path) as f:
            f.readline()
            reader = csv.reader(f, delimiter=',',
                                quotechar='"')
            for row in reader:
                _, tag, _, content = row
                raw_lines.append([tag, content])
                self.raw_tags.append(tag)
                self.raw_contents.append(content)
        if data_cache_path and os.path.exists(data_cache_path):
            self.data = torch.load(data_cache_path)
        else:
            res_list = []
            for lines in chunks(raw_lines, len(raw_lines) // 20):
                res = self.load_data(lines, vocabs)
                res_list.append(res)

            contents = []
            bert_contents = []
            tags = []
            for records in tqdm(res_list):
                contents.extend(records[0])
                bert_contents.extend(records[1])
                tags.extend(records[2])
            self.data = [(contents[i], bert_contents[i], tags[i])
                         for i in range(len(contents))]

            if data_cache_path:
                torch.save(self.data, data_cache_path)

    def load_data(self, lines, vocabs):
        contents = []
        bert_contents = []
        tags = []
        for tag, content in lines:
            content_words = content.split()[:self.opt.max_content_length]
            content_idx = [vocabs["word"].to_idx(t) for t in content_words]
            contents.append(content_idx)
            tokenized_text = self.bert_tokenizer.tokenize("".join(content_words))[:self.opt.max_content_length]
            indexed_tokens = self.bert_tokenizer.convert_tokens_to_ids(tokenized_text)
            bert_contents.append([self.bert_tokenizer.vocab["[CLS]"]]
                                 + indexed_tokens
                                 + [self.bert_tokenizer.vocab["[SEP]"]])
            tags.append(tag2idx[tag])
            # print("=" * 50)
            # print(content_words)
            # print(content_idx)
            # print(tag)

        return contents, bert_contents, tags

    def __getitem__(self, index):
        """

        :param index: int
        :return:
        """
        return self.data[index], self.opt

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    batch = Batch([t[0] for t in data], data[0][-1])
    return batch


class NewsDatasetIterator(object):
    def __init__(self, file_path, vocabs, file_cache_path=None, epochs=None, batch_size=16,
                 is_train=True, n_workers=0,
                 use_cuda=True, opt=None):
        self.dataset = NewsDataset(file_path, vocabs, opt, file_cache_path)
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.is_train = is_train
        self.use_cuda = use_cuda

    def __iter__(self):
        if self.is_train:
            i = 0
            while True:
                data_loader = torch.utils.data.DataLoader(
                    dataset=self.dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    collate_fn=collate_fn,
                    num_workers=self.n_workers)
                for batch in data_loader:
                    if self.use_cuda:
                        batch.cuda()
                    yield batch
                i += 1
                if self.epochs and i == self.epochs:
                    break
        else:
            data_loader = torch.utils.data.DataLoader(
                dataset=self.dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=self.n_workers)
            for batch in data_loader:
                if self.use_cuda:
                    batch.cuda()
                yield batch
