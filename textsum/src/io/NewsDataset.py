from multiprocessing import Pool
import os
import torch
import torch.utils.data
from tqdm import tqdm
import json
from textsum.src.toolbox.utils import padding_list, chunks


def all_len_eq(_list, x):
    for item in _list:
        if len(item) != x:
            return False
    return True


class Batch(object):
    def __init__(self, data, opt):
        self.opt = opt
        self.enc_inps = None
        self.dec_inps = None
        self.dec_start_inps = None
        self.dec_tgts = None

        self.build_batch(data)

    def build_batch(self, data):
        pad_idx = self.opt.pad_idx
        bos_idx = self.opt.bos_idx
        eos_idx = self.opt.eos_idx
        posts_lens = [len(t[0]) + 1 for t in data]
        enc_inps = [padding_list(t[0] + [eos_idx], self.opt.max_content_length + 1, pad_idx) for t in data]
        self.enc_inps = (torch.LongTensor(enc_inps),
                         torch.LongTensor(posts_lens))
        assert all_len_eq(enc_inps, self.opt.max_content_length + 1), print(data)

        response_lens = [len(t[1]) + 1 for t in data]
        dec_inps = [padding_list([bos_idx] + t[1], self.opt.max_headline_length + 1, pad_idx) for t in data]
        dec_tgts = [padding_list(t[1] + [eos_idx], self.opt.max_headline_length + 1, pad_idx) for t in data]
        assert all_len_eq(dec_inps, self.opt.max_headline_length + 1), print(data)
        assert all_len_eq(dec_tgts, self.opt.max_headline_length + 1), print(data)
        self.dec_inps = (torch.LongTensor(dec_inps),
                         torch.LongTensor(response_lens))
        self.dec_start_inps = torch.LongTensor([[[bos_idx]] for _ in range(len(data))])
        self.dec_tgts = (torch.LongTensor(dec_tgts),
                         torch.LongTensor(response_lens))

    def cuda(self):
        self.enc_inps = [t.cuda(async=True) for t in self.enc_inps]
        self.dec_inps = [t.cuda(async=True) for t in self.dec_inps]
        self.dec_start_inps = self.dec_start_inps.cuda()
        self.dec_tgts = [t.cuda(async=True) for t in self.dec_tgts]


class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, vocabs, opt, data_cache_path=None):
        self.opt = opt
        with open(data_path) as f:
            raw_lines = f.readlines()

        if data_cache_path and os.path.exists(data_cache_path):
            self.data = torch.load(data_cache_path)
        else:
            if len(raw_lines) > 1000:
                res_list = []
                pool = Pool(20)
                for lines in chunks(raw_lines, len(raw_lines) // 20):
                    res = pool.apply_async(self.load_data, args=(lines, vocabs,))
                    res_list.append(res)
                pool.close()
                pool.join()

                contents = []
                headlines = []
                for res in res_list:
                    records = res.get()
                    contents.extend(records[0])
                    headlines.extend(records[1])
                self.data = [(contents[i], headlines[i])
                             for i in range(len(contents))]
            else:
                t = self.load_data(raw_lines, vocabs)
                self.data = [(t[0][i], t[1][i])
                             for i in range(len(t[0]))]
            if data_cache_path:
                torch.save(self.data, data_cache_path)

    def load_data(self, lines, vocabs):
        contents = []
        headlines = []
        for line in lines:
            *_, headline, content = line.split(",")
            post_idx = [vocabs["word"].to_idx(t) for t in content.split()[:self.opt.max_content_length]]
            contents.append(post_idx)
            response_idx = [vocabs["word"].to_idx(t) for t in headline.lower().split()[:self.opt.max_headline_length]]
            headlines.append(response_idx)

        return contents, headlines

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
