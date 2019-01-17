import os
import pickle
import sys
import numpy as np
import torch
from tqdm import tqdm
from textclf.io.NewsDataset import NewsDatasetIterator, tag2idx
from sklearn.metrics import accuracy_score, f1_score, \
    precision_score, recall_score


def model_infer(model_path, inp_path):
    model_file = torch.load(model_path)
    train_opt = model_file["train_opt"]
    vocabs = model_file["vocabs"]

    meta_opt = train_opt.meta
    meta_opt.use_copy = True
    test_iter = NewsDatasetIterator(
        file_path=inp_path, vocabs=vocabs,
        epochs=meta_opt.epochs, batch_size=1,
        is_train=False, n_workers=meta_opt.n_workers,
        use_cuda=meta_opt.use_cuda, opt=meta_opt)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(meta_opt.gpu)

    model = model_file["model"]
    model.eval()
    model.flatten_parameters()

    y_pred = []
    total_loss = 0
    total_word_num = 0
    for batch in tqdm(test_iter):
        result_dict = model.run_batch(batch)
        total_loss += result_dict["loss"].item()
        total_word_num += result_dict["num_words"]
        preds = model.predict_batch(batch)
        y_pred.extend(preds)
    y_true = [tag2idx[t] for t in test_iter.dataset.raw_tags]
    print("Accuracy: ", accuracy_score(y_true, y_pred))
    print("F1: ", f1_score(y_true, y_pred, average="macro"))
    print("Precision: ", precision_score(y_true, y_pred, average="macro"))
    print("Recall: ", recall_score(y_true, y_pred, average="macro"))


if __name__ == "__main__":
    model_infer(sys.argv[1], sys.argv[2])