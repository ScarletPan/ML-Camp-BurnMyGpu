import os
import pickle
import sys
import time
import numpy as np
import torch
from tqdm import tqdm
from textsum.io.NewsDataset import NewsDatasetIterator
from textsum.toolbox.vocab import BOS_WORD, EOS_WORD, UNK_WORD


def model_infer(model_path, inp_path, outp_path):
    model_file = torch.load(model_path)
    train_opt = model_file["train_opt"]
    vocabs = model_file["vocabs"]

    meta_opt = train_opt.meta
    # meta_opt.use_copy = True
    test_iter = NewsDatasetIterator(
        file_path=inp_path, vocabs=vocabs,
        epochs=meta_opt.epochs, batch_size=1,
        is_train=False, n_workers=meta_opt.n_workers,
        use_cuda=meta_opt.use_cuda, opt=meta_opt)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(meta_opt.gpu)

    model = model_file["model"]
    model.eval()
    model.flatten_parameters()

    pred_list = []
    attns_list = []
    pgns_list = []
    total_loss = 0
    total_word_num = 0
    for batch in tqdm(test_iter):
        print(batch.enc_inps[0].shape)
        print(batch.dec_start_inps.shape)
        print(batch.ext_enc_inps[0].shape)
        st = time.time()
        result_dict = model.run_batch(batch)
        print("Forward in {:.2f} s".format(time.time() - st))
        total_loss += result_dict["loss"].item()
        total_word_num += result_dict["num_words"]
        st = time.time()
        preds_idx, attns, pgns, _ = model.predict_batch(
            batch, max_len=20, beam_size=5, eos_val=vocabs["word"].to_idx(EOS_WORD))
        preds = []
        print("Decode in {:.2f} s".format(time.time() - st))
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
            preds.append(pred)
        pred_list.extend(preds)
        if attns is not None:
            attns_list.extend(attns)
        if pgns is not None:
            pgns_list.extend(pgns)

    per_word_loss = total_loss / total_word_num
    s1 = "Valid, Loss: {:.2f}, PPL: {:.2f}".format(np.log(model_file["score"]), model_file["score"])
    s2 = "Test, Loss: {:.2f}, PPL: {:.2f}".format(per_word_loss, np.exp(per_word_loss))
    print(s1)
    print(s2)
    with open(outp_path, "w") as f:
        for i, pred in enumerate(pred_list):
            # print("[SRC]")
            # print("\t" + test_iter.dataset.raw_contents[i])
            # print("[TGT]")
            # print("\t" + test_iter.dataset.raw_headlines[i])
            # print("[PRED]")
            # print("\t" + " ".join(pred) + "\n")
            if BOS_WORD in pred:
                pred.remove(BOS_WORD)
            if EOS_WORD in pred:
                pred.remove(EOS_WORD)
            f.write(" ".join(pred) + "\n")
    with open(outp_path + ".debug.pkl", "wb") as f:
        pickle.dump({
            "inps": test_iter.dataset.raw_contents,
            "tgts": test_iter.dataset.raw_headlines,
            "preds": pred_list,
            "attns": attns_list,
            "pgns": pgns_list
        }, f)
    with open(outp_path + ".ppl.txt", "w") as f:
        f.write(s1 + "\n")
        f.write(s2 + "\n")


if __name__ == "__main__":
    model_infer(sys.argv[1], sys.argv[2], sys.argv[3])