# -*- encoding: utf-8 -*-
import argparse
import os
import sys
import time
import random
import subprocess
import shutil
import numpy as np
import pyrouge


def read_and_merge(cand_fn, ref_fn, out):
    cand = []
    ref = []
    with open(cand_fn, 'r') as f:
        for line in f: cand.append(''.join(line.strip().split()))

    with open(ref_fn, 'r') as f:
        for line in f: ref.append(''.join(line.strip().split()))

    assert len(cand) == len(ref), (len(cand), len(ref))
    with open(out, 'w') as f:
        for a, b in zip(cand, ref): f.write('|||'.join([a, b]) + '\n')


def read_and_test(out):
    cand = []
    ref = []
    with open(out, 'r') as f:
        for line in f:
            a, b = line.strip().split('|||')
            cand.append(a)
            ref.append(b)

    TP = 0
    FP = 0
    FN = 0
    for c, r in zip(cand, ref):
        aspect_c = list(set([ item.split('-')[0] for item in c.split(',')]))
        aspect_r = list(set([item.split('-')[0] for item in r.split(',')]))
        # print(aspect_c, aspect_r)
        for item in aspect_c:
            if item in aspect_r:
                TP += 1
            else:
                FP += 1
        for item in aspect_r:
            if item not in aspect_c:
                FN += 1
    # print(TP, FP, FN)
    P = TP / (TP + FP)
    R = TP / (TP + FN)

    return P, R, 2 * P * R / (P + R) if (P+R) != 0 else 0.0


def turn2idx(candidates, references):
    cand_tokens = [line.split(" ") for line in candidates]
    ref_tokens = [line.split(" ") for line in references]
    vocab = {}
    cnt = 0
    cand_idxes = []
    for tokens in cand_tokens:
        tmp_idxes = []
        for token in tokens:
            if token not in vocab:
                vocab[token] = cnt
                cnt += 1
            tmp_idxes.append(vocab[token])
        cand_idxes.append(tmp_idxes)
    ref_idxes = []
    for tokens in ref_tokens:
        tmp_idxes = []
        for token in tokens:
            if token not in vocab:
                vocab[token] = cnt
                cnt += 1
            tmp_idxes.append(vocab[token])
        ref_idxes.append(tmp_idxes)
    candidates = [" ".join([str(idx) for idx in idxes]) 
                 for idxes in cand_idxes]
    references = [" ".join([str(idx) for idx in idxes]) 
                 for idxes in ref_idxes]
    return candidates, references


def test_aspect(cand, ref):
    out = "merge.txt"
    dw = "aspect.txt"
    read_and_merge(cand, ref, out)
    subprocess.check_output("sh aspect_eval.sh %s %s" % (out, dw), shell=True)
    p, r, f1 = read_and_test(dw)
    print(">> Aspect Precision/Recall/F1: {:.4f}/{:.4f}/{:.4f}".format(
              p, r, f1))


def test_bleu(cand_file, ref_file, idx=False):
    if os.path.exists("multi-bleu.perl"):
        bleu_path = "multi-bleu.perl"
    else:
        bleu_path = "scripts/multi-bleu.perl"
    if idx:
        with open(cand_file, encoding="utf-8") as f_cand:
            candidates = [line.strip() for line in f_cand]
        with open(ref_file, encoding="utf-8") as f_ref:
            references = [line.strip() for line in f_ref]
        candidates, references = turn2idx(candidates, references)
        with open("cand-idx.txt", "w") as f:
            for cand in candidates:
                f.write(cand + "\n")
        with open("ref-idx.txt", "w") as f:
            for ref in references:
                f.write(ref + "\n")
        res = subprocess.check_output("perl {} ref-idx.txt < cand-idx.txt".format(bleu_path),
                                      shell=True)
        if os.path.exists("cand-idx.txt"):
            os.remove("cand-idx.txt")
        if os.path.exists("ref-idx.txt"):
            os.remove("ref-idx.txt")
    else:
        res = subprocess.check_output("perl {} {} < {}".format(
                                  bleu_path, ref_file, cand_file), shell=True)
    print(">> " + res.decode("utf-8").strip())


def test_rouge(cand_file, ref_file, idx=False):
    f_cand = open(cand_file, encoding="utf-8")
    f_ref = open(ref_file, encoding="utf-8")
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = ".rouge-tmp-{}".format(current_time)
    try:
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
            os.mkdir(tmp_dir + "/candidate")
            os.mkdir(tmp_dir + "/reference")
        candidates = [line.strip() for line in f_cand]
        references = [line.strip() for line in f_ref]
        if idx:
            candidates, references = turn2idx(candidates, references)
        assert len(candidates) == len(references)
        cnt = len(candidates)
        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        f_cand.close()
        f_ref.close()
        r = pyrouge.Rouge155()
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = 'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        results_dict = r.output_to_dict(rouge_results)
        print(">> ROUGE(1/2/3/L/SU4): {:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}".format(
              results_dict["rouge_1_f_score"] * 100,
              results_dict["rouge_2_f_score"] * 100,
              results_dict["rouge_3_f_score"] * 100,
              results_dict["rouge_l_f_score"] * 100,
              results_dict["rouge_su*_f_score"] * 100))
    finally:
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)

def test_sent_rouge(cand, ref, idx=False, method="rouge_l_f_score"):
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = ".rouge-tmp-{}".format(current_time)
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    with open(tmp_dir + "/candidate/cand.0.txt", "w",
              encoding="utf-8") as f:
        f.write(cand)
    with open(tmp_dir + "/reference/ref.0.txt", "w",
              encoding="utf-8") as f:
        f.write(ref)
    r = pyrouge.Rouge155()
    r.model_dir = tmp_dir + "/reference/"
    r.system_dir = tmp_dir + "/candidate/"
    r.model_filename_pattern = 'ref.#ID#.txt'
    r.system_filename_pattern = 'cand.(\d+).txt'
    rouge_results = r.convert_and_evaluate()
    results_dict = r.output_to_dict(rouge_results)
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    return results_dict[method]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, default="candidate.txt",
                        help='Candidate file name')
    parser.add_argument('-r', type=str, default="reference.txt",
                        help='Reference file name')
    parser.add_argument('-i', default=False, action='store_true',
                        help='Turn tokens into idx')
    parser.add_argument('--report-bleu', default=False, action='store_true',
                        help='Report Bleu Score')
    parser.add_argument('--report-rouge', default=False, action='store_true',
                        help='Report Rouge Score')
    parser.add_argument('--accuracy-k', type=int, default="3",
                        help='Hit set')
    args = parser.parse_args()
    if args.report_bleu:
        test_bleu(args.c, args.r, args.i)
    if args.report_rouge:
        test_rouge(args.c, args.r, args.i)