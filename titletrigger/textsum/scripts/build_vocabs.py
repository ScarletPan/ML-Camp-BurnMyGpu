from collections import Counter
import csv
from textsum.toolbox.vocab import Vocabulary, UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD, get_pretrained_embedding
import sys
import torch
from tqdm import tqdm


WORD_VOCAB_SIZE = 50000


def build_vocabs(counters):
    word_vocab = Vocabulary(special_tokens=[UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD])
    word_vocab.build_from_counter(counters["word"], WORD_VOCAB_SIZE)
    # pre_emb = get_pretrained_embedding(
    #     word_vocab.stoi,
    #     pretrained_w2v_path="/home/hjpan/projects/sActivityNet/dialogue/data/glove.6B.300d.txt")
    pre_emb = None

    vocabs = {
        "word": word_vocab,
        "pre_word_emb": pre_emb
    }
    return vocabs


if __name__ == "__main__":
    word_counter = Counter()

    with open(sys.argv[1]) as f:
        f.readline()
        reader = csv.reader(f, delimiter=',',
                            quotechar='"')
        for row in reader:
            _, _, headline, content = row
            content = content.strip()
            word_counter.update(headline.split())
            word_counter.update(content.split())

    print("Word counter size: ", len(word_counter))

    counters = {
        "word": word_counter
    }
    vocabs = build_vocabs(counters)
    torch.save(vocabs, sys.argv[2])
