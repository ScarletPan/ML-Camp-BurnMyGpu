import sys
from nltk import ngrams


if __name__ == "__main__":
    fout = open(sys.argv[2], "w")
    with open(sys.argv[1]) as f:
        for line in f:
            tokens = line.strip().split()
            res = []
            old_token = ""
            for token in tokens:
                if token == old_token:
                    continue
                else:
                    res.append(token)
                    old_token = token
            bigram_tokens = ["_".join(t) for t in ngrams(res, 2)]
            res = []
            old_old_word = None
            old_token = None
            for token in bigram_tokens:
                if token == old_token or token == old_old_word:
                    continue
                else:
                    res.append(token.split("_")[0])
                old_old_word = old_token
                old_token = token
            res_ = []
            old_token = ""
            for token in res:
                if token == old_token:
                    continue
                else:
                    res_.append(token)
                    old_token = token
            fout.write(" ".join(res_) + "\n")
    fout.close()