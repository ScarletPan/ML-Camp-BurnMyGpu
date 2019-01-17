import csv
import os
import sys
from tqdm import tqdm

if __name__ == "__main__":
    fout = open(sys.argv[2], "w")
    tag_set = set()
    with open(sys.argv[1]) as f:
        f.readline()
        reader = csv.reader(f, delimiter=',',
                            quotechar='"')
        for _, tag, _, content in tqdm(reader):
            content = content.strip().replace("\n", "")
            fout.write(content + " __label__" + tag + "\n")
            tag_set.add(tag)
    print(tag_set)
    fout.close()

