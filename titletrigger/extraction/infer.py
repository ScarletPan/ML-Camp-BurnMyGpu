import csv
import sys
from titletrigger.extraction.extractive_summarize import extractive_summarize, lead_summarize
from tqdm import tqdm


if __name__ == "__main__":
    fout = open(sys.argv[2], "w")
    with open(sys.argv[1]) as f:
        f.readline()
        reader = csv.reader(f, delimiter=',',
                            quotechar='"')
        for row in tqdm(reader):
            _, _, _, content = row
            sum_ = extractive_summarize(content.replace("\n", ""))
            if not sum_:
                sum_ = lead_summarize(content)
            fout.write(" ".join(sum_) + "\n")
    fout.close()