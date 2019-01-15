import csv
import sys


if __name__ == "__main__":
    fout = open(sys.argv[2], "w")
    with open(sys.argv[1]) as f:
        f.readline()
        reader = csv.reader(f, delimiter=',',
                            quotechar='"')
        for row in reader:
            _, _, _, content = row
            content = content.strip()
            summary = []
            for token in content.split():
                if token == "。":
                    break
                if token == "\n":
                    continue
                summary.append(token)
            fout.write(" ".join(summary) + "\n")
    fout.close()