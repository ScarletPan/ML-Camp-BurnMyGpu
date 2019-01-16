import jieba
from pycorenlp import StanfordCoreNLP


stanford_nlp = None


def tokenize(s, engine="jieba", port=None):
    if engine == "jieba":
        return list(jieba.cut(s))
    elif engine == "stanford":
        global stanford_nlp
        if not stanford_nlp:
            stanford_nlp = StanfordCoreNLP("http://localhost:%d" % port)
            result = stanford_nlp.annotate(s, properties={
                "annotators": "tokenize",
                "outputFormat": "json",
                "tokenize.language": "zh"
            })
            return [t["word"] for t in result["tokens"]]