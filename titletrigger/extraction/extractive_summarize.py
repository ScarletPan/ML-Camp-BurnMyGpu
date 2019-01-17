# encoding: utf8
from gensim.summarization.summarizer import summarize
import re
import jieba

paragraph = u'\u70ed\u5e26\u98ce\u66b4\u5c1a\u5854\u5c14\u662f2001\u5e74\u5927\u897f\u6d0b\u98d3\u98ce\u5b63\u7684\u4e00\u573a\u57288\u6708\u7a7f\u8d8a\u4e86\u52a0\u52d2\u6bd4\u6d77\u7684\u5317\u5927\u897f\u6d0b\u70ed\u5e26\u6c14\u65cb\u3002\u5c1a\u5854\u5c14\u4e8e8\u670814\u65e5\u7531\u70ed\u5e26\u5927\u897f\u6d0b\u7684\u4e00\u80a1\u4e1c\u98ce\u6ce2\u53d1\u5c55\u800c\u6210\uff0c\u5176\u5b58\u5728\u7684\u5927\u90e8\u5206\u65f6\u95f4\u91cc\u90fd\u5728\u5feb\u901f\u5411\u897f\u79fb\u52a8\uff0c\u9000\u5316\u6210\u4e1c\u98ce\u6ce2\u540e\u7a7f\u8d8a\u4e86\u5411\u98ce\u7fa4\u5c9b\u3002'


def zng(paragraph):
    for sent in re.findall(u'[^!?。\.\!\?]+[!?。\.\!\?]?', paragraph, flags=re.U):
        yield sent


def extractive_summarize(content):
    def split(content):
        '''分句和分词'''
        sents = list(zng(content))  # 先进行分句
        _sents = []
        for sent in sents:
            words = list(jieba.cut(sent)) # 分词
            sent = ' '.join(words) # 用空格把词隔开
            _sents.append(sent)
        content = '. '.join(_sents)  # 用.把句子隔开
        return content

    def clean(content):
        content = content.replace('.', '') # 删除句子分隔符
        content = content.replace(' ', '') # 删除空格
        content = content.replace("\n", "")
        return content

    tokens = split(content)
    try:
        result = summarize(tokens)
        result = clean(result)
    except:
        result = None
    if result:
        return list(jieba.cut("".join(result.split(" "))))
    else:
        return result


def lead_summarize(content):
    sents = list(zng(content))  # 先进行分句
    return list(jieba.cut("".join(sents[0].split())))