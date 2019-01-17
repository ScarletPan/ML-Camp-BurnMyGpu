import logging
import sys
import time
import json
import jieba.analyse
jieba.setLogLevel(logging.ERROR)
jieba.initialize()

from titletrigger.api import load_model, abs_summarize, tag_classify, extract_keywords, ext_summarize
sys.path.append('titletrigger/textsum')
sys.path.append('titletrigger/textclf')


if __name__ == "__main__":
    sum_model_path = "/home/hjpan/projects/ML-Camp-BurnMyGpu/titletrigger/textsum/cache/copynet/model/best_model.pt"
    clf_model_path = "/home/hjpan/projects/ML-Camp-BurnMyGpu/titletrigger/textclf/cache/rcnn/model/best_model.pt"

    content = """国务院总理李克强21日下午在中南海紫光阁会见中印边界问题印方特别代表、印度国家安全顾问多瓦尔。
    李克强表示，中印边界问题特别代表会晤机制为双方增进互信、扩大共识发挥了建设性作用。
    我们要继续从中印关系大局出发，探讨通过外交途径以和平方式妥善解决边界问题。
    在找到公平合理、双方都能接受的解决方案前，一定要管控好分歧，共同致力于维护边境地区的和平与安宁。
    这也可以为两国深入推进经贸合作提供稳定的预期。李克强指出，当前世界经济复苏乏力，地缘政治动荡更加突出。
    中印作为两个最大的新兴经济体，经济保持中高速增长，对世界是鼓舞，对亚洲是带动。
    双方要珍惜和维护好两国关系发展势头，充分发挥经济互补优势，开展多领域务实合作，
    密切在国际和地区事务中的沟通协调，发出中印携手维护和平稳定、促进发展进步的积极信号。
    多瓦尔表示，印中关系取得了积极进展，两国既面临发展经济的艰巨挑战，也拥有开展合作的巨大机遇。
    印方愿同中方加强高层交往，深化经济、安全等各领域合作，妥善处理边界问题，推动两国关系取得更大发展。"""

    content_list = [" ".join(jieba.cut(content))]
    ext_headline = " ".join(ext_summarize(content_list)[0])
    print("Loading model...\n")
    sum_model_file = load_model(sum_model_path)
    clf_model_file = load_model(clf_model_path)
    # print("="*50)

    st = time.time()
    result_dict = abs_summarize(content_list, sum_model_file)
    print("Textsum Finished in {:.4f} s".format(time.time() - st))
    preds = result_dict["preds"]

    st = time.time()
    tag = tag_classify(content_list, clf_model_file)
    print("Textclf Finished in {:.4f} s".format(time.time() - st))
    print("News Content: ")
    print("".join(content_list[0].split()))
    print()
    print("Headline: ")
    print("EXT:")
    print(ext_headline)
    print()
    print("ABS:")
    for pred, score in result_dict["all_preds"][0]:
        print("{:.2f}: {}".format(score, "".join(pred)))
    print()
    print("Tag: ")
    print(tag[0])
    print()
    print("Keywords: ")
    keywords = " ".join(
        extract_keywords("".join(content_list[0].split()))).replace("\n", "")
    print(keywords)

