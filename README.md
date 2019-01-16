# ML-Camp-BurnMyGpu
Google ML冬令营-燃烧我的GPU-项目

## 项目进度

* Tag Classification
- [x] (Pipeline) Training code
- [x] (Pipeline) Test code
- [x] (Pipeline) Inference API
- [x] (Model) PoolLSTM
- [ ] (Model) +Bert

* Keywords Extraction
- [x] (Pipeline) Test code
- [ ] (Pipeline) Inference API

* Abstractive Summarization
- [x] (Pipeline) Training code
- [x] (Pipeline) Test code
- [x] (Pipeline) Inference API
- [x] (Model) Seq2seq+Attn
- [x] (Model) CopyNet
- [ ] (Model) +Coverage
- [ ] (Transfer) +Sougou
- [ ] (Performance) Real-time inference

* Text Style Transfering
- [ ] (Pipeline) Training code
- [ ] (Pipeline) Test code
- [ ] (Pipeline) Inference API

* Demo
- [ ] Basic Layout
- [ ] Text Highlight


## 路径说明
```
data/
    data/raw/         # 原始未处理数据
        chinese_new/  # - 比赛数据（新闻联播）
        sougou/       # - 搜狐大数据数据
    data/tokenized    # 分词后数据
        chinese_new/
        sougou/
documents/ # AI camp 下发的一些文档

notebook/ # ipython notebooks

textsum/  # 文本摘要生成代码
```

## 数据统计信息
### 新闻联播数据
总数据20738，其中107为无用数据，因此总可用数据为20631个样本。