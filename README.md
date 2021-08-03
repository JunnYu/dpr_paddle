# DPR paddle2.x实现
[Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/pdf/2004.04906.pdf)论文paddlepaddle2.x复现

验收标准： 
1. 复现竞品dpr模型（参考论文和竞品实现链接）
2. 完成模型权重从pytorch到paddle的转换代码，转换2种（6个）预训练权重
["facebook/dpr-ctx_encoder-single-nq-base",
"facebook/dpr-ctx_encoder-multiset-base",
"facebook/dpr-question_encoder-single-nq-base",
"facebook/dpr-question_encoder-multiset-base",
"facebook/dpr-reader-single-nq-base",
"facebook/dpr-reader-multiset-base"]
3. DPRContextEncoder网络，DPRQuestionEncoder网络和DPRReader网络前向推理输出对齐竞品（上述两种权重）

# requirements
```bash
pip install paddlenlp
pip install torch
pip install transformers
```

# 一、权重转换
## 方法一：本地转换权重
### (1)下载huggingface权重
```bash
    python download_hg_model.py
```
### (2)权重转换
```bash
    python convert_hg_to_paddle.py
```
## 方法二：从谷歌网盘下载
下载地址：https://drive.google.com/file/d/1dB3hI0weP1uar8S3anIlRwdxhbS2sctS/view?usp=sharing



# 二、DPRContextEncoder网络，DPRQuestionEncoder网络和DPRReader网络前向推理输出对齐
准备好模型权重后，进行精度对齐。
```python
    python compare.py
    dpr-ctx_encoder-multiset-base : tensor(4.5952e-07)
    dpr-ctx_encoder-single-nq-base  : tensor(6.9349e-07)
    dpr-question_encoder-multiset-base :tensor(1.7658e-06)
    dpr-question_encoder-single-nq-base :tensor(4.1322e-07)
    dpr-reader-multiset-base : tensor(5.0176e-06),tensor(1.8369e-06),tensor(1.3828e-05)
    dpr-reader-single-nq-base : tensor(1.9160e-05),tensor(3.0648e-05),tensor(2.8610e-06)
```

# 注！
由于下载hg权重的时候使用了`wget`，如果不是linux系统，请自己手动下载权重！
