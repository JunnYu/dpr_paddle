# dpr_paddle
dpr_paddle
Dense Passage Retrieval for Open-Domain Question Answering paddlepaddle复现

论文名称： Dense Passage Retrieval for Open-Domain Question Answering
数据集： 无
验收标准： 1. 复现竞品dpr模型（参考论文和竞品实现链接）
2. 完成模型权重从pytorch到paddle的转换代码，转换2种（6个）预训练权重（“facebook/dpr-ctx_encoder-single-nq-base”，
“facebook/dpr-ctx_encoder-multiset-base”，
“facebook/dpr-question_encoder-single-nq-base”，“facebook/dpr-question_encoder-multiset-base”，
“facebook/dpr-reader-single-nq-base”，“facebook/dpr-reader-multiset-base”）
3. DPRContextEncoder网络，DPRQuestionEncoder网络和DPRReader网络前向推理输出对齐竞品（上述两种权重）

# requirements
pip install paddlenlp

# convert
```bash
python convert_hg_to_paddle.py \
    --pytorch_checkpoint_path hg/dpr-reader-multiset-base/pytorch_model.bin \
    --paddle_dump_path pd/dpr-reader-multiset-base/model_state.pdparams
```

# 精度对齐
```python
python compare.py
dpr-ctx_encoder-multiset-base
tensor(4.5952e-07)
dpr-ctx_encoder-single-nq-base
tensor(6.9349e-07)
dpr-question_encoder-multiset-base
tensor(1.7658e-06)
dpr-question_encoder-single-nq-base
tensor(4.1322e-07)
dpr-reader-multiset-base
tensor(5.0176e-06)
tensor(1.8369e-06)
tensor(1.3828e-05)
dpr-reader-single-nq-base
tensor(1.9160e-05)
tensor(3.0648e-05)
tensor(2.8610e-06)
```