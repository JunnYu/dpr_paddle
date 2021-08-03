models = [
    "facebook/dpr-ctx_encoder-single-nq-base",
    "facebook/dpr-ctx_encoder-multiset-base",
    "facebook/dpr-question_encoder-single-nq-base",
    "facebook/dpr-question_encoder-multiset-base",
    "facebook/dpr-reader-single-nq-base",
    "facebook/dpr-reader-multiset-base",
]

urls = [
    "https://huggingface.co/facebook/dpr-ctx_encoder-single-nq-base/resolve/main/pytorch_model.bin",
    "https://huggingface.co/facebook/dpr-ctx_encoder-multiset-base/resolve/main/pytorch_model.bin",
    "https://huggingface.co/facebook/dpr-question_encoder-single-nq-base/resolve/main/pytorch_model.bin",
    "https://huggingface.co/facebook/dpr-question_encoder-multiset-base/resolve/main/pytorch_model.bin",
    "https://huggingface.co/facebook/dpr-reader-single-nq-base/resolve/main/pytorch_model.bin",
    "https://huggingface.co/facebook/dpr-reader-multiset-base/resolve/main/pytorch_model.bin",
]

model2url = dict(zip(models,urls))
import os
os.chdir("hg")
for model,url in model2url.items():
    print("download",model)
    os.chdir(model.split("/")[1])
    os.system(f"wget {url}")
    os.chdir("..")
