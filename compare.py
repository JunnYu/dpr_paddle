import os
import torch
import paddle
import dpr_paddle
import transformers.models.dpr.modeling_dpr as dpr_hg

paddle.set_device("cpu")


for path in os.listdir("pd"):
    print(path)
    isreader = False
    if "ctx_encoder" in path:
        PDCLS = dpr_paddle.DPRContextEncoder
        PTCLS = dpr_hg.DPRContextEncoder
    if "question_encoder" in path:
        PDCLS = dpr_paddle.DPRQuestionEncoder
        PTCLS = dpr_hg.DPRQuestionEncoder
    if "reader" in path:
        isreader = True
        PDCLS = dpr_paddle.DPRReader
        PTCLS = dpr_hg.DPRReader

    pd_model = PDCLS.from_pretrained("pd/" + path)
    pd_model.eval()

    pt_model = PTCLS.from_pretrained("hg/" + path)
    pt_model.eval()

    ptmask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
    ptx = torch.tensor([[13, 212, 53, 421, 523, 126, 0, 0, 0, 0, 0]])

    pdmask = paddle.to_tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
    pdx = paddle.to_tensor([[13, 212, 53, 421, 523, 126, 0, 0, 0, 0, 0]])

    with paddle.no_grad():
        o = pd_model(pdx,attention_mask=pdmask)
        if isreader:
            pdoutputs = []
            for k in o:
                pdoutputs.append(torch.from_numpy(k.numpy()))
        else:
            pdoutputs = torch.from_numpy(o.numpy())

    with torch.no_grad():
        if isreader:
            start_logits, end_logits, relevance_logits = pt_model(ptx,attention_mask=ptmask)[:3]
            ptoutputs = [start_logits, end_logits, relevance_logits]
        else:
            ptoutputs = pt_model(ptx,attention_mask=ptmask)[0]

    def compare(a, b):
        if isinstance(a, list):
            for aa, bb in zip(a, b):
                print((aa - bb).abs().mean())
        else:
            print((a - b).abs().mean())

    compare(pdoutputs, ptoutputs)

