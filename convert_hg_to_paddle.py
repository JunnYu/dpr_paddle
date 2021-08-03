from collections import OrderedDict
import argparse

huggingface_to_paddle = {
    "embeddings.LayerNorm": "embeddings.layer_norm",
    "encoder.layer": "encoder.layers",
    "attention.self.query": "self_attn.q_proj",
    "attention.self.key": "self_attn.k_proj",
    "attention.self.value": "self_attn.v_proj",
    "attention.output.dense": "self_attn.out_proj",
    "intermediate.dense": "linear1",
    "output.dense": "linear2",
    "attention.output.LayerNorm": "norm1",
    "output.LayerNorm": "norm2",
}

def convert_pytorch_checkpoint_to_paddle(pytorch_checkpoint_path,
                                         paddle_dump_path):

    import torch
    import paddle
    pytorch_state_dict = torch.load(pytorch_checkpoint_path, map_location="cpu")
    paddle_state_dict = OrderedDict()
    for k, v in pytorch_state_dict.items():
        if k[-7:] == ".weight":
            if ".embeddings." not in k and ".LayerNorm." not in k:
                v = v.transpose(0, 1)
        oldk = k
        for huggingface_name, paddle_name in huggingface_to_paddle.items():
            k = k.replace(huggingface_name, paddle_name)

        print(f"Converting: {oldk} => {k}")
        paddle_state_dict[k] = v.data.numpy()

    paddle.save(paddle_state_dict, paddle_dump_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pytorch_checkpoint_path",
        default="dpr-reader-multiset-base/pytorch_model.bin",
        type=str,
        required=True,
        help="Path to the Pytorch checkpoint path.")
    parser.add_argument(
        "--paddle_dump_path",
        default="dpr-reader-multiset-base/model_state.pdparams",
        type=str,
        required=True,
        help="Path to the output Paddle model.")
    args = parser.parse_args()
    convert_pytorch_checkpoint_to_paddle(args.pytorch_checkpoint_path,
                                         args.paddle_dump_path)