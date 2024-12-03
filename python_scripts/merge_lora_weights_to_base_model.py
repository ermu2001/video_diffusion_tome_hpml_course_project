import sys
import os.path as osp
from safetensors import safe_open
from safetensors.torch import save_file
import argparse

import torch
from accediff.utils.factory.pipeline_factory import get_sd3_pipeline_with_lora

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Lora weights to base model weights")
    parser.add_argument("--base_pipeline", help="")
    parser.add_argument("--lora_weight_path", help="Path to the output weight file")
    parser.add_argument("--output_dir", help="Path to the output weight dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        base_pipeline = get_sd3_pipeline_with_lora(args.base_pipeline, args.lora_weight_path)
        base_pipeline.unload_lora_weights()
        base_pipeline.save_pretrained(args.output_dir)