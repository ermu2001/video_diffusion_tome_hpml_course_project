import sys
import os.path as osp
from safetensors import safe_open
from safetensors.torch import save_file

if __name__ == "__main__":
    input_weight_path = sys.argv[1]
    output_weight_path = osp.join(osp.dirname(input_weight_path), "converted_" + osp.basename(input_weight_path))
    with safe_open(input_weight_path, "pt") as f:
        weights = {k: f.get_tensor(k) for k in f.keys()}
    
    new_weights = {}
    for k, v in weights.items():
        if k.startswith("transformer.base_model.model."):
            k = k.replace("base_model.model.", "", 1)
        new_weights[k] = v
    
    print(new_weights.keys())
    save_file(new_weights, output_weight_path)