from datasets import load_dataset
import os
import os.path as osp
import json
from tqdm import tqdm

if __name__ == "__main__":
    # Initialize Ray

    ds = load_dataset("CortexLM/midjourney-v6")
    output_dir = "DATAS/midjourney-v6-jsonl-shards"
    os.makedirs(output_dir, exist_ok=True)
    train_ds = ds['train']
    sample_per_shard = 300 # 1000 shard
    f = None    
    for i, sample in enumerate(tqdm(train_ds)):
        if i % sample_per_shard == 0:
            if f is not None:
                f.close() 
            shard_id = i // sample_per_shard
            shard_dir = osp.join(output_dir, f"shard_{shard_id:05}.jsonl")
            f = open(shard_dir, "w")
        # Keep track of the result for each sample
        time_stamp = str(sample['timestamp'])
        img_filename = osp.basename(sample['image_url'])
        sample_id = '--'.join([time_stamp, sample['prompt'], img_filename])
        f.write(json.dumps({
            "id": sample_id,
            "image": img_filename,
            "prompt": sample['prompt'],
            "timestamp": time_stamp
        }) + "\n")
    f.close()