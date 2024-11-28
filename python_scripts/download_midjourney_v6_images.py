import ray
from datasets import load_dataset
import os
import os.path as osp
import json
from tqdm import tqdm

def ensure_parent_exists(path):
    parent = osp.dirname(path)
    if not osp.exists(parent):
        os.makedirs(parent)

@ray.remote
def download(url, output_dir):
    try:
        output_file = osp.join(output_dir, osp.basename(url))
        if osp.exists(output_file):
            return {
                "status": "already_exists",
                "output_file": output_file,
            }
        ensure_parent_exists(output_file)
        os.system(f"wget -q {url} -O {output_file}")
        
    except Exception as e:
        return str(e)

    return {
        "status": "success",
        "output_file": output_file,
    }

if __name__ == "__main__":
    # Initialize Ray
    ray.init(ignore_reinit_error=True)  # This starts Ray in the current process

    ds = load_dataset("CortexLM/midjourney-v6")
    output_dir = "tmp/midjourney-v6"
    train_ds = ds['train']
    
    result_ds = []
    futures = []  # List to hold the futures for Ray tasks
    
    for i, sample in enumerate(tqdm(train_ds)):
        # Call the download function remotely
        futures.append(download.remote(sample['image_url'], output_dir))
        
        # Keep track of the result for each sample
        sample['timestamp'] = str(sample['timestamp'])
        # sample['future'] = futures[-1]  # Store the future associated with this sample
        
        result_ds.append(sample)

    # Wait for all download tasks to complete and gather their results
    for i, future in enumerate(futures):
        result = ray.get(future)  # This will block until the result is ready
        result_ds[i].update(result)
        
        # Write the result to a file periodically
        if i % 100 == 0 or i == len(train_ds) - 1:
            with open(osp.join(output_dir, "result.json"), "w") as f:
                json.dump(result_ds, f)

    ray.shutdown()  # Shutdown Ray when done
