import os.path as osp
from pathlib import Path


def iter_prompts_from_txt_file(file):
    with open(file, "r") as f:
        for line in f:
            yield line.strip()
    
def iter_prompts_from_folder_of_txt_files(dir):
    txt_files = Path(dir).rglob("*.txt")
    for txt_file in txt_files:
        yield from iter_prompts_from_txt_file(txt_file)
    
def iter_prompts(prompt_sources):
    for prompt_source in prompt_sources:
        if osp.isdir(prompt_source):
            yield from iter_prompts_from_folder_of_txt_files(prompt_source)
        else:
            yield from iter_prompts_from_txt_file(prompt_source)