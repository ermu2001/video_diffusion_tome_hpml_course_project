import json
import logging
import os
import hydra
import itertools
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import os.path as osp
from accediff import LOCAL_PATH
from accediff.utils.factory.prompt_factory import iter_prompts
from accediff.utils.utils import benchmark_time_iterator
logger = logging.getLogger(__name__)

@benchmark_time_iterator(num_warmup=3, logger=logger)
def iter_generate_prompt2image(
    pipe,
    prompt_sources,
    num_total,
    diffusion_kwargs,
):
    for prompt in itertools.islice(iter_prompts(prompt_sources), num_total):
        yield pipe(prompt, **diffusion_kwargs).images[0], prompt

@hydra.main(version_base=None, config_path=f"{LOCAL_PATH}/configs/inference/generate", config_name="config_text_to_image")
def main(cfg: DictConfig):
    logger.info(str(OmegaConf.to_yaml(cfg)))
    main_generate(cfg)

def main_generate(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    pipe = hydra.utils.call(cfg.get_pipeline)

    generated_image_iterator = iter_generate_prompt2image(
        pipe,
        cfg.prompt_sources,
        cfg.num_images,
        cfg.generate_kwargs,
    )
    images_progress_bar = tqdm(generated_image_iterator, total=cfg.num_images)
    
    logger.info(f"Saving images to {cfg.output_dir}")
    os.makedirs(cfg.output_dir, exist_ok=True)
    generated_image_dir = osp.join(cfg.output_dir, "generated_images")
    generation_info_filepath = osp.join(cfg.output_dir, "generation_info.json")
    result_infos = []
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(generated_image_dir, exist_ok=True)
    for i, (image, prompt) in enumerate(images_progress_bar):
        image_filename = osp.join(generated_image_dir, f"{i}.jpg")
        image.save(image_filename)
        result_infos.append({
            "prompt": prompt,
            "image_filename": f"{i}.jpg",
        })
    
    with open(generation_info_filepath, "w") as f:
        json.dump(result_infos, f, indent=4)


if __name__ == "__main__":
    import accelerate
    accelerate.utils.set_seed(42, deterministic=False)
    main()