import logging
import os
import hydra
import itertools
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from accediff import LOCAL_PATH
from accediff.utils.factory.pipeline_factory import get_sd3_quantized_pipeline
from accediff.utils.factory.prompt_factory import iter_prompts
logger = logging.getLogger(__name__)

def iter_generate_prompt2image(
    pipe,
    prompt_sources,
    diffusion_kwargs,
):
    for prompt in iter_prompts(prompt_sources):
        yield pipe(prompt, **diffusion_kwargs).images[0]

@hydra.main(version_base=None, config_path=f"{LOCAL_PATH}/configs/inference/generate", config_name="validate")
def main(cfg: DictConfig):
    logger.info(str(OmegaConf.to_yaml(cfg)))
    main_generate(cfg)

def main_generate(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    pipe = hydra.utils.call(cfg.get_pipeline)

    generated_image_iterator = iter_generate_prompt2image(
        pipe,
        cfg.prompt_sources,
        cfg.generate_kwargs,
    )
    generated_image_iterator = itertools.islice(generated_image_iterator, cfg.num_images)
    images_progress_bar = tqdm(generated_image_iterator, total=cfg.num_images)
    
    logger.info(f"Saving images to {cfg.output_dir}")
    os.makedirs(cfg.output_dir, exist_ok=True)
    for i, image in enumerate(images_progress_bar):
        image.save(f"{cfg.output_dir}/{i}.jpg")

if __name__ == "__main__":
    import accelerate
    accelerate.utils.set_seed(42, deterministic=False)
    main()