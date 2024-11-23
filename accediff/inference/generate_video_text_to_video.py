import logging
import os
import hydra
import itertools
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from diffusers.utils import export_to_video

from accediff import LOCAL_PATH
from accediff.utils.factory.prompt_factory import iter_prompts

def iter_generate_prompt2video(
    pipe,
    prompt_sources,
    diffusion_kwargs,
):
    for prompt in iter_prompts(prompt_sources):
        yield pipe(prompt, **diffusion_kwargs).frames[0]

@hydra.main(version_base=None, config_path=f"{LOCAL_PATH}/configs/inference/generate", config_name="config_text_to_video")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    main_generate(cfg)

def main_generate(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    pipe = hydra.utils.call(cfg.get_pipeline)

    generated_image_iterator = iter_generate_prompt2video(
        pipe,
        cfg.prompt_sources,
        cfg.generate_kwargs,
    )
    generated_image_iterator = itertools.islice(generated_image_iterator, cfg.num_videos)
    images_progress_bar = tqdm(generated_image_iterator, total=cfg.num_videos)
    
    logger.info(f"Saving videos to {cfg.output_dir}")
    os.makedirs(cfg.output_dir, exist_ok=True)
    for i, image in enumerate(images_progress_bar):
        # image.save(f"{cfg.output_dir}/{i}.jpg")
        export_to_video(image, f"{cfg.output_dir}/{i}.mp4", fps=8)


if __name__ == "__main__":
    import accelerate
    accelerate.utils.set_seed(42, deterministic=False)
    main()