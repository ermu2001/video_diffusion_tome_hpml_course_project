import logging
import os
import time
import hydra
import itertools
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from diffusers.utils import export_to_video

from accediff import LOCAL_PATH
from accediff.utils.utils import benchmark_time_iterator
from accediff.utils.factory.prompt_factory import iter_prompts
import logging
logger = logging.getLogger(__name__)

@benchmark_time_iterator(num_warmup=3, logger=logger)
def iter_generate_prompt2video(
    pipe,
    prompt_sources,
    num_total,
    diffusion_kwargs,
):
    for prompt in itertools.islice(iter_prompts(prompt_sources), num_total):
        yield pipe(prompt, **diffusion_kwargs).frames[0]

@hydra.main(version_base=None, config_path=f"{LOCAL_PATH}/configs/inference/generate", config_name="config_text_to_video")
def main(cfg: DictConfig):
    logger.info(str(OmegaConf.to_yaml(cfg)))
    main_generate(cfg)

def main_generate(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    pipe = hydra.utils.call(cfg.get_pipeline)

    generated_image_iterator = iter_generate_prompt2video(
        pipe,
        cfg.prompt_sources,
        cfg.num_videos,
        cfg.generate_kwargs,
    )
    frames_progress_bar = tqdm(generated_image_iterator, total=cfg.num_videos)
    
    logger.info(f"Saving videos to {cfg.output_dir}")
    os.makedirs(cfg.output_dir, exist_ok=True)
    for i, frames in enumerate(frames_progress_bar):
        export_to_video(frames, f"{cfg.output_dir}/{i}.mp4", fps=8)


if __name__ == "__main__":
    import accelerate
    accelerate.utils.set_seed(42, deterministic=False)
    main()