import json
import logging
import os
import random
import hydra
import itertools
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import os.path as osp
from accediff import LOCAL_PATH
import gradio as gr

logger = logging.getLogger(__name__)

MAX_SEED = 4294967295
MAX_IMAGE_SIZE = 1024

@hydra.main(version_base=None, config_path=f"{LOCAL_PATH}/configs/inference/generate", config_name="config_text_to_image")
def main(cfg: DictConfig):
    logger.info(str(OmegaConf.to_yaml(cfg)))
    main_serve(cfg)

def main_serve(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    pipe = hydra.utils.call(cfg.get_pipeline)

    def infer(
        prompt,
        negative_prompt="",
        seed=42,
        randomize_seed=False,
        width=1024,
        height=1024,
        guidance_scale=4.5,
        num_inference_steps=40,
        progress=gr.Progress(track_tqdm=True),
    ):
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)

        generator = torch.Generator().manual_seed(seed)

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
        ).images[0]

        return image, seed


    examples = [
            "A capybara wearing a suit holding a sign that reads Hello World",
    ]

    css = """
    #col-container {
        margin: 0 auto;
        max-width: 640px;
    }
    """

    with gr.Blocks(css=css) as demo:
        with gr.Column(elem_id="col-container"):
            gr.Markdown(" # [Stable Diffusion 3.5 Large (8B)](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)")
            gr.Markdown("[Learn more](https://stability.ai/news/introducing-stable-diffusion-3-5) about the Stable Diffusion 3.5 series. Try on [Stability AI API](https://platform.stability.ai/docs/api-reference#tag/Generate/paths/~1v2beta~1stable-image~1generate~1sd3/post), or [download model](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) to run locally with ComfyUI or diffusers.")
            with gr.Row():
                prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    container=False,
                )

                run_button = gr.Button("Run", scale=0, variant="primary")

            result = gr.Image(label="Result", show_label=False)

            with gr.Accordion("Advanced Settings", open=False):
                negative_prompt = gr.Text(
                    label="Negative prompt",
                    max_lines=1,
                    placeholder="Enter a negative prompt",
                    visible=True,
                    value=cfg.generate_kwargs.negative_prompt,
                )

                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )

                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

                with gr.Row():
                    width = gr.Slider(
                        label="Width",
                        minimum=512,
                        maximum=MAX_IMAGE_SIZE,
                        step=32,
                        value=1024, 
                    )

                    height = gr.Slider(
                        label="Height",
                        minimum=512,
                        maximum=MAX_IMAGE_SIZE,
                        step=32,
                        value=1024,
                    )

                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance scale",
                        minimum=0.0,
                        maximum=7.5,
                        step=0.1,
                        value=cfg.generate_kwargs.guidance_scale,
                    )

                    num_inference_steps = gr.Slider(
                        label="Number of inference steps",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=cfg.generate_kwargs.num_inference_steps,
                    )

            gr.Examples(examples=examples, inputs=[prompt], outputs=[result, seed], fn=infer, cache_examples=False, cache_mode="lazy")
        gr.on(
            triggers=[run_button.click, prompt.submit],
            fn=infer,
            inputs=[
                prompt,
                negative_prompt,
                seed,
                randomize_seed,
                width,
                height,
                guidance_scale,
                num_inference_steps,
            ],
            outputs=[result, seed],
        )

    demo.launch(share=True)

if __name__ == "__main__":
    import accelerate
    accelerate.utils.set_seed(42, deterministic=False)
    main()