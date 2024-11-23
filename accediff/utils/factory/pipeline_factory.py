import torch
from diffusers import StableDiffusion3Pipeline
from transformers import T5EncoderModel, BitsAndBytesConfig
import logging
logger = logging.getLogger(__name__)

def get_sd3_quantized_pipeline(repo_id="stabilityai/stable-diffusion-3-medium-diffusers", drop_text_encoder_3=True):
    logger.info(f"Loading model from {repo_id}")
    torch.set_float32_matmul_precision("high")
    torch._inductor.config.conv_1x1_as_mm = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.epilogue_fusion = False
    torch._inductor.config.coordinate_descent_check_all_directions = True

    # Make sure you have `bitsandbytes` installed.
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model_id = repo_id
    extra_kwargs = {}
    if drop_text_encoder_3:
        extra_kwargs["text_encoder_3"] = None
        extra_kwargs["tokenizer_3"] = None
    else:
        text_encoder = T5EncoderModel.from_pretrained(
            model_id,
            subfolder="text_encoder_3",
            quantization_config=quantization_config,
        )
        extra_kwargs["text_encoder_3"] = text_encoder
        
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        device_map="balanced",
        torch_dtype=torch.float16,
        **extra_kwargs,
    )
    pipe.set_progress_bar_config(disable=True)
    # pipe.transformer.to(memory_format=torch.channels_last)
    # pipe.vae.to(memory_format=torch.channels_last)

    # pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
    # pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

    # # Warm Up
    # prompt = "a photo of a cat holding a sign that says hello world"
    # for _ in range(3):
    #     image = pipe(prompt=prompt, generator=torch.manual_seed(1))
    #     logger.info(f"Warming up output {image}")


    # pipe.enable_model_cpu_offload()
    return pipe

def get_sd3_pipeline(repo_id="stabilityai/stable-diffusion-3-medium-diffusers"):
    pipe = StableDiffusion3Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    return pipe



def get_sd3_pipeline_with_lora(
    repo_id="stabilityai/stable-diffusion-3.5-medium",
    lora_weight_dir=None,
):
    pipe = StableDiffusion3Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
    # lora_weight_path = osp.join(lora_weight_dir, 'trnasformer_lora', 'pytorch_lora_weights.safetensors')
    pipe.load_lora_weights(
        lora_weight_dir,
        weight_name="converted_pytorch_lora_weights.safetensors", 
        adapter_name="default",
    )

    pipe.fuse_lora()
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe

if __name__ == "__main__":
    # repo_id = "stabilityai/stable-diffusion-3-medium-diffusers"
    # torch.seed()

    repo_id = "stabilityai/stable-diffusion-3.5-large"

    pipe = get_sd3_quantized_pipeline(repo_id)
    # pipe = pipe.to("cuda")

    image = pipe(
        "A cat holding a sign that says hello world",
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=7.0,
    ).images[0]
    image.save('test.jpg')