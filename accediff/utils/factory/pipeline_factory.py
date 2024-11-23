import functools
import math
from typing import List, Tuple
import torch
from diffusers import StableDiffusion3Pipeline
from diffusers import CogVideoXPipeline
from transformers import T5EncoderModel, BitsAndBytesConfig
import torch
import torch.nn as nn
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

def get_cogvideox_pipeline(repo_id="THUDM/CogVideoX-5b"):
    pipe = CogVideoXPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to("cuda")
    return pipe


def find_modules_with_ending_pattern(model, patterns: List[str]):
    """
    Recursively finds all submodules whose names end with the specified pattern.
    
    Args:
        model (nn.Module): The PyTorch model to search through.
        pattern (str): The pattern to match at the end of module names.
    
    Returns:
        list: A list of tuples (name, module) matching the pattern.
    """
    matching_modules = []
    
    for name, module in model.named_modules():
        if any(name.endswith(pattern) for pattern in patterns):
            matching_modules.append((name, module))
    
    return matching_modules

def _wrap_token_merging(model: nn.Module, tome_modele_names=None, pre_forward_hook=None, post_forward_hook=None):
    for module_name, module in find_modules_with_ending_pattern(model, tome_modele_names):
        if pre_forward_hook is not None:
            logger.info(f"Wrapping {module_name} with pre forward hook {post_forward_hook}")
            module.register_forward_pre_hook(pre_forward_hook, with_kwargs=True)
        if post_forward_hook is not None:
            logger.info(f"Wrapping {module_name} with post forward hook {post_forward_hook}")
            module.register_forward_hook(post_forward_hook, with_kwargs=True)


def create_3d_gaussian_kernel(kernel_size=(1, 3, 3), sigma=1.0):
    # Generate grid coordinates
    z = torch.arange(-(kernel_size[0] // 2), kernel_size[0] // 2 + 1, dtype=torch.float32)
    y = torch.arange(-(kernel_size[1] // 2), kernel_size[1] // 2 + 1, dtype=torch.float32)
    x = torch.arange(-(kernel_size[2] // 2), kernel_size[2] // 2 + 1, dtype=torch.float32)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")

    # Compute Gaussian values
    gaussian = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))

    # Normalize kernel
    gaussian /= gaussian.sum()

    return gaussian

def _get_cogvideox_naive_gaussian_token_merging_hooks(pipe, token_merging_shape):
    frames_token_size = (13, pipe.transformer.config.sample_height // pipe.transformer.config.patch_size, pipe.transformer.config.sample_width // pipe.transformer.config.patch_size) # cogvideox has 13 frames
    frames_token_size_after_merging = (frames_token_size[0] // token_merging_shape[0], frames_token_size[1] // token_merging_shape[1], frames_token_size[2] // token_merging_shape[2])
    num_vision_tokens = math.prod(frames_token_size)
    num_vision_tokens_after_merging = math.prod(frames_token_size_after_merging)
    # use 3d gaussian kernel to merge tokens
    downsample_weight = create_3d_gaussian_kernel(kernel_size=token_merging_shape, sigma=1.0).reshape(1, 1, *token_merging_shape).to(pipe.transformer.device)
    def tome_pre_forward_hook(module, args, kwargs):
        # naive token merging
        if "hidden_states" in kwargs:
            hidden_states = kwargs["hidden_states"]
        elif len(args) > 0:
            hidden_states = args[0]
        else:
            raise ValueError("No hidden states found")
        
        # print("Before forward pass:", hidden_states.shape)

        bsz, seqlen, dim = hidden_states.shape
        # token merge on hidden states // video view
        encoder_hidden_states, hidden_states = hidden_states[:, :seqlen - num_vision_tokens, :], hidden_states[:,seqlen - num_vision_tokens: , :]
        
        assert hidden_states.shape[1] == num_vision_tokens
        hidden_states = hidden_states.view(bsz, frames_token_size[0], frames_token_size[1], frames_token_size[2], dim).permute(0, 4, 1, 2, 3).contiguous()
        # get gaussian kernel
        hidden_states = nn.functional.conv3d(hidden_states, downsample_weight, stride=token_merging_shape,)
        hidden_states = hidden_states.view(bsz, dim, -1).permute(0, 2, 1).contiguous()
        assert hidden_states.shape[1] == num_vision_tokens_after_merging

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        if "hidden_states" in kwargs:
            kwargs["hidden_states"] = hidden_states
        else:
            args = (hidden_states, *args[1:])
        return args, kwargs
    
    def tome_post_forward_hook(module, args, kwargs, output):
        # naive token unmerging
        if isinstance(output, torch.Tensor):
            hidden_states = output
        elif isinstance(output, tuple):
            hidden_states = output[0]
        else:
            raise ValueError("Output must be a tensor or a tuple")
        bsz, seqlen, dim = hidden_states.shape
        encoder_hidden_states, hidden_states = hidden_states[:, :seqlen - num_vision_tokens_after_merging, :], hidden_states[:, seqlen - num_vision_tokens_after_merging: , :]
        assert hidden_states.shape[1] == num_vision_tokens_after_merging
        hidden_states = hidden_states.view(bsz, frames_token_size_after_merging[0], frames_token_size_after_merging[1], frames_token_size_after_merging[2], dim).permute(0, 4, 1, 2, 3).contiguous()
        hidden_states = hidden_states.repeat_interleave(token_merging_shape[0], dim=2).repeat_interleave(token_merging_shape[1], dim=3).repeat_interleave(token_merging_shape[2], dim=4)
        hidden_states = hidden_states.view(bsz, dim, -1).permute(0, 2, 1).contiguous()
        assert hidden_states.shape[1] == num_vision_tokens
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        # print("After forward pass:", hidden_states.shape)
        if isinstance(output, torch.Tensor):
            return hidden_states
        elif isinstance(output, tuple):
            return (hidden_states, *output[1:])
    return tome_pre_forward_hook, tome_post_forward_hook

def get_cogvideox_pipeline_with_tome(repo_id="THUDM/CogVideoX-5b", tome_modele_names: List[str]=None):
    token_merging_shape = (1, 3, 3) # (temporal, height, width)
    
    if tome_modele_names is None:
        raise ValueError("tome_model_names must be provided")
        
    pipe = get_cogvideox_pipeline(repo_id)

    pre_forward_hook, post_forward_hook = _get_cogvideox_naive_gaussian_token_merging_hooks(pipe, token_merging_shape)
    _wrap_token_merging(pipe.transformer, tome_modele_names, pre_forward_hook, post_forward_hook)    
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