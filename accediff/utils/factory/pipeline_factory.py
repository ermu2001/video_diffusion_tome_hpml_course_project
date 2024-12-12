import functools
import itertools
import os.path as osp
import math
from typing import Callable, List, Tuple
from braceexpand import braceexpand
import torch
from diffusers import StableDiffusion3Pipeline
from diffusers import CogVideoXPipeline
from transformers import T5EncoderModel, BitsAndBytesConfig
import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)

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
    patterns = list(itertools.chain.from_iterable(braceexpand(pattern) for pattern in patterns))

    for name, module in model.named_modules():
        in_pattern = [name.endswith(pattern) for pattern in patterns]
        if any(in_pattern):
            matching_modules.append((name, module))
    
    return matching_modules

def _wrap_token_merging(model: nn.Module, tome_modele_names=None, pre_forward_hook=None, post_forward_hook=None):
    register_list = find_modules_with_ending_pattern(model, tome_modele_names)
    if len(register_list) == 0:
        raise ValueError(f"No modules found with names ending with {tome_modele_names}")
    for module_name, module in register_list:
        if pre_forward_hook is not None:
            logger.info(f"Wrapping {module_name} with pre forward hook {post_forward_hook}")
            module.register_forward_pre_hook(pre_forward_hook, with_kwargs=True)
        if post_forward_hook is not None:
            logger.info(f"Wrapping {module_name} with post forward hook {post_forward_hook}")
            module.register_forward_hook(post_forward_hook, with_kwargs=True)



def do_nothing(x: torch.Tensor, mode: str = None):
    return x

def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


def _get_cogvideox_tome_token_merging_hooks_3d(pipe, merge_ratio, st, sx, sy):
    
    def bipartite_soft_matching_random3d(metric: torch.Tensor,
                                         f: int, w: int, h: int, st:int, sx: int, sy: int, r: int,
                                         no_rand: bool = False,
                                         generator: torch.Generator = None) -> Tuple[Callable, Callable]:
        """
        Partitions the tokens into src and dst and merges r tokens from src to dst.
        Dst tokens are partitioned by choosing one randomly in each (sx, sy) region.

        Args:
         - metric [B, N, C]: metric to use for similarity
         - f: video frames in tokens
         - w: image width in tokens
         - h: image height in tokens
         - st: stride in the time dimension for dst, must divide f
         - sx: stride in the x dimension for dst, must divide w
         - sy: stride in the y dimension for dst, must divide h
         - r: number of tokens to remove (by merging)
         - no_rand: if true, disable randomness (use top left corner only)
         - generator: random number generator
        """
        B, N, _ = metric.shape

        if r <= 0:
            return do_nothing, do_nothing

        gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

        with torch.no_grad():
            fst, hsy, wsx = f // st, h // sy, w // sx

            # Ensure sx and sy divide w and h, no need
            # assert f % st == 0 and w % sx == 0 and h % sy == 0, f"st ({st}) must divide f ({f}), sx ({sx}) must divide w ({w}) and sy ({sy}) must divide h ({h}) respectively."

            # For each sy by sx kernel, randomly assign one token to be dst and the rest src
            if no_rand:
                rand_idx = torch.zeros(fst, hsy, wsx, 1, device=metric.device, dtype=torch.int64)
            else:
                rand_idx = torch.randint(st * sy * sx, size=(fst, hsy, wsx, 1), device=generator.device, generator=generator).to(metric.device)

            idx_buffer_view = torch.zeros(fst, hsy, wsx, st * sy * sx, device=metric.device, dtype=torch.int64)
            idx_buffer_view.scatter_(dim=3, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
            idx_buffer_view = idx_buffer_view.view(fst, hsy, wsx, st, sy, sx).permute(0, 3, 1, 4, 2, 5).reshape(fst * st, hsy * sy, wsx * sx)

            if (fst * st) < f or (hsy * sy) < h or (wsx * sx) < w:
                idx_buffer = torch.zeros(f, h, w, device=metric.device, dtype=torch.int64)
                idx_buffer[:(fst * st), :(hsy * sy), :(wsx * sx)] = idx_buffer_view
            else:
                idx_buffer = idx_buffer_view

            rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1) # add a batch size dim and feature dim for broadcasting

            del idx_buffer, idx_buffer_view

            num_dst = fst * hsy * wsx
            num_src = N - num_dst

            a_idx = rand_idx[:, num_dst:, :]  # src
            b_idx = rand_idx[:, :num_dst, :]  # dst

            def split(x):
                C = x.shape[-1]
                src = gather(x, dim=1, index=a_idx.expand(B, num_src, C))
                dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
                return src, dst

            # Cosine similarity between A and B
            metric_norm = metric / metric.norm(dim=-1, keepdim=True)
            a, b = split(metric_norm)
            scores = a @ b.transpose(-1, -2) # (B, num_src, num_dst)

            # Can't reduce more than the # tokens in src
            r = min(a.shape[1], r)

            # Find the most similar greedily
            node_max, node_idx = scores.max(dim=-1) # find the most similar dst for each src
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
            src_idx = edge_idx[..., :r, :]  # Merged Tokens
            dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
            src, dst = split(x)
            n, t1, c = src.shape

            unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
            src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

            return torch.cat([unm, dst], dim=1)

        def unmerge(x: torch.Tensor) -> torch.Tensor:
            unm_len = unm_idx.shape[1]
            unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
            _, _, c = unm.shape

            src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

            # Combine back to the original shape
            out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
            out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
            out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
            out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)

            return out

        return merge, unmerge

    # Print available config keys to identify the correct keys
    logger.info(f"Available config keys: {pipe.transformer.config.keys()}")

    # Use the correct key for the number of frames
    # Assuming 'num_frames' is the correct key; if not, adjust accordingly
    frames_token_size = (
        pipe.transformer.config.get('num_frames', 12),  # Default to 12 if not found
        pipe.transformer.config['sample_height'] // pipe.transformer.config['patch_size'],
        pipe.transformer.config['sample_width'] // pipe.transformer.config['patch_size']
    )
    frames, h, w = frames_token_size

    num_vision_tokens = frames * h * w

    logger.info(f"f: {frames}, h: {h}, w: {w}, st: {st}, sx: {sx}, sy: {sy}")

    # Ensure st, sx and sy divide f, h and w, no need
    # assert frames % st == 0 and w % sx == 0 and h % sy == 0, f"st ({st}) must divide f ({frames}), sx ({sx}) must divide w ({w}) and sy ({sy}) must divide h ({h}) respectively."
    # Number of tokens to remove per frame
    fst, hsy, wsx = frames // st, h // sy, w // sx
    # num_dst_per_frame = hsy * wsx
    # num_src_per_frame = h * w - num_dst_per_frame
    num_dst = fst * hsy * wsx
    num_src = num_vision_tokens - num_dst
    
    # Set merge ratio (fraction of src tokens to merge)
    r = int(num_src * merge_ratio)

    # Generator for randomness
    generator = torch.Generator(device=pipe.transformer.device)

    merge_fns = {}
    unmerge_fns = {}
    num_tokens_after_merging_per_frame = {}

    def tome_pre_forward_hook(module, args, kwargs):
        module_id = id(module)
        nonlocal merge_fns, unmerge_fns, num_tokens_after_merging_per_frame

        if "hidden_states" in kwargs:
            hidden_states = kwargs["hidden_states"]
        elif len(args) > 0:
            hidden_states = args[0]
        else:
            raise ValueError("No hidden states found")

        bsz, seqlen, dim = hidden_states.shape

        # Split encoder and vision tokens
        encoder_hidden_states = hidden_states[:, :seqlen - num_vision_tokens, :]
        vision_hidden_states = hidden_states[:, seqlen - num_vision_tokens:, :]

        # (B, N, C) == （B, f * h * w, C)
        # Call bipartite_soft_matching_random2d
        merge_fn, unmerge_fn = bipartite_soft_matching_random3d(
            metric=vision_hidden_states,
            w=w,
            h=h,
            f=frames,
            sx=sx,
            sy=sy,
            st=st,
            r=r,
            no_rand=False,
            generator=generator
        )

        # Store merge and unmerge functions for this module
        merge_fns[module_id] = merge_fn
        unmerge_fns[module_id] = unmerge_fn

        # Merge the vision_hidden_states
        vision_hidden_states = merge_fn(vision_hidden_states, mode="mean")
        # Store the number of tokens per frame after merging
        num_tokens_per_frame_after_merging = vision_hidden_states.shape[1]
        num_tokens_after_merging_per_frame[module_id] = num_tokens_per_frame_after_merging

        # Update hidden_states
        hidden_states = torch.cat([encoder_hidden_states, vision_hidden_states], dim=1)

        if "hidden_states" in kwargs:
            kwargs["hidden_states"] = hidden_states
        else:
            args = (hidden_states, *args[1:])

        return args, kwargs

    def tome_post_forward_hook(module, args, kwargs, output):
        module_id = id(module)
        nonlocal merge_fns, unmerge_fns, num_tokens_after_merging_per_frame

        if isinstance(output, torch.Tensor):
            hidden_states = output
        elif isinstance(output, tuple):
            hidden_states = output[0]
        else:
            raise ValueError("Output must be a tensor or a tuple")

        bsz, seqlen, dim = hidden_states.shape

        # Retrieve merge and unmerge functions
        merge_fn = merge_fns.get(module_id)
        unmerge_fn = unmerge_fns.get(module_id)
        num_tokens_per_frame_after_merging = num_tokens_after_merging_per_frame.get(module_id)

        if merge_fn is None or unmerge_fn is None or num_tokens_per_frame_after_merging is None:
            raise ValueError("Merge and unmerge functions not found for module")

        # Calculate the number of vision tokens after merging
        num_vision_tokens_after_merging = num_tokens_per_frame_after_merging * frames

        # Split encoder and vision tokens
        encoder_hidden_states = hidden_states[:, :seqlen - num_vision_tokens_after_merging, :]
        vision_hidden_states = hidden_states[:, seqlen - num_vision_tokens_after_merging:, :]

        # Apply unmerge function
        vision_hidden_states = unmerge_fn(vision_hidden_states)

        # Reshape back to (B, frames, h * w, dim)
        vision_hidden_states = vision_hidden_states.reshape(bsz, frames, h * w, dim)

        # Flatten frames and spatial dimensions
        vision_hidden_states = vision_hidden_states.reshape(bsz, frames * h * w, dim)

        # Concatenate back
        hidden_states = torch.cat([encoder_hidden_states, vision_hidden_states], dim=1)

        if isinstance(output, torch.Tensor):
            return hidden_states
        elif isinstance(output, tuple):
            return (hidden_states, *output[1:])

    return tome_pre_forward_hook, tome_post_forward_hook

from matplotlib import pyplot as plt
import os
def _debug_plot_histogram(t, prefix=''):
    debug_out_dir = "debug_out"
    if not os.path.exists(debug_out_dir):
        os.makedirs(debug_out_dir)
    b, c = t.shape
    t = t.detach().cpu().numpy()
    # fig plt.subplot(1, b, 1)
    fig, axs = plt.subplots(1, b)
    for i, t_ in enumerate(t):
        axs[i].hist(t_, bins=100)
        axs[i].set_title(f"Histogram {i + 1}")
    
    # Save the figure
    fig.tight_layout()
    fig.savefig(f"debug_out/{prefix}_histogram.png")
    plt.close(fig)  # Close the figure to free memory

def _get_cogvideox_tome_token_merging_hooks_attnbin(pipe, merge_ratio, sb):
    def bipartite_soft_matching_attnbin(metric: torch.Tensor,
                                        num_bins: int,
                                        sb: int,
                                        r: int,
                                        no_rand: bool = False,
                                        generator: torch.Generator = None) -> Tuple[Callable, Callable]:
        """
        Partitions the tokens into src and dst and merges r tokens from src to dst.
        Dst tokens are partitioned by choosing one randomly in each (sx, sy) region.

        Args:
         - metric [B, N, C]: metric to use for similarity
         - f: video frames in tokens
         - w: image width in tokens
         - h: image height in tokens
         - st: stride in the time dimension for dst, must divide f
         - sx: stride in the x dimension for dst, must divide w
         - sy: stride in the y dimension for dst, must divide h
         - r: number of tokens to remove (by merging)
         - no_rand: if true, disable randomness (use top left corner only)
         - generator: random number generator
        """
        B, N, feature_dim = metric.shape
        if r <= 0:
            return do_nothing, do_nothing

        gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather


        with torch.no_grad():
            # For each sy by sx kernel, randomly assign one token to be dst and the rest src
            if no_rand:
                rand_idx = torch.zeros(num_bins, 1, device=metric.device, dtype=torch.int64)
            else:
                rand_idx = torch.randint(sb, size=(num_bins, 1), device=generator.device, generator=generator).to(metric.device)

            idx_buffer_view = torch.zeros(num_bins, sb, device=metric.device, dtype=torch.int64)
            idx_buffer_view.scatter_(dim=1, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
            idx_buffer_view = idx_buffer_view.reshape(num_bins * sb)
            if (num_bins * sb) < N:
                idx_buffer = torch.zeros(N, device=metric.device, dtype=torch.int64)
                idx_buffer[:(num_bins * sb)] = idx_buffer_view
            else:
                idx_buffer = idx_buffer_view

            rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1) # add a batch size dim and feature dim for broadcasting
            del idx_buffer, idx_buffer_view

            num_dst = num_bins
            num_src = N - num_dst

            a_idx = rand_idx[:, num_dst:, :]  # src
            b_idx = rand_idx[:, :num_dst, :]  # dst

            def split(x):
                C = x.shape[-1]
                src = gather(x, dim=1, index=a_idx.expand(B, num_src, C))
                dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
                return src, dst

            # take bin by:
            # norm of the features
            # metric_sort_value = metric.norm(dim=-1, keepdim=True) # not ok
            # # select the most important feature as dst
            metric_sort_value = metric.norm(dim=-1, keepdim=True) # not ok
            # # similarity to major direction
            # # _debug_plot_histogram(metric.norm(dim=-1), prefix=f'layer')
            # # major direction 1 : ALL BLACK
            # major_direction_index = torch.argmax(metric.norm(dim=-1, keepdim=True), dim=1, keepdim=True)
            # metric_normed = metric / metric.norm(dim=-1, keepdim=True)
            # major_direction = gather(metric_normed, 1, major_direction_index.expand(-1, -1, feature_dim))
            # metric_sort_value = torch.sum(metric_normed * major_direction, dim=-1, keepdim=True) # sim_to_major_direction
            
            # # major direction 2 : QUICKLY DISPAIRS
            # metric_normed = metric / metric.norm(dim=-1, keepdim=True)
            # major_direction = torch.mean(metric_normed, 1, keepdim=True)
            # metric_sort_value = torch.sum(metric_normed * major_direction, dim=-1, keepdim=True) # sim_to_major_direction

            # sort and tome
            sort_value, sort_index = torch.sort(metric_sort_value, dim=1, descending=True) # from small to large sort

            sorted_metric = gather(metric, 1, sort_index.expand(-1, -1, feature_dim))
            sorted_metric = sorted_metric / sort_value
            a, b = split(sorted_metric)
            scores = a @ b.transpose(-1, -2) # (B, num_src, num_dst)

            # Can't reduce more than the # tokens in src
            r = min(a.shape[1], r)

            # Find the most similar greedily
            node_max, node_idx = scores.max(dim=-1) # find the most similar dst for each src
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            # index back to the unsroted index

            unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
            src_idx = edge_idx[..., :r, :]  # Merged Tokens
            dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
            x = gather(x, 1, sort_index.expand(-1, -1, feature_dim))
            src, dst = split(x)
            n, t1, c = src.shape

            unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
            src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
            return torch.cat([unm, dst], dim=1)

        def unmerge(x: torch.Tensor) -> torch.Tensor:
            unm_len = unm_idx.shape[1]
            unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
            _, _, c = unm.shape

            src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

            # Combine back to the original shape
            out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
            out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
            out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
            out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)
            out = gather(out, 1, torch.argsort(sort_index, dim=1).expand(-1, -1, c))
            return out

        return merge, unmerge

    # Print available config keys to identify the correct keys
    logger.info(f"Available config keys: {pipe.transformer.config.keys()}")

    # Use the correct key for the number of frames
    # Assuming 'num_frames' is the correct key; if not, adjust accordingly
    frames_token_size = (
        pipe.transformer.config.get('num_frames', 12),  # Default to 12 if not found
        pipe.transformer.config['sample_height'] // pipe.transformer.config['patch_size'],
        pipe.transformer.config['sample_width'] // pipe.transformer.config['patch_size']
    )
    frames, h, w = frames_token_size

    num_vision_tokens = frames * h * w

    # stride bin determines the shape for selecting the dst
    num_bins = num_vision_tokens // sb # num_vision_tokens ~ 12 * 30 * 45
    num_dst = num_bins
    num_src = num_vision_tokens - num_dst
    logger.info(f"f: {frames}, h: {h}, w: {w}, sb: {sb}")
    
    # Set merge ratio (fraction of src tokens to merge)
    r = int(num_src * merge_ratio)

    # Generator for randomness
    generator = torch.Generator(device=pipe.transformer.device)

    merge_fns = {}
    unmerge_fns = {}
    num_tokens_after_mergings = {}

    def tome_pre_forward_hook(module, args, kwargs):
        module_id = id(module)
        nonlocal merge_fns, unmerge_fns, num_tokens_after_mergings

        if "hidden_states" in kwargs:
            hidden_states = kwargs["hidden_states"]
        elif len(args) > 0:
            hidden_states = args[0]
        else:
            raise ValueError("No hidden states found")

        bsz, seqlen, dim = hidden_states.shape

        # Split encoder and vision tokens
        encoder_hidden_states = hidden_states[:, :seqlen - num_vision_tokens, :]
        vision_hidden_states = hidden_states[:, seqlen - num_vision_tokens:, :]

        # (B, N, C) == （B, f * h * w, C)

        # Call bipartite_soft_matching_random2d
        merge_fn, unmerge_fn = bipartite_soft_matching_attnbin(
            metric=vision_hidden_states,
            num_bins=num_bins,
            sb=sb,
            r=r,
            no_rand=False,
            generator=generator
        )

        # Store merge and unmerge functions for this module
        merge_fns[module_id] = merge_fn
        unmerge_fns[module_id] = unmerge_fn

        # Merge the vision_hidden_states
        vision_hidden_states = merge_fn(vision_hidden_states, mode="mean")
        # Store the number of tokens per frame after merging
        num_tokens_after_merging = vision_hidden_states.shape[1]
        num_tokens_after_mergings[module_id] = num_tokens_after_merging

        # Update hidden_states
        hidden_states = torch.cat([encoder_hidden_states, vision_hidden_states], dim=1)

        if "hidden_states" in kwargs:
            kwargs["hidden_states"] = hidden_states
        else:
            args = (hidden_states, *args[1:])

        return args, kwargs

    def tome_post_forward_hook(module, args, kwargs, output):
        module_id = id(module)
        nonlocal merge_fns, unmerge_fns, num_tokens_after_mergings

        if isinstance(output, torch.Tensor):
            hidden_states = output
        elif isinstance(output, tuple):
            hidden_states = output[0]
        else:
            raise ValueError("Output must be a tensor or a tuple")

        bsz, seqlen, dim = hidden_states.shape

        # Retrieve merge and unmerge functions
        merge_fn = merge_fns.get(module_id)
        unmerge_fn = unmerge_fns.get(module_id)
        num_tokens_after_merging = num_tokens_after_mergings.get(module_id)

        if merge_fn is None or unmerge_fn is None or num_tokens_after_merging is None:
            raise ValueError("Merge and unmerge functions not found for module")

        # Calculate the number of vision tokens after merging
        num_vision_tokens_after_merging = num_tokens_after_merging

        # Split encoder and vision tokens
        encoder_hidden_states = hidden_states[:, :seqlen - num_vision_tokens_after_merging, :]
        vision_hidden_states = hidden_states[:, seqlen - num_vision_tokens_after_merging:, :]

        # Apply unmerge function
        vision_hidden_states = unmerge_fn(vision_hidden_states)

        # Concatenate back
        hidden_states = torch.cat([encoder_hidden_states, vision_hidden_states], dim=1)

        if isinstance(output, torch.Tensor):
            return hidden_states
        elif isinstance(output, tuple):
            return (hidden_states, *output[1:])

    return tome_pre_forward_hook, tome_post_forward_hook


def get_cogvideox_pipeline_with_tome_attnbin(repo_id="THUDM/CogVideoX-5b", tome_modele_names: List[str]=None, hook_kwargs={}):
    
    if tome_modele_names is None:
        raise ValueError("tome_model_names must be provided")
        
    pipe = get_cogvideox_pipeline(repo_id)

    pre_forward_hook, post_forward_hook = _get_cogvideox_tome_token_merging_hooks_attnbin(pipe, **hook_kwargs)
    # pre_forward_hook, post_forward_hook = _get_cogvideox_tome_token_merging_hooks_attn(pipe)
    # pre_forward_hook, post_forward_hook = _get_cogvideox_tome_token_merging_hooks_3d(pipe)
    _wrap_token_merging(pipe.transformer, tome_modele_names, pre_forward_hook, post_forward_hook)    
    return pipe



def get_cogvideox_pipeline_with_tome_3d(repo_id="THUDM/CogVideoX-5b", tome_modele_names: List[str]=None, hook_kwargs={}):
    if tome_modele_names is None:
        raise ValueError("tome_model_names must be provided")
        
    pipe = get_cogvideox_pipeline(repo_id)

    pre_forward_hook, post_forward_hook = _get_cogvideox_tome_token_merging_hooks_3d(pipe, **hook_kwargs)
    # pre_forward_hook, post_forward_hook = _get_cogvideox_tome_token_merging_hooks_attn(pipe)
    # pre_forward_hook, post_forward_hook = _get_cogvideox_tome_token_merging_hooks_3d(pipe)
    _wrap_token_merging(pipe.transformer, tome_modele_names, pre_forward_hook, post_forward_hook)    
    return pipe

