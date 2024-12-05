import hydra
import logging
from omegaconf import DictConfig, OmegaConf

from typing import List
from PIL import Image
import os.path as osp

from torchvision import transforms
import torch
from accediff import LOCAL_PATH
from torchmetrics.image.fid import FrechetInceptionDistance
from typing import List
from accediff.evaluation.utils import load_generated_images, read_prompts_from_json
logger = logging.getLogger(__name__)


def calculate_fid(real_images_tensor, generated_images_tensor, device):
    """
    Calculates the FID score between real and generated images.

    Args:
        real_images_tensor (torch.Tensor): Tensor of real images, shape [N, C, H, W], values in [0, 1].
        generated_images_tensor (torch.Tensor): Tensor of generated images, shape [N, C, H, W], values in [0, 1].
        device (torch.device): Device to perform computations on.

    Returns:
        fid_score (float): The calculated FID score.
    """
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    fid_metric.reset()
    fid_metric.update(real_images_tensor.to(device), real=True)
    fid_metric.update(generated_images_tensor.to(device), real=False)
    fid_score = fid_metric.compute().item()
    return fid_score


def preprocess_images_for_fid(images_pil):
    """
    Preprocesses PIL images for FID calculation.

    Args:
        images_pil (List[PIL.Image.Image]): List of PIL images.

    Returns:
        images_tensor (torch.Tensor): Tensor of images, shape [N, C, H, W].
    """
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    images_tensor = []
    for img in images_pil:
        if img is not None:
            img_tensor = preprocess(img)
            images_tensor.append(img_tensor)
        else:
            # Create a zero tensor if image is missing
            images_tensor.append(torch.zeros(3, 299, 299))

    images_tensor = torch.stack(images_tensor)
    return images_tensor

def main_evaluate(cfg: DictConfig):
# Set the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generation_output_dir = cfg.generation_output_dir
    # # ------------------------------------
    # # Step 1: Load GT json
    # # ------------------------------------
    # gt_json_file = osp.join( generation_output_dir, "output_prompts.json")  # Replace with your JSON file path
    
    gt_json_file = cfg.gt_json_file
    gt_prompts, gt_image_filenames = read_prompts_from_json(gt_json_file)
    logger.info(f"Total gt prompts read: {len(gt_prompts)}")

    # ------------------------------------
    # Step 2: Read the JSON file
    # ------------------------------------
    # gt_prompt_to_filename = {prompt: filename for prompt, filename in zip(gt_prompts, gt_image_filenames)}
    # filename_to_prompt = {filename: prompt for prompt, filename in zip(prompts, image_filenames)}
    # gt_image_filenames = [gt_prompt_to_filename[filename_to_prompt[filename]] for filename in image_filenames]


    json_file = osp.join( generation_output_dir, "generation_info.json")  # Replace with your JSON file path
    prompts, image_filenames = read_prompts_from_json(json_file)
    logger.info(f"Total prompts read: {len(prompts)}")

    # ------------------------------------
    # Step 3: Load images
    # ------------------------------------
    common_prompts = set(gt_prompts) and set(prompts)

    prompts = list(common_prompts)
    assert len(prompts) > 0, "No common prompts found between GT and generated images"
    logger.info(f"Common prompts for FID: {len(prompts)}")
    logger.info(f"unique prompt in GT: {set(gt_prompts) - common_prompts}")
    logger.info(f"unique prompt in generated: {set(prompts) - common_prompts}")

    prompt2filename = {prompt: filename for prompt, filename in zip(prompts, image_filenames)}
    gt_prompt2filename = {prompt: filename for prompt, filename in zip(gt_prompts, gt_image_filenames)}
    image_filenames = [prompt2filename[prompt] for prompt in prompts]
    gt_image_filenames = [gt_prompt2filename[prompt] for prompt in prompts]
    
    # gt_images_folder = osp.join(f"{LOCAL_PATH}/RESULTS/coco", 'output_images')  # Replace with your GT images folder path
    gt_images_folder = cfg.gt_images_folder
    gt_images_pil, gt_images_tensor = load_generated_images(gt_images_folder, gt_image_filenames)

    generated_images_folder = osp.join(generation_output_dir, 'generated_images')  # Replace with your generated images folder path
    generated_images_pil, generated_images_tensor = load_generated_images(generated_images_folder, image_filenames)

    # # ------------------------------------
    # # Step 4: Calculate FID
    # # ------------------------------------

    generated_images_tensor = preprocess_images_for_fid(generated_images_pil)
    gt_images_tensor = preprocess_images_for_fid(gt_images_pil)


    logger.info("Calculating FID...")
    fid_score = calculate_fid(gt_images_tensor, generated_images_tensor, device)
    logger.info(f"FID Score: {fid_score}")

@hydra.main(version_base=None, config_path=f"{LOCAL_PATH}/configs/evaluation/evaluate_image_fid_score", config_name="coco")
def main(cfg: DictConfig):
    logger.info(str(OmegaConf.to_yaml(cfg)))
    main_evaluate(cfg)

if __name__ == "__main__":
    main()