import hydra
import logging
from omegaconf import DictConfig, OmegaConf

from typing import List
from PIL import Image
import os.path as osp

import torch
from transformers import CLIPModel, CLIPProcessor
from accediff import LOCAL_PATH
from accediff.evaluation.utils import load_generated_images, read_prompts_from_json
logger = logging.getLogger(__name__)


def calculate_clip_score(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    images_pil: List[Image.Image],
    prompts: List[str],
    device: torch.device,
) -> float:
    """
    Calculates the average CLIPScore between images and their corresponding prompts.

    Args:
        clip_model (CLIPModel): Loaded CLIP model.
        clip_processor (CLIPProcessor): Corresponding CLIP processor.
        images_pil (List[PIL.Image.Image]): List of PIL images.
        prompts (List[str]): List of text prompts.
        device (torch.device): Device to perform computations on.

    Returns:
        average_clip_score (float): The average CLIPScore.
    """
    # Filter out None images and corresponding texts
    valid_pairs = [(img, text) for img, text in zip(images_pil, prompts) if img is not None]

    if not valid_pairs:
        return 0.0

    images, texts = zip(*valid_pairs)

    # Preprocess images and texts
    inputs = clip_processor(
        text=list(texts),
        images=list(images),
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        # Extract image and text features
        image_features = clip_model.get_image_features(pixel_values=inputs.pixel_values)
        text_features = clip_model.get_text_features(
            input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
        )

    # Normalize the feature vectors
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    # Compute cosine similarity and scale by 100
    similarities = (image_features * text_features).sum(dim=-1) * 100
    scores = similarities.cpu().numpy()

    # Calculate the average CLIPScore
    average_clip_score = scores.mean().item()

    return average_clip_score

def main_evaluate(cfg: DictConfig):
# Set the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generation_output_dir = cfg.generation_output_dir

    # ------------------------------------
    # Step 1: Read the JSON file
    # ------------------------------------
    json_file = osp.join( generation_output_dir, "generation_info.json")  # Replace with your JSON file path
    prompts, image_filenames = read_prompts_from_json(json_file)
    logger.info(f"Total prompts read: {len(prompts)}")

    # ------------------------------------
    # Step 2: Load generated images
    # ------------------------------------
    generated_images_folder = osp.join(generation_output_dir, 'generated_images')  # Replace with your generated images folder path
    generated_images_pil, generated_images_tensor = load_generated_images(generated_images_folder, image_filenames)
    logger.info(f"Total generated images loaded: {len(generated_images_pil)}")

    # ------------------------------------
    # Step 3: Calculate CLIP Score
    # ------------------------------------
    logger.info("Calculating CLIP Score...")
    clip_model_name = 'openai/clip-vit-base-patch32'
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    clip_score_value = calculate_clip_score(clip_model, clip_processor, generated_images_pil, prompts, device)
    logger.info(f"Average CLIP Score: {clip_score_value}")
    logger.info("Evaluation completed.")



@hydra.main(version_base=None, config_path=f"{LOCAL_PATH}/configs/evaluation/evaluate_image_clip_score", config_name="config")
def main(cfg: DictConfig):
    logger.info(str(OmegaConf.to_yaml(cfg)))
    main_evaluate(cfg)


if __name__ == "__main__":
    main()