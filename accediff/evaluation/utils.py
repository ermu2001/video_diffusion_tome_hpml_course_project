
from PIL import Image
import os
import json
import torch
from torchvision import transforms


def read_prompts_from_json(json_file):
    """
    Reads prompts and corresponding image filenames from a JSON file.

    Args:
        json_file (str): Path to the JSON file.

    Returns:
        prompts (List[str]): List of text prompts.
        image_filenames (List[str]): List of image filenames.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    prompts = []
    image_filenames = []

    for item in data:
        prompts.append(item['prompt'])
        image_filenames.append(item['image_filename'])

    return prompts, image_filenames

def load_generated_images(image_folder, image_filenames):
    """
    Loads generated images from a folder based on filenames.

    Args:
        image_folder (str): Path to the folder containing generated images.
        image_filenames (List[str]): List of image filenames to load.

    Returns:
        images_pil (List[PIL.Image.Image]): List of loaded PIL images.
        images_tensor (torch.Tensor): Tensor of images for FID calculation.
    """
    images_pil = []
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    images_tensor = []

    for filename in image_filenames:
        img_path = os.path.join(image_folder, filename)
        if os.path.isfile(img_path):
            img = Image.open(img_path).convert('RGB')
            images_pil.append(img)
            img_tensor = preprocess(img)
            images_tensor.append(img_tensor)
        else:
            print(f"Warning: Generated image {img_path} not found.")
            images_pil.append(None)
            images_tensor.append(torch.zeros(3, 299, 299))

    images_tensor = torch.stack(images_tensor)
    return images_pil, images_tensor


