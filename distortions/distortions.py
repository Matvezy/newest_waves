import random
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
import torch
import io
import sys
import os
# Get the current working directory
current_dir = os.getcwd()

# Add it to sys.path
sys.path.insert(0, current_dir)
from utils import set_random_seed, to_tensor, to_pil


distortion_strength_paras = dict(
    rotation=(0, 45),
    resizedcrop=(1, 0.5),
    erasing=(0, 0.25),
    brightness=(1, 2),
    contrast=(1, 2),
    blurring=(0, 20),
    noise=(0, 0.1),
    compression=(90, 10),
)


def relative_strength_to_absolute(strength, distortion_type):
    assert 0 <= strength <= 1
    strength = (
        strength
        * (
            distortion_strength_paras[distortion_type][1]
            - distortion_strength_paras[distortion_type][0]
        )
        + distortion_strength_paras[distortion_type][0]
    )
    strength = max(strength, min(*distortion_strength_paras[distortion_type]))
    strength = min(strength, max(*distortion_strength_paras[distortion_type]))
    return strength


def apply_distortion(
    images,
    distortion_type,
    strength=None,
    distortion_seed=0,
    same_operation=False,
    relative_strength=True,
    return_image=True,
):
    # Convert images to PIL images if they are tensors
    if not isinstance(images[0], Image.Image):
        images = to_pil(images)
    # Check if strength is relative and convert if needed
    if relative_strength:
        strength = relative_strength_to_absolute(strength, distortion_type)
    # Apply distortions
    distorted_images = []
    seed = distortion_seed
    for image in images:
        distorted_images.append(
            apply_single_distortion(
                image, distortion_type, strength, distortion_seed=seed
            )
        )
        # If not applying the same distortion, increment the seed
        if not same_operation:
            seed += 1
    # Convert to tensors if needed
    if not return_image:
        distorted_images = to_tensor(distorted_images)
    return distorted_images


def apply_single_distortion(image, distortion_type, strength=None, distortion_seed=0):
    # Accept a single image
    assert isinstance(image, Image.Image)
    # Set the random seed for the distortion if given
    set_random_seed(distortion_seed)
    # Assert distortion type is valid
    assert distortion_type in distortion_strength_paras.keys()
    # Assert strength is in the correct range
    if strength is not None:
        assert (
            min(*distortion_strength_paras[distortion_type])
            <= strength
            <= max(*distortion_strength_paras[distortion_type])
        )

    # Apply the distortion
    if distortion_type == "rotation":
        angle = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["rotation"])
        )
        distorted_image = F.rotate(image, angle)

    elif distortion_type == "resizedcrop":
        scale = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["resizedcrop"])
        )
        i, j, h, w = T.RandomResizedCrop.get_params(
            image, scale=(scale, scale), ratio=(1, 1)
        )
        distorted_image = F.resized_crop(image, i, j, h, w, image.size)

    elif distortion_type == "erasing":
        scale = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["erasing"])
        )
        image = to_tensor([image], norm_type=None)
        i, j, h, w, v = T.RandomErasing.get_params(
            image, scale=(scale, scale), ratio=(1, 1), value=[0]
        )
        distorted_image = F.erase(image, i, j, h, w, v)
        distorted_image = to_pil(distorted_image, norm_type=None)[0]

    elif distortion_type == "brightness":
        factor = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["brightness"])
        )
        enhancer = ImageEnhance.Brightness(image)
        distorted_image = enhancer.enhance(factor)

    elif distortion_type == "contrast":
        factor = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["contrast"])
        )
        enhancer = ImageEnhance.Contrast(image)
        distorted_image = enhancer.enhance(factor)

    elif distortion_type == "blurring":
        kernel_size = (
            int(strength)
            if strength is not None
            else random.uniform(*distortion_strength_paras["blurring"])
        )
        distorted_image = image.filter(ImageFilter.GaussianBlur(kernel_size))

    elif distortion_type == "noise":
        std = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["noise"])
        )
        image = to_tensor([image], norm_type=None)
        noise = torch.randn(image.size()) * std
        distorted_image = to_pil((image + noise).clamp(0, 1), norm_type=None)[0]

    elif distortion_type == "compression":
        quality = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["compression"])
        )
        quality = int(quality)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=quality)
        distorted_image = Image.open(buffered)

    else:
        assert False

    return distorted_image

if __name__ == "__main__":
    import os
    from PIL import Image
    #"/home/ubuntu/verb-workspace/WAVES/data/main/imagenet/real_1",
    # Define paths
    img_paths = ["/home/ubuntu/verb-workspace/LlamaGen/samples/GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-dist_free"]
    """
    "/home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-0",
    "/home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-1",
        "/home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-2",
        "/home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-3",
        "/home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-4",
        "/home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-5",
        "/home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-6",
        "/home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-7",
        "/home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-8",
        "/home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-9",
    """

    # Define the main attacked directory
    attacked_base_dir = "/home/ubuntu/verb-workspace/WAVES/data/attacked/imagenet"

    # List of distortion strengths
    distortion_strengths = [0.05, 0.1]

    # Function to apply distortions and save images
    def save_distorted_images(img_paths, distortion_type, strengths):
        for p in img_paths:
            for strength in strengths:
                # Create output directory based on folder name, distortion type, and strength
                folder_name = os.path.basename(p) + f"_{distortion_type}_{strength}"
                output_dir = os.path.join(attacked_base_dir, folder_name)
                
                # Ensure the output directory exists
                os.makedirs(output_dir, exist_ok=True)
                
                # Load images
                images_p = [Image.open(os.path.join(p, i)) for i in os.listdir(p) if i.endswith(".png")]
                
                # Apply distortion
                distorted_images = apply_distortion(images_p, distortion_type, strength)
                
                # Save each distorted image
                for idx, img in enumerate(distorted_images):
                    save_path = os.path.join(output_dir, f"distorted_{idx}_{distortion_type}_{strength}.png")
                    img.save(save_path)
                    #print(f"Image saved to {save_path}")

    # Apply the function with multiple strengths
    save_distorted_images(img_paths, "resizedcrop", distortion_strengths)
