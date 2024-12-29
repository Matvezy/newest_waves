import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
import os
from PIL import Image
import numpy as np
import argparse
import math
import sys
sys.path.append(os.getcwd())
import csv
import seaborn as sns

from llamagen.tokenizer.tokenizer_image.vq_model import VQ_models
from llamagen.lm_watermarking.fixed_processor import WatermarkLogitsProcessor, WatermarkDetector

# Function to save statistics to CSV
def save_statistics(num_images, average_green_ratio, std_dev_green_ratio, average_red_ratio, std_dev_red_ratio, csv_file):
    # Check if file exists to add header only once
    file_exists = os.path.isfile(csv_file)

    # Open the CSV file in append mode
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header only if the file is new
        if not file_exists:
            writer.writerow(["Total Images", "Average Green Ratio", "Std Dev Green Ratio", "Average Red Ratio", "Std Dev Red Ratio"])
        # Write the row of statistics
        writer.writerow([num_images, average_green_ratio, std_dev_green_ratio, average_red_ratio, std_dev_red_ratio])

def main(args):
    # Ensure CUDA is available
    assert torch.cuda.is_available(), "This script requires a GPU."
    csv_file = "statistics.csv"
    # Set up device
    device = 0  # Use GPU device 0
    torch.set_grad_enabled(False)
    torch.manual_seed(args.global_seed)
    torch.cuda.set_device(device)
    print(f"Using device: {device}")

    # Load VQ-GAN model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim
    ).to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint

    # Load the fixed greenlist
    greenlist_file = args.greenlist_file
    assert os.path.exists(greenlist_file), f"Greenlist file not found at {greenlist_file}"
    greenlist_ids = torch.load(greenlist_file).to(device)
    greenlist_set = set(greenlist_ids.cpu().numpy())

    # Load the fixed redlist
    redlist_file = args.redlist_file
    assert os.path.exists(redlist_file), f"Redlist file not found at {redlist_file}"
    redlist_ids = torch.load(redlist_file).to(device)
    redlist_set = set(redlist_ids.cpu().numpy())

    # Prepare to collect statistics
    green_ratios = []
    red_ratios = []
    total_green_tokens = 0
    total_red_tokens = 0
    total_tokens = 0

    # Read images from the specified folder
    image_folder = args.image_folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    num_images = len(image_files)
    print(f"Found {num_images} images in {image_folder}")

    # Process images in batches
    batch_size = args.batch_size
    for i in range(0, num_images, batch_size):
        batch_files = image_files[i:i + batch_size]
        images = []
        for file_name in batch_files:
            img = Image.open(os.path.join(image_folder, file_name)).convert('RGB')
            img = img.resize((args.image_size, args.image_size), Image.BICUBIC)
            img = np.array(img).astype(np.float32)
            img = (img - 128.0) / 127.5  # Normalize to [-1, 1]
            images.append(img)
        images = np.stack(images)
        images = torch.from_numpy(images).permute(0, 3, 1, 2).to(device)

        # Encode images to code indices
        with torch.no_grad():
            indices = vq_model.encode_to_code(images)
            indices = indices.view(indices.size(0), -1)  # Flatten to (batch_size, num_tokens)

        # Count green and red tokens for each image
        for idx, image_indices in enumerate(indices):
            image_indices_list = image_indices.cpu().numpy()
            num_total_tokens = image_indices.numel()

            # Create masks for green and red tokens
            green_mask = np.isin(image_indices_list, list(greenlist_set))
            red_mask = np.isin(image_indices_list, list(redlist_set))

            num_green_tokens = np.sum(green_mask)
            num_red_tokens = np.sum(red_mask)

            # Sanity check: Ensure that green_tokens + red_tokens == total_tokens
            assert num_green_tokens + num_red_tokens == num_total_tokens, "Mismatch in token counts."

            # Update statistics
            green_ratio = num_green_tokens / num_total_tokens
            red_ratio = num_red_tokens / num_total_tokens

            green_ratios.append(green_ratio)
            red_ratios.append(red_ratio)
            total_green_tokens += num_green_tokens
            total_red_tokens += num_red_tokens
            total_tokens += num_total_tokens

            # Optionally, print per-image statistics
            # print(f"Image: {batch_files[idx]}")
            # print(f"  Total tokens: {num_total_tokens}")
            # print(f"  Green tokens: {num_green_tokens}")
            # print(f"  Red tokens: {num_red_tokens}")
            # print(f"  Green ratio: {green_ratio:.4f}")
            # print(f"  Red ratio: {red_ratio:.4f}")

    # Compute overall statistics
    average_green_ratio = np.mean(green_ratios)
    std_dev_green_ratio = np.std(green_ratios)
    average_red_ratio = np.mean(red_ratios)
    std_dev_red_ratio = np.std(red_ratios)

    print("\n--- Overall Statistics ---")
    print(f"Total images processed: {num_images}")
    print(f"Average green token ratio: {average_green_ratio:.4f}")
    print(f"Standard deviation of green token ratio: {std_dev_green_ratio:.4f}")
    print(f"Average red token ratio: {average_red_ratio:.4f}")
    print(f"Standard deviation of red token ratio: {std_dev_red_ratio:.4f}")

    #save_statistics(num_images, average_green_ratio, std_dev_green_ratio, average_red_ratio, std_dev_red_ratio, csv_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, required=True, help="Checkpoint path for VQ-GAN model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="Codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="Codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, default=256, help="Image size (images will be resized to this size)")
    parser.add_argument("--image-folder", type=str, required=True, help="Folder containing images to process")
    parser.add_argument("--greenlist-file", type=str, required=True, help="Path to the greenlist.pt file")
    parser.add_argument("--redlist-file", type=str, required=True, help="Path to the redlist.pt file")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing images")
    parser.add_argument("--global-seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    main(args)
