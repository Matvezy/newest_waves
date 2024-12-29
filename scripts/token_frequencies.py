import torch
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import sys
sys.path.append(os.getcwd())
from llamagen.tokenizer.tokenizer_image.vq_model import VQ_models

def compute_token_counts(args):
    # Load the VQ model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to("cuda")
    vq_model.eval()
    
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint

    # Load the greenlist and redlist
    greenlist = torch.load(args.sample_dir + "greenlist.pt")
    redlist = torch.load(args.sample_dir + "redlist.pt")

    # Initialize token counts
    token_counts = torch.zeros(args.codebook_size, dtype=torch.int64, device="cuda")

    # Process each image
    for sample in os.listdir(args.sample_dir):
        if sample.endswith(".png"):
            #torch.cuda.empty_cache()
            img = Image.open(f"{args.sample_dir}/{sample}")
            img_array = np.array(img)
            if img_array.shape[-1] == 4:
                img_array = img_array[:, :, :3]
            sample_tensor = (torch.tensor(img_array).permute(2, 0, 1).float() - 128.0) / 127.5
            sample_tensor = sample_tensor.unsqueeze(0).to("cuda")
            # Tokenize the image
            tokens = vq_model.encode_to_code(sample_tensor).squeeze(0).flatten()
            
            # Update token counts
            unique_tokens, counts = tokens.unique(return_counts=True)
            token_counts[unique_tokens] += counts
            
    # Move token_counts to CPU for further processing
    return token_counts.cpu(), greenlist, redlist

def plot_token_histogram(token_counts, greenlist, redlist, output_path="token_histogram.png"):
    token_ids = torch.arange(len(token_counts))
    counts = token_counts.cpu().numpy()
    greenlist = set(greenlist.cpu().numpy())
    redlist = set(redlist.cpu().numpy())

    # Assign colors based on token id
    colors = ['blue'] * len(token_counts)  # Default to blue
    for i in range(len(token_counts)):
        if i in greenlist:
            colors[i] = 'green'
        elif i in redlist:
            colors[i] = 'red'

    # Determine y-axis limit (99.9th percentile for good scale)
    y_max = np.percentile(counts, 99.9)

    # Create a figure with space on the right for annotations
    fig, ax = plt.subplots(figsize=(16, 8))
    plt.subplots_adjust(right=0.75)  # Leave space on the right for text
    sns.barplot(x=token_ids, y=counts, palette=colors, ax=ax)
    ax.set_ylim(0, y_max + 0.05 * y_max)  # Add a small buffer above the limit

    # Remove x-axis labels
    ax.set_xticks([])  # Remove tick labels entirely

    # Identify outliers
    outliers = [(token_id, count) for token_id, count in enumerate(counts) if count > y_max]

    # Add text to the right of the plot for outliers
    if outliers:
        ax_text = fig.add_axes([0.8, 0.1, 0.2, 0.8])  # Add a new axis for text
        ax_text.axis("off")  # Hide the axis
        outlier_texts = []
        for token_id, count in outliers:
            color = 'green' if token_id in greenlist else 'red' if token_id in redlist else 'blue'
            outlier_texts.append(f"Token {token_id}: {count}")
            ax_text.text(
                0, 1 - 0.05 * len(outlier_texts),  # Position: top-down for each outlier
                f"Token {token_id}: {count}",
                color=color, fontsize=10, va='top'
            )

    # Add labels and title
    ax.set_xlabel("Token ID")  # Optional: Add general label without tick labels
    ax.set_ylabel("Count")
    ax.set_title("VQGAN Token Distribution")
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, required=True, help="Path to VQGAN checkpoint")
    parser.add_argument("--sample-dir", type=str, required=True, help="Directory containing image samples")
    parser.add_argument("--codebook-size", type=int, default=16384, help="Codebook size of VQGAN")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="Embedding dimension of VQGAN")
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()

    token_counts, greenlist, redlist = compute_token_counts(args)
    with torch.no_grad():
        plot_token_histogram(token_counts, greenlist, redlist, output_path = args.out_path)
