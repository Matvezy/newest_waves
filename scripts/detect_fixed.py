import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
import torch.distributed as dist

from tqdm import tqdm
import sys
import os
sys.path.append(os.getcwd())
from PIL import Image
import numpy as np
import math
import argparse
import shutil

from llamagen.tokenizer.tokenizer_image.vq_model import VQ_models
from llamagen.autoregressive.models.gpt import GPT_models
from llamagen.autoregressive.models.generate import generate

from llamagen.lm_watermarking.fixed_processor import WatermarkLogitsProcessor, WatermarkDetector

def main(args):
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    #dist.init_process_group("nccl")
    #rank = dist.get_rank()
    device = 0 #rank % torch.cuda.device_count()
    seed = args.global_seed #* dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting seed={seed}.") #, rank={rank}, world_size={dist.get_world_size()}.")

    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint

    vocab = vq_model.quantize.embedding.weight.data

    watermark_detector = WatermarkDetector(
        vocab=vocab,
        gamma=args.gamma,
        delta=args.delta,  # Include delta if required
        seeding_scheme=args.seeding_scheme,
        select_green_tokens=True,  # Ensure this matches the processor
        device="cuda",  # Use the same device as the processor
        tokenizer=None,  # If you have a tokenizer, include it here
        z_threshold=4.0,
        normalizers=[],
        ignore_repeated_ngrams=True,
        greenlist_file="/home/ubuntu/verb-workspace/WAVES/data/main/imagenet/real_1/greenlist.pt",
    )

    divergent_examples_dir = "./divergent_examples"
    os.makedirs(divergent_examples_dir, exist_ok=True)

    # Load samples from disk as individual .png files
    samples = []
    image_files = []  # To keep track of the image filenames
    for sample in os.listdir(args.sample_dir):
        if sample.endswith(".png"):
            img = Image.open(f"{args.sample_dir}/{sample}")
            samples.append(np.array(img))
            image_files.append(sample)

    # Convert samples to PyTorch tensors
    samples = torch.from_numpy(np.stack(samples)).to(dtype=torch.float32, device="cpu")

    # Reverse the normalization and permute dimensions
    #samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy() orig
    samples = (samples.permute(0, 3, 1, 2).float() - 128.0) / 127.5
    #or samples = (torch.clamp(samples.permute(0, 3, 1, 2).float(), 0, 255) - 128.0) / 127.5

    # If image size was changed during evaluation, reverse the interpolation
    # If image size was changed during evaluation, reverse the interpolation
    if args.image_size_eval != args.image_size:
        samples = F.interpolate(samples, size=(args.image_size, args.image_size), mode='bicubic')

    print("samples shape", samples.shape)
    #samples_out = []
    #for i in range(samples.size(0)):
    #    sample_item = samples[i].unsqueeze(0)
    #    samples_out.append(vq_model.encode_to_code(sample_item))

    #Do the same but with batches of 32
    samples_out = []
    samples = samples.to(device)
    for i in range(0, samples.size(0), args.per_proc_batch_size):
        print(i)
        sample_batch = samples[i:i + args.per_proc_batch_size]
        out = vq_model.encode_to_code(sample_batch)
        #print(type(out))
        #print(out.shape)
        samples_out.append(out)

    score_dicts = []
    print(len(samples_out))
    for i in range(len(samples_out)):
        print(i)
        for sample_det in samples_out[i]:
            sample_det = sample_det.squeeze(0).flatten()
            score_dict = watermark_detector.detect(tokenized_text=sample_det)
            #print(score_dict)
            #print(f"Watermark detection score for sample:")
            #print(score_dict)
            score_dicts.append(score_dict)

    # Check if sample_dir matches the specific directory
    if args.sample_dir == "./data/attacked/imagenet/none-2-real":
        # Loop over score_dicts and copy images with prediction == 0
        for idx, score_dict in enumerate(score_dicts):
            if score_dict['prediction'] == 0:
                # Copy the corresponding image from the original folder to divergent_examples
                original_image_path = os.path.join(args.sample_dir, image_files[idx])
                destination_path = os.path.join(divergent_examples_dir, image_files[idx])
                shutil.copy(original_image_path, destination_path)
                print(f"Copied divergent example to {destination_path}")

    print("Watermark detection scores:")
    #print(score_dicts)
    #save the scores to a file
    #with open("watermark_detection_scores.txt", "w") as f:
    #    for score_dict in score_dicts:
    #        f.write(str(score_dict) + "\n")

    #tensor_dict = {'tensor_list': score_dicts}
    #type_img = args.sample_dir.split("/")[1][-1]
    #num_img = args.sample_dir.split("/")[-1].split("_")[-1]
    #torch.save(tensor_dict, f"results_llamagen/"+args.sample_dir.split("/")[-1]+"/results_" + type_img + "_" + num_img + ".pt")
    
    # Assuming tensor_dict and other necessary variables are already defined
    tensor_dict = {'tensor_list': score_dicts}
    type_img = args.sample_dir.split("/")[1][-1]
    num_img = args.sample_dir.split("/")[-1].split("_")[-1]
    
    # Define the directory path
    save_dir = f"results_llamagen/{args.sample_dir.split('/')[-1]}"
    
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the tensor
    torch.save(tensor_dict, os.path.join(save_dir, f"results_{type_img}_{num_img}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=True)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--image-size-eval", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=1024)
    parser.add_argument("--sample_dir", type=str, default="samples")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=0,help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--gamma",  type=float, default=0.25)
    parser.add_argument("--delta", type=float, default=2)
    parser.add_argument("--seeding_scheme", type=str, default="simple_1")
    args = parser.parse_args()
    main(args)