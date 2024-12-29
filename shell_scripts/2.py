import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics

import torch

from optim_utils import *
from io_utils import *
from pytorch_fid.fid_score import *

def main():

    #num_iters = (args.end - args.start) // args.bs
    num_iters = 1
    counter = 0

    for i in range(num_iters):
        #seed = i + args.gen_seed
        rank = 0
        device = rank % torch.cuda.device_count()
        seed = 0 * 1 + rank
        ### generation
        # generation without watermarking
        print(seed)
        #set_random_seed(seed)
        #init_latents_no_w = torch.randn(*shape, device=device)
        torch.manual_seed(seed)
        model_kwargs = {}
        #if args.class_cond:
        classes = torch.randint(
            low=0, high=1000, size=(32,), device=device
        )
        print(classes)
        model_kwargs["y"] = classes

    rank = 0
    device = rank % torch.cuda.device_count()
    seed = 0 * 1 + rank
    torch.manual_seed(seed)
    #set_random_seed(seed)
    print(seed)
    for _ in range(1):
        # Sample inputs:
        c_indices = torch.randint(0, 1000, (32,), device=device)
        print(c_indices)

if __name__ == "__main__":
    main()