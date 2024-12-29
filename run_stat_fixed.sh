python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-9 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-9/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-9/redlist.pt \
  --batch-size 32 \
  --global-seed 0

python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder ./data/attacked/imagenet/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-9_resizedcrop_0.05 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-9/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-9/redlist.pt \
  --batch-size 32 \
  --global-seed 0

python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder ./data/attacked/imagenet/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-9_resizedcrop_0.1 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-9/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-9/redlist.pt \
  --batch-size 32 \
  --global-seed 0


python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-8 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-8/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-8/redlist.pt \
  --batch-size 32 \
  --global-seed 0

python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder ./data/attacked/imagenet/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-8_resizedcrop_0.05 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-8/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-8/redlist.pt \
  --batch-size 32 \
  --global-seed 0

python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder ./data/attacked/imagenet/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-8_resizedcrop_0.1 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-8/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-8/redlist.pt \
  --batch-size 32 \
  --global-seed 0  


python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-7 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-7/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-7/redlist.pt \
  --batch-size 32 \
  --global-seed 0

python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder ./data/attacked/imagenet/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-7_resizedcrop_0.05 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-7/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-7/redlist.pt \
  --batch-size 32 \
  --global-seed 0

python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder ./data/attacked/imagenet/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-7_resizedcrop_0.1 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-7/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-7/redlist.pt \
  --batch-size 32 \
  --global-seed 0 


python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-6 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-6/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-6/redlist.pt \
  --batch-size 32 \
  --global-seed 0

python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder ./data/attacked/imagenet/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-6_resizedcrop_0.05 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-6/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-6/redlist.pt \
  --batch-size 32 \
  --global-seed 0

python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder ./data/attacked/imagenet/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-6_resizedcrop_0.1 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-6/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-6/redlist.pt \
  --batch-size 32 \
  --global-seed 0 


python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-5 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-5/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-5/redlist.pt \
  --batch-size 32 \
  --global-seed 0

python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder ./data/attacked/imagenet/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-5_resizedcrop_0.05 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-5/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-5/redlist.pt \
  --batch-size 32 \
  --global-seed 0

python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder ./data/attacked/imagenet/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-5_resizedcrop_0.1 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-5/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-5/redlist.pt \
  --batch-size 32 \
  --global-seed 0 


python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-4 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-4/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-4/redlist.pt \
  --batch-size 32 \
  --global-seed 0

python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder ./data/attacked/imagenet/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-4_resizedcrop_0.05 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-4/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-4/redlist.pt \
  --batch-size 32 \
  --global-seed 0

python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder ./data/attacked/imagenet/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-4_resizedcrop_0.1 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-4/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-4/redlist.pt \
  --batch-size 32 \
  --global-seed 0 


python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-3 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-3/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-3/redlist.pt \
  --batch-size 32 \
  --global-seed 0

python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder ./data/attacked/imagenet/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-3_resizedcrop_0.05 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-3/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-3/redlist.pt \
  --batch-size 32 \
  --global-seed 0

python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder ./data/attacked/imagenet/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-3_resizedcrop_0.1 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-3/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-3/redlist.pt \
  --batch-size 32 \
  --global-seed 0 


python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-2 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-2/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-2/redlist.pt \
  --batch-size 32 \
  --global-seed 0

python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder ./data/attacked/imagenet/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-2_resizedcrop_0.05 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-2/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-2/redlist.pt \
  --batch-size 32 \
  --global-seed 0

python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder ./data/attacked/imagenet/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-2_resizedcrop_0.1 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-2/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-2/redlist.pt \
  --batch-size 32 \
  --global-seed 0 


python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-1 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-1/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-1/redlist.pt \
  --batch-size 32 \
  --global-seed 0

python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder ./data/attacked/imagenet/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-1_resizedcrop_0.05 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-1/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-1/redlist.pt \
  --batch-size 32 \
  --global-seed 0

python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder ./data/attacked/imagenet/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-1_resizedcrop_0.1 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-1/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-1/redlist.pt \
  --batch-size 32 \
  --global-seed 0 


python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-0 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-0/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-0/redlist.pt \
  --batch-size 32 \
  --global-seed 0

python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder ./data/attacked/imagenet/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-0_resizedcrop_0.05 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-0/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-0/redlist.pt \
  --batch-size 32 \
  --global-seed 0

python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder ./data/attacked/imagenet/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-0_resizedcrop_0.1 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-0/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-0/redlist.pt \
  --batch-size 32 \
  --global-seed 0 