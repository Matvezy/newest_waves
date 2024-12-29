python scripts/run_stat_fixed.py \
  --vq-model VQ-16 \
  --vq-ckpt /home/ubuntu/verb-workspace/LlamaGen/vq_ds16_c2i.pt \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --image-size 256 \
  --image-folder ./data/main/imagenet/real \
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
  --image-folder ./data/attacked/imagenet/real_resizedcrop_0.05 \
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
  --image-folder ./data/attacked/imagenet/real_resizedcrop_0.1 \
  --greenlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-9/greenlist.pt \
  --redlist-file /home/ubuntu/verb-workspace/LlamaGen/samples/fixed-simple_1-GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0-splitseed-9/redlist.pt \
  --batch-size 32 \
  --global-seed 0