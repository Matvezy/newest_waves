python scripts/detect_fixed.py --vq-ckpt ./vq_model/vq_ds16_c2i.pt --image-size 256 --image-size-eval 256 --gamma 0.25 --delta 2.0 --seeding_scheme simple_1 --sample_dir ./data/attacked/imagenet/real_1_resizedcrop_0.05

python scripts/detect_fixed.py --vq-ckpt ./vq_model/vq_ds16_c2i.pt --image-size 256 --image-size-eval 256 --gamma 0.25 --delta 2.0 --seeding_scheme simple_1 --sample_dir ./data/attacked/imagenet/real_1_resizedcrop_0.1

python scripts/detect_fixed.py --vq-ckpt ./vq_model/vq_ds16_c2i.pt --image-size 256 --image-size-eval 256 --gamma 0.25 --delta 2.0 --seeding_scheme simple_1 --sample_dir ./data/attacked/imagenet/real_1_resizedcrop_0.2

python scripts/token_frequencies.py --vq-ckpt ./vq_model/vq_ds16_c2i.pt --image-size 256 --image-size-eval 256 --gamma 0.25 --delta 2.0 --seeding_scheme simple_1 --sample_dir ./data/attacked/imagenet/fixed-2-real

"""
python scripts/detect_llamagen.py --vq-ckpt ./vq_model/vq_ds16_c2i.pt --image-size 256 --image-size-eval 256 --gamma 0.25 --delta 2.0 --seeding_scheme simple_1 --sample_dir ./data/attacked/imagenet/none-2-real
"""
"""
python scripts/detect_llamagen.py --vq-ckpt ./vq_model/vq_ds16_c2i.pt --image-size 256 --image-size-eval 256 --gamma 0.25 --delta 2.0 --seeding_scheme simple_1 --sample_dir ./data/attacked/imagenet/real_resizedcrop

python scripts/detect_llamagen.py --vq-ckpt ./vq_model/vq_ds16_c2i.pt --image-size 256 --image-size-eval 256 --gamma 0.25 --delta 2.0 --seeding_scheme simple_1 --sample_dir ./data/main/imagenet/real

python scripts/detect_llamagen.py --vq-ckpt ./vq_model/vq_ds16_c2i.pt --image-size 256 --image-size-eval 256 --gamma 0.25 --delta 2.0 --seeding_scheme simple_1 --sample_dir ./data/attacked/imagenet/real_resizedcrop_0.05

python scripts/detect_llamagen.py --vq-ckpt ./vq_model/vq_ds16_c2i.pt --image-size 256 --image-size-eval 256 --gamma 0.25 --delta 2.0 --seeding_scheme simple_1 --sample_dir ./data/attacked/imagenet/real_resizedcrop_0.1

python scripts/detect_llamagen.py --vq-ckpt ./vq_model/vq_ds16_c2i.pt --image-size 256 --image-size-eval 256 --gamma 0.25 --delta 2.0 --seeding_scheme simple_1 --sample_dir ./data/attacked/imagenet/real_resizedcrop_0.2

python scripts/detect_llamagen.py --vq-ckpt ./vq_model/vq_ds16_c2i.pt --image-size 256 --image-size-eval 256 --gamma 0.25 --delta 2.0 --seeding_scheme simple_1 --sample_dir ./data/attacked/imagenet/none-2-real_resizedcrop_0.05

python scripts/detect_llamagen.py --vq-ckpt ./vq_model/vq_ds16_c2i.pt --image-size 256 --image-size-eval 256 --gamma 0.25 --delta 2.0 --seeding_scheme simple_1 --sample_dir ./data/attacked/imagenet/none-2-real_resizedcrop_0.1

python scripts/detect_llamagen.py --vq-ckpt ./vq_model/vq_ds16_c2i.pt --image-size 256 --image-size-eval 256 --gamma 0.25 --delta 2.0 --seeding_scheme simple_1 --sample_dir ./data/attacked/imagenet/none-2-real_resizedcrop_0.2

python scripts/detect_llamagen.py --vq-ckpt ./vq_model/vq_ds16_c2i.pt --image-size 256 --image-size-eval 256 --gamma 0.25 --delta 2.0 --seeding_scheme simple_1 --sample_dir ./data/attacked/imagenet/resnet18_pwm-2-real

python scripts/detect_llamagen.py --vq-ckpt ./vq_model/vq_ds16_c2i.pt --image-size 256 --image-size-eval 256 --gamma 0.25 --delta 2.0 --seeding_scheme simple_1 --sample_dir ./data/attacked/imagenet/none-2-real_resizedcrop

python scripts/detect_llamagen.py --vq-ckpt ./vq_model/vq_ds16_c2i.pt --image-size 256 --image-size-eval 256 --gamma 0.25 --delta 2.0 --seeding_scheme simple_1 --sample_dir ./data/attacked/imagenet/resnet18-2-real
"""