## Official Implementation of ["Content-Style Learning from Unaligned Domains: Identifiability under Unknown Latent Dimensions"](https://arxiv.org/abs/2411.03755), ICLR 2025

## Setup

The code is based on [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch). Please refer to [StyleGAN2-ADA repo](https://github.com/NVlabs/stylegan2-ada-pytorch) for the environment setup. 

## Dataset Preparation

First, download AFHQ or CELEBA-HQ dataset from https://github.com/clovaai/stargan-v2, and run the following with appropriate path to the dataset

```bash
python dataset_tool.py --source=datasets/afhq/train --dest=datasets/afhq.zip --height=256 --width=256
```

## Training Multi-Domain Generative model
Run the following to train the generative model on AFHQ dataset
```bash
python train.py --outdir=logs/and/checkpoint/dir \
                --data=dataset_folder/afhq_v2.zip \
                --gpus=1 \
                --map_type=fixed \
                --num_c_res=5 \
                --i_dim=128 \
                --sparse_weight=0.3 \
                --cfg=mlp3 \
                --style_mixing_prob=0.0 \
                --metrics=pr50k3_full,fid50k_full \
                --cond=true \
                --wandb_run_name=$RUN_NAME;
```
The results and model checkpoints are saved in outdir. Sample images and all training losses will be logged in wandb.

## Multi-Domain Translation using the trained generative model
Using the trained generative model one can run domain translation using the following example command (for translation from cat to dog)
```bash
python translation.py --network $PRETRAINED_NETWORK_PKL_FILE \
        --content $AFHQ_DATASET/test/cat \
        --style $AFHQ_DATASET/test/dog \
        --source_class 0 \
        --target_class 1 \
        --outdir $RESULT_PATH \
        --num-steps 400 \
        --batchsize 16 \
        --psi 1.0 \
        --num_styles_per_content 10;
```
The translated images will be saved in $RESULT_PATH folder.