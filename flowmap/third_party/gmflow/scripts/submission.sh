#!/usr/bin/env bash


# generate prediction results for submission on sintel and kitti online servers


# GMFlow without refinement

# submission to sintel
CUDA_VISIBLE_DEVICES=0 python main.py \
--submission \
--output_path submission/sintel-gmflow-norefine \
--val_dataset sintel \
--resume pretrained/gmflow_sintel-0c07dcb3.pth

# submission to kitti
CUDA_VISIBLE_DEVICES=0 python main.py \
--submission \
--output_path submission/kitti-gmflow-norefine \
--val_dataset kitti \
--resume pretrained/gmflow_kitti-285701a8.pth


# you can also visualize the predictions before submission
# CUDA_VISIBLE_DEVICES=0 python main.py \
# --submission \
# --output_path submission/sintel-gmflow-norefine-vis \
# --save_vis_flow \
# --no_save_flo \
# --val_dataset sintel \
# --resume pretrained/gmflow_sintel.pth




# GMFlow with refinement

# submission to sintel
CUDA_VISIBLE_DEVICES=0 python main.py \
--submission \
--output_path submission/sintel-gmflow-withrefine \
--val_dataset sintel \
--resume pretrained/gmflow_with_refine_sintel-3ed1cf48.pth \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1

# submission to kitti
CUDA_VISIBLE_DEVICES=0 python main.py \
--submission \
--output_path submission/kitti-gmflow-withrefine \
--val_dataset kitti \
--resume pretrained/gmflow_with_refine_kitti-8d3b9786.pth \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1





