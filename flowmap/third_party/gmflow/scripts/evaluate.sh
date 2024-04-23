#!/usr/bin/env bash

# evaluate GMFlow without refinement

# evaluate chairs & things trained model on things and sintel (Table 3 of GMFlow paper)
# the output should be:
# Number of validation image pairs: 1024
# Validation Things test set (things_clean) EPE: 3.475
# Validation Things test (things_clean) s0_10: 0.666, s10_40: 1.310, s40+: 8.968
# Number of validation image pairs: 1041
# Validation Sintel (clean) EPE: 1.495, 1px: 0.161, 3px: 0.059, 5px: 0.040
# Validation Sintel (clean) s0_10: 0.457, s10_40: 1.770, s40+: 8.257
# Number of validation image pairs: 1041
# Validation Sintel (final) EPE: 2.955, 1px: 0.209, 3px: 0.098, 5px: 0.071
# Validation Sintel (final) s0_10: 0.725, s10_40: 3.446, s40+: 17.701

CUDA_VISIBLE_DEVICES=0 python main.py \
--eval \
--resume pretrained/gmflow_things-e9887eda.pth \
--val_dataset things sintel \
--with_speed_metric



# evaluate GMFlow with refinement

# evaluate chairs & things trained model on things and sintel (Table 3 of GMFlow paper)
# the output should be:
# Validation Things test set (things_clean) EPE: 2.804
# Validation Things test (things_clean) s0_10: 0.527, s10_40: 1.009, s40+: 7.314
# Number of validation image pairs: 1041
# Validation Sintel (clean) EPE: 1.084, 1px: 0.092, 3px: 0.040, 5px: 0.028
# Validation Sintel (clean) s0_10: 0.303, s10_40: 1.252, s40+: 6.261
# Number of validation image pairs: 1041
# Validation Sintel (final) EPE: 2.475, 1px: 0.147, 3px: 0.077, 5px: 0.058
# Validation Sintel (final) s0_10: 0.511, s10_40: 2.810, s40+: 15.669

CUDA_VISIBLE_DEVICES=0 python main.py \
--eval \
--resume pretrained/gmflow_with_refine_things-36579974.pth \
--val_dataset things sintel \
--with_speed_metric \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1



# evaluate matched & matched on sintel

# evaluate GMFlow without refinement

CUDA_VISIBLE_DEVICES=0 python main.py \
--eval \
--evaluate_matched_unmatched \
--resume pretrained/gmflow_things-e9887eda.pth \
--val_dataset sintel

# evaluate GMFlow with refinement

CUDA_VISIBLE_DEVICES=0 python main.py \
--eval \
--evaluate_matched_unmatched \
--resume pretrained/gmflow_with_refine_things-36579974.pth \
--val_dataset sintel \
--with_speed_metric \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1








