#!/usr/bin/env bash

# inference GMFlow without refinement

# sintel

# only predict forward flow
CUDA_VISIBLE_DEVICES=0 python main.py \
--inference_dir demo/sintel_market_1 \
--output_path output/gmflow-norefine-sintel_market_1 \
--resume pretrained/gmflow_sintel-0c07dcb3.pth

# predict forward & backward flow
CUDA_VISIBLE_DEVICES=0 python main.py \
--inference_dir demo/sintel_market_1 \
--output_path output/gmflow-norefine-sintel_market_1 \
--pred_bidir_flow \
--resume pretrained/gmflow_sintel-0c07dcb3.pth


# predict forward & backward flow with forward-backward consistency check
CUDA_VISIBLE_DEVICES=0 python main.py \
--inference_dir demo/sintel_market_1 \
--output_path output/gmflow-norefine-sintel_market_1 \
--pred_bidir_flow \
--fwd_bwd_consistency_check \
--resume pretrained/gmflow_sintel-0c07dcb3.pth


# davis

CUDA_VISIBLE_DEVICES=0 python main.py \
--inference_dir demo/davis_breakdance-flare \
--output_path output/gmflow-norefine-davis_breakdance-flare \
--resume pretrained/gmflow_sintel-0c07dcb3.pth




# inference GMFlow with refinement

CUDA_VISIBLE_DEVICES=0 python main.py \
--inference_dir demo/davis_breakdance-flare \
--output_path output/gmflow-withrefine-davis_breakdance-flare \
--resume pretrained/gmflow_with_refine_sintel-3ed1cf48.pth \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1




CUDA_VISIBLE_DEVICES=0 python main.py \
--inference_dir demo/sintel_test_clean_market_1 \
--output_path output/gmflow-norefine-sintel_test_clean_market_1 \
--pred_bidir_flow \
--fwd_bwd_consistency_check \
--resume pretrained/gmflow_sintel-0c07dcb3.pth


