for ABLATION in \
    ablation_none \
    ablation_explicit_depth \
    ablation_explicit_focal_length \
    ablation_explicit_pose \
    ablation_single_stage \
    ablation_random_initialization_long
do
    python3 -m scripts.run_slurm python3 -m flowmap.overfit \
        wandb.mode=disabled \
        dataset=colmap \
        dataset.colmap.root=results/flowmap_input/co3d_hydrant \
        local_save_root=results/flowmap_${ABLATION}/co3d_hydrant \
        +experiment=[${ABLATION},dump_ate] \
        visualizer.trajectory.ate_save_path=results/ates/${ABLATION}.json
done