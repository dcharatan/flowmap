for ABLATION in \
    ablation_none \
    ablation_no_correspondence_weights \
    ablation_no_tracks \
    ablation_explicit_depth \
    ablation_explicit_focal_length \
    ablation_explicit_pose \
    ablation_random_initialization \
    ablation_random_initialization_long \
    ablation_single_stage
do
    for SCENE in results/flowmap_input/*/ ; do
        python3 -m scripts.run_slurm python3 -m flowmap.overfit \
            wandb.mode=online \
            wandb.tags=[paper_v17_${ABLATION}] \
            wandb.name=paper_v17_${ABLATION}_$(basename ${SCENE}) \
            dataset=colmap \
            dataset.colmap.root=results/flowmap_input/$(basename ${SCENE}) \
            local_save_root=results/flowmap_${ABLATION}/$(basename ${SCENE}) \
            +experiment=${ABLATION}
    done
done