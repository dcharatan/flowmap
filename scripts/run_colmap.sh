for SCENE in /scratch/datasets/flowmap/*/ ; do
    export JOB_NAME=colmap_$(basename ${SCENE})
    python3 -m scripts.run_slurm python3 -m scripts.colmap.run_both_at_flowmap_resolution \
        /scratch/datasets/flowmap/$(basename ${SCENE}) \
        results/colmap/$(basename ${SCENE}) \
        results/mvscolmap/$(basename ${SCENE}) \
        results/workspace/$(basename ${SCENE})
done
