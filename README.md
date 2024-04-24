# FlowMap

https://github.com/dcharatan/flowmap/assets/13124225/9dc9cc9a-083e-4fd1-b833-09365385cf59

This is the official implementation for **FlowMap: High-Quality Camera Poses, Intrinsics, and Depth via Gradient Descent** by Cameron Smith*, David Charatan*, Ayush Tewari, and Vincent Sitzmann.

Check out the project website [here](https://cameronosmith.github.io/flowmap/).

## Installation

To get started on Linux, create a Python virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For pretraining, make sure GMFlow is installed as a submodule:

```
git submodule update --init --recursive
```

If the above requirements don't work, you can try `requirements_exact.txt` instead.

## Running the Code

The main entry point is `flowmap/overfit.py`. Call it via:

```bash
python3 -m flowmap.overfit dataset=images dataset.images.root=path/to/folder/with/images
```

Make sure the virtual environment has been activated via `source venv/bin/activate` first.

## Pre-trained Initialization

The checkpoint we used to initialize FlowMap can be found [here](https://drive.google.com/drive/folders/1PqByQSfzyLjfdZZDwn6RXIECso7WB9IY?usp=drive_link). To train your own, download the [Real Estate 10k](https://google.github.io/realestate10k/) and [CO3Dv2](https://github.com/facebookresearch/co3d) datasets and run the following script:

```bash
python3 -m flowmap.pretrain
```

Some of the videos in the Real Estate 10k dataset are no longer publicly available. Reach out to us via email if you want our downloaded version of the dataset.

## Evaluation Datasets

We evaluated FlowMap using video subsets of the [Local Light Field Fusion (LLFF)](https://drive.google.com/drive/folders/1M-_Fdn4ajDa0CS8-iqejv0fQQeuonpKF?usp=drive_link), [Mip-NeRF 360](https://jonbarron.info/mipnerf360/), and [Tanks & Temples](https://www.tanksandtemples.org/download/) datasets. We've uploaded a compilation of these datasets [here](https://drive.google.com/drive/folders/1PqByQSfzyLjfdZZDwn6RXIECso7WB9IY?usp=drive_link).

<details>
<summary>Dataset Details</summary>

### NeRF Local Light Field Fusion (LLFF) Scenes

These are the LLFF scenes from the [NeRF](https://www.matthewtancik.com/nerf) paper, which were originally uploaded [here](https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7?usp=drive_link). We used all 8 scenes (`fern`, `flower`, `fortress`, `horns`, `leaves`, `orchids`, `room`, and `trex`).

### Mip-NeRF 360 Scenes

These are scenes from the [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) paper, which were originally uploaded [here](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip). We used the `bonsai`, `counter`, and `kitchen` scenes. The original `kitchen` scene consists of several concatenated video sequences; for FlowMap, we use the first one (65 frames). We also included the `garden` scene, which is somewhat video-like, but contain large jumps that make optical flow estimation struggle.

### Tanks & Temples Scenes

We used all scenes from the [Tanks & Temples](https://tanksandtemples.org/download/) dataset: `auditorium`, `ballroom`, `barn`, `caterpillar`, `church`, `courthouse`, `family`, `francis`, `horse`, `ignatius`, `lighthouse`, `m60`, `meetingroom`, `museum`, `palace`, `panther`, `playground`, `temple`, `train`, and `truck`. We preprocessed the raw videos from the dataset using the script at `flowmap/subsample.py`. This script samples 150 frames from the first minute of video evenly based on mean optical flow.

</details>

## Running Ablations

Each ablation shown in the paper has a [Hydra](https://hydra.cc/docs/intro/) configuration at `config/experiment`. For example, to run the ablation where point tracking is disabled, add `+experiment=ablation_no_tracks` to the overfitting command. Note that you can stack most of the ablations, e.g., `+experiment=[ablation_no_tracks,ablation_random_initialization]`.

## Figure and Table Generation

Some of the code used to generate the tables and figures in the paper can be found in the `assets` folder.

## BibTeX

```bibtex
@inproceedings{smith24flowmap,
      title={FlowMap: High-Quality Camera Poses, Intrinsics, and Depth via Gradient Descent},
      author={Cameron Smith and David Charatan and Ayush Tewari and Vincent Sitzmann},
      year={2024},
      booktitle={arXiv},
}
```

## Acknowledgements

This work was supported by the National Science Foundation under Grant No. 2211259, by the Singapore DSTA under DST00OECI20300823 (New Representations for Vision), by the Intelligence Advanced Research Projects Activity (IARPA) via Department of Interior/ Interior Business Center (DOI/IBC) under 140D0423C0075, by the Amazon Science Hub, and by IBM. The Toyota Research Institute also partially supported this work. The views and conclusions contained herein reflect the opinions and conclusions of its authors and no other entity.
