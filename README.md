# LagerNVS: Latent Geometry for Fully Neural Real-Time Novel View Synthesis

<p align="center">
  <a href="https://arxiv.org/abs/2603.20176"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b" alt="arXiv"></a>
  <a href="https://szymanowiczs.github.io/lagernvs"><img src="https://img.shields.io/badge/🌐-Project_Page-orange" alt="Project Page"></a>
  <a href="https://github.com/facebookresearch/lagernvs"><img src="https://img.shields.io/badge/GitHub-Repo-blue" alt="GitHub"></a>
  <a href="https://huggingface.co/collections/facebook/lagernvs"><img src="https://img.shields.io/badge/HuggingFace-Model-green?logo=huggingface" alt="Models"></a>
</p>

<p align="center">
  Stanislaw Szymanowicz<sup>1,2</sup>, Minghao Chen<sup>1,2</sup>, Jianyuan Wang<sup>1,2</sup>, Christian Rupprecht<sup>1</sup>, Andrea Vedaldi<sup>1,2</sup>
</p>

<p align="center">
  <sup>1</sup>Visual Geometry Group (VGG), University of Oxford &nbsp;&nbsp; <sup>2</sup>Meta AI
</p>

LagerNVS is a feed-forward model for novel view synthesis (NVS). Given one or more input images of a scene, it synthesizes new views from a target cameras. It generalizes to in-the-wild data, renders new views in real time and can operate with or without known source camera poses.
The model uses 3D biases without explicit 3D representations. The architecture features a large 3D-aware encoder (from VGGT pre-training) to extract scene tokens and a transformer-based renderer that conditions on these tokens via cross-attention to render novel views.

## Announcements
[27 Mar 2026] We are aware that the model produces unsatisfactory results when target camera intrinsics vary substantially from source camera intrinsics. We are working on a fix.

[27 Mar 2026] We have technical issues with granting access to the model. It might take a couple of days after filing the request before you get the access, and we are actively working to reduce that time.

## Installation

```bash
# Clone the repository
git clone https://github.com/facebookresearch/lagernvs.git
cd lagernvs

# Create conda environment
conda create -n lagernvs python=3.10
conda activate lagernvs

# Install PyTorch (CUDA 12.6)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# Install remaining dependencies
pip install -r requirements.txt
```

## Model Access

The model checkpoints are hosted on HuggingFace as **gated** repositories. You must authenticate before downloading:

1. **Create a HuggingFace account** at [https://huggingface.co](https://huggingface.co) if you don't have one.
2. **Request access** by visiting the model page (e.g., [`facebook/lagernvs_general_512`](https://huggingface.co/facebook/lagernvs_general_512)) and clicking **"Agree and access repository"**.
3. **Create an access token** at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Create a token with at least `Read` scope.
4. **Set the token** as an environment variable:
```bash
export HF_TOKEN=hf_your_token_here
```

To persist this across sessions, add the export to your `~/.bashrc` or write the token to the HuggingFace cache:
```bash
mkdir -p ~/.cache/huggingface
echo "hf_your_token_here" > ~/.cache/huggingface/token
```

You can verify access with:
```bash
python -c "from huggingface_hub import list_repo_files; print('\n'.join(list_repo_files('facebook/lagernvs_general_512')))"
```

## Minimal Inference

Run inference with the general model on your own images:

```bash
python minimal_inference.py --images path/to/img1.png path/to/img2.png
```

To use a different checkpoint (see [Available Checkpoints](#available-checkpoints) below):

```bash
# Re10k model (256px, 2-view, posed)
python minimal_inference.py \
    --images path/to/img1.png path/to/img2.png \
    --model_repo facebook/lagernvs_re10k_2v_256 \
    --attention_type full_attention \
    --target_size 256 \
    --mode square_crop

# DL3DV model (256px, 2-6 views, posed)
python minimal_inference.py \
    --images path/to/img1.png path/to/img2.png path/to/img3.png \
    --model_repo facebook/lagernvs_dl3dv_2-6_v_256 \
    --attention_type bidirectional_cross_attention \
    --target_size 256 \
    --mode square_crop
```

Run `python minimal_inference.py --help` for all options (`--video_length`, `--output`, etc.).

See [`minimal_inference.py`](minimal_inference.py) for the fully commented source of truth. For interactive step-by-step exploration with visualization of intermediate results (loaded images, camera trajectories, sampled output frames), see the [`inference.ipynb`](inference.ipynb) notebook.

## Available Checkpoints

All models use the `EncDecVitB/8` architecture (VGGT encoder + 12-layer renderer, patch size 8). Three checkpoints are available on HuggingFace.
We recommend using the **general** model for most use cases, as it is trained on a large dataset of scenes and can handle a wide range of input conditions.
Re10k and DL3DV models are shared primarily for benchmarking and reproducitbility.

| Checkpoint | HuggingFace Repo | Training Data | Resolution | Train Cond. Views | Camera Poses |
|-----------|-----------------|---------------|------------|-------------|--------------|
| General | [`facebook/lagernvs_general_512`](https://huggingface.co/facebook/lagernvs_general_512) | 13 datasets | 512 (longer side) | 1-10 | Posed and unposed |
| Re10k | [`facebook/lagernvs_re10k_2v_256`](https://huggingface.co/facebook/lagernvs_re10k_2v_256) | Re10k only | 256x256 | 2 | Posed only |
| DL3DV | [`facebook/lagernvs_dl3dv_2-6_v_256`](https://huggingface.co/facebook/lagernvs_dl3dv_2-6_v_256) | DL3DV only | 256x256 | 2-6 | Posed only |

Checkpoints are auto-downloaded from HuggingFace when using `hf://` paths in config files.

## Evaluation

### Re10k-only model: posed, 2-view, 256x256

1. Download Re10k data prepared by pixelSplat (CVPR 2024), [hosted here](http://schadenfreude.csail.mit.edu:8000/) and unzip.

2. Set up your data root directory. Download and run the preprocessing script from LVSM [process_data.py](https://github.com/Haian-Jin/LVSM/blob/main/process_data.py):
```bash
export LAGERNVS_DATA_ROOT=/path/to/your/data

# Preprocess test split
python process_data.py \
    --base_path /path/to/downloaded_and_unzipped/re10k_from_pixelsplat \
    --output_dir $LAGERNVS_DATA_ROOT/re10k \
    --mode test
```

The expected dataset organization after preprocessing is:
```
$LAGERNVS_DATA_ROOT/re10k/
└── test/
    ├── images/
    │   ├── <sequence_id>/
    │   │   ├── 00000.png
    │   │   ├── 00001.png
    │   │   └── ...
    ├── metadata/
    │   ├── <sequence_id>.json
    │   └── ...
    └── full_list.txt
```

3. Run evaluation:
```bash
# Evaluate on Re10k (posed, 2-view, 256x256)
torchrun --nproc_per_node=8 run_eval.py \
    -c config/eval_re10k.yaml \
    -e re10k_eval
```
The script defaults to using 8 GPUs with global batch size 512. By default, it saves images and renders videos as part of evaluation - this can be slow and use a lot of memory and storage. Adjust the batch and GPU size according to your hardware and optionally remove visualization saving from run_eval.py .


5. Verify expected scores: PSNR: 31.39 SSIM: 0.928 LPIPS: 0.078


### DL3DV-only model: posed, 2-, 4-, 6- view, 256x256

1. Download the DL3DV benchmark subset required for evaluation:
```bash
export LAGERNVS_DATA_ROOT=/path/to/your/data

cd data_prep/dl3dv
python download_eval.py \
    --output_dir $LAGERNVS_DATA_ROOT/dl3dv \
    --view_indices_path ../../assets/dl3dv_6v.json
```

This script automatically downloads scenes from the correct HuggingFace repositories:
- Most scenes come from [DL3DV/DL3DV-ALL-960P](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-960P) (scenes with XK prefix like "2K/...", "3K/...")
- A few (5) benchmark scenes are not included in the 'ALL' version, and thus have to be separately downloaded from [DL3DV/DL3DV-10K-Benchmark](https://huggingface.co/datasets/DL3DV/DL3DV-10K-Benchmark)

**Note:** You need to request access to the DL3DV datasets on HuggingFace and authenticate via `huggingface-cli login` or add your token to environment variables before running the download script.

2. The data structure should look like:
```
$LAGERNVS_DATA_ROOT/dl3dv/
├── <batch>/<sequence_id>/
│   ├── images_4/
│   │   ├── frame_00001.png
│   │   └── ...
│   └── transforms.json
├── full_list_train.txt
└── full_list_test.txt
```

3. Run evaluation:
```bash
# Evaluate on DL3DV (posed, 6-view, 256x256)
torchrun --nproc_per_node=8 run_eval.py \
    -c config/eval_dl3dv.yaml \
    -e dl3dv_eval
```
The script defaults to using 8 GPUs with global batch size 512. By default, it saves images and renders videos as part of evaluation - this can be slow and use a lot of memory and storage. Adjust the batch and GPU size according to your hardware and optionally remove visualization saving from run_eval.py.

We provide view indices for 2-view, 4-view, and 6-view evaluation in `assets/dl3dv_2v.json`, `assets/dl3dv_4v.json`, and `assets/dl3dv_6v.json`. To evaluate with a different number of views, update the download command and modify `config/eval_dl3dv.yaml` to point to the appropriate JSON file and set `num_cond_views` accordingly.

4. Verify expected scores (6-view): PSNR: 29.45 SSIM: 0.904 LPIPS: 0.068


### General Model (512 resolution)

The general model can be evaluated on any dataset that has been preprocessed in the format described above (Re10k or DL3DV format). Here we show unposed evaluation on DL3DV at 512x512 resolution.

1. Ensure DL3DV data is prepared as described in the [DL3DV section](#dl3dv-dataset) above.

2. Run evaluation:
```bash
# Evaluate general model on DL3DV (unposed, 6-view, 512x512)
torchrun --nproc_per_node=8 run_eval.py \
    -c config/eval_dl3dv_general.yaml \
    -e dl3dv_general_eval
```

3. Customizing evaluation settings:

The config file `config/eval_dl3dv_general.yaml` can be modified for different evaluation scenarios:
- **Posed vs unposed**: Set `zero_out_cam_cond_p: 0.0` for posed evaluation (ground truth poses) or `zero_out_cam_cond_p: 1.0` for unposed evaluation (source cameras unavailable, ground truth poses used for scale normalization and specifying target camera).
- **Number of views**: Update `num_cond_views` and `test_view_indices_path` to use `assets/dl3dv_2v.json`, `assets/dl3dv_4v.json`, or `assets/dl3dv_6v.json`
- **Resolution**: Adjust `data.im_size_hw` to change input/output resolution. Images are center-cropped to match the target aspect ratio, then resized to the specified dimensions. Both dimensions must be divisible by 8, longer side must be 512.

For evaluation on other datasets, organize your data in a similar format (images in a directory, camera parameters in JSON) and adapt the dataloader accordingly. Camera parameters should follow OpenCV conventions (right-handed coordinate system, camera looks along +Z axis, +Y points down).


## Training

Training LagerNVS requires preparing the training data first. We provide data loaders and training configs for Re10k and DL3DV datasets.

### Training Data Preparation

#### Re10k Dataset (for Training)

1. Download Re10k data prepared by pixelSplat (CVPR 2024) from [here](http://schadenfreude.csail.mit.edu:8000/) and unzip.

2. Set up your data root directory and run preprocessing:
```bash
export LAGERNVS_DATA_ROOT=/path/to/your/data

# Download the preprocessing script from LVSM
# https://github.com/Haian-Jin/LVSM/blob/main/process_data.py

# Preprocess train split
python process_data.py \
    --base_path /path/to/downloaded_and_unzipped/re10k_from_pixelsplat \
    --output_dir $LAGERNVS_DATA_ROOT/re10k \
    --mode train

# Preprocess test split
python process_data.py \
    --base_path /path/to/downloaded_and_unzipped/re10k_from_pixelsplat \
    --output_dir $LAGERNVS_DATA_ROOT/re10k \
    --mode test
```

The expected structure after preprocessing:
```
$LAGERNVS_DATA_ROOT/re10k/
├── train/
│   ├── images/<sequence_id>/*.png
│   ├── metadata/<sequence_id>.json
│   └── full_list.txt
└── test/
    ├── images/<sequence_id>/*.png
    ├── metadata/<sequence_id>.json
    └── full_list.txt
```

#### DL3DV Dataset (for Training)

1. Request access to the DL3DV dataset on HuggingFace:
   - Go to https://huggingface.co/datasets/DL3DV/DL3DV-ALL-960P
   - Request access and authenticate via `huggingface-cli login`

2. Download the DL3DV metadata CSV:
```bash
wget https://raw.githubusercontent.com/DL3DV-10K/Dataset/main/cache/DL3DV-valid.csv
```

3. Download training data using our provided script:
```bash
export LAGERNVS_DATA_ROOT=/path/to/your/data

# Download a batch of DL3DV training data (e.g., batch 2K with 1000 scenes)
cd data_prep/dl3dv
python download_train.py \
    --output_dir $LAGERNVS_DATA_ROOT/dl3dv \
    --batch 2K \
    --csv_path /path/to/DL3DV-valid.csv \
    --max_scenes 1000  # adjust based on your needs
    --test_view_indices_paths ../../assets/dl3dv_2v.json ../../assets/dl3dv_4v.json ../../assets/dl3dv_6v.json # excludes eval data from train download and .txt

# For full training, download multiple batches (1K through 10K)
# Each batch contains approximately 1000 scenes
```

4. Download evaluation data:
```bash
python download_eval.py \
    --output_dir $LAGERNVS_DATA_ROOT/dl3dv \
    --view_indices_path ../../assets/dl3dv_6v.json
```

The expected structure:
```
$LAGERNVS_DATA_ROOT/dl3dv/
├── <batch>/<sequence_id>/
│   ├── images_4/
│   │   ├── frame_00001.png
│   │   └── ...
│   └── transforms.json
├── full_list_train.txt
└── full_list_test.txt
```

### Training Commands

Train on Re10k or DL3DV from scratch:

```bash
# Train on Re10k (256x256, 2-view conditioning)
torchrun --nproc_per_node=8 train.py \
    -c config/train_re10k.yaml \
    -e re10k_train

# Train on DL3DV (256x256, 2-6 view conditioning)
torchrun --nproc_per_node=8 train.py \
    -c config/train_dl3dv.yaml \
    -e dl3dv_train

# Train general model on Re10k + DL3DV (512x512, 1-10 view conditioning)
torchrun --nproc_per_node=8 train.py \
    -c config/train_base.yaml \
    -e base_train
```

Training hyperparameters can be overridden via command line:
```bash
# Reduce batch size for fewer GPUs
torchrun --nproc_per_node=4 train.py \
    -c config/train_re10k.yaml \
    -e re10k_train_4gpu \
    --opt.batch_size 256

# Single GPU training for testing
torchrun --nproc_per_node=1 train.py \
    -c config/train_re10k.yaml \
    -e re10k_test \
    --opt.batch_size 4 \
    --opt.num_iter_total 100
```

### Adding Custom Datasets

To train on your own datasets:

1. **Prepare data** in Re10k format:
   - Images: `images/<sequence_id>/NNNNN.png`
   - Camera parameters: `metadata/<sequence_id>.json` with per-frame entries containing:
     - `fxfycxcy`: [fx, fy, cx, cy] intrinsics in pixels
     - `w2c`: 4x4 world-to-camera matrix (OpenCV convention: +Y down, +Z forward)
   - See `data_root/re10k/test/metadata/` for examples of the expected JSON format.

2. **Create a dataset class** in `data/sources/` following `re10k_dataset.py` as a template.

3. **Register your dataset** in `data/dataset_factory.py`:
```python
_dataset_registry = {
    "re10k": (re10k_dataset.Re10kDataset, SEQUENTIAL),
    "dl3dv": (dl3dv_dataset.Dl3dvDataset, SEQUENTIAL),
    "my_dataset": (my_dataset.MyDataset, SEQUENTIAL),  # Add this
}
```

4. **Add to config**:
```yaml
data:
  subdataset_list:
    - name: my_dataset
      view_sampler_range: [10, 30]
      expansion_factor: 1.0
      equalization_length: 50000
```

### Training Notes

- Checkpoints are saved every 5000 iterations to `output/<exp_name>/checkpoints/`
- Training logs are written to TensorBoard in `output/<exp_name>/`
- Included training dataloaders normalize scene scales based on camera positions (do not require depth maps)
- For multi-dataset training, use the base config template and add multiple subdatasets

The training configs for Re10k and DL3DV are provided in `config/train_re10k.yaml` and `config/train_dl3dv.yaml`. For the general model, the checkpoint is available at [`facebook/lagernvs_general_512`](https://huggingface.co/facebook/lagernvs_general_512). Full details necessary for reproducing the general model, including hyperparameters and datasets used, are available in our [paper](TODO_PAPER_LINK).

## Citation

If you use this code, please cite:

```bibtex
@InProceedings{lagernvs2026,
  title={LagerNVS: Latent Geometry for Fully Neural Real-time Novel View Synthesis},
  author={Stanislaw Szymanowicz and Minghao Chen and Jianyuan Wang and Christian Rupprecht and Andrea Vedaldi},
  booktitle={CVPR},
  year={2026}
}
```

## License

This project is licensed under the FAIR Noncommercial Research License. See [LICENSE](LICENSE) for details.

## Limitations

- The model is not expected to perform well on dynamic inputs, humans or animals.
- The model was trained with data where all source and target images have the same intrinsics. Deviation from this, or using images with large distortion will lead to performance degradation.
- Performance on very high-frequency regions such as grass or tree leaves is poor.
- The model does not hallucinate unobserved regions, because it is not a generative model.

---

*This release is intended to support the open source research community.*
