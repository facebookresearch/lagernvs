# LagerNVS

LagerNVS is a feed-forward model for novel view synthesis (NVS). Given one or more input images of a scene, it synthesizes new views from arbitrary camera viewpoints **in a single forward pass, without per-scene optimization**. The model uses a VGGT-based encoder to extract 3D-aware scene features and a transformer-based renderer that conditions on target camera rays via cross-attention to generate novel views.

## Installation

```bash
# Clone the repository
git clone https://github.com/facebookresearch/lagernvs.git
cd lagernvs

# Create conda environment
conda create -n lagernvs python=3.10
conda activate lagernvs

# Install PyTorch (CUDA 12.4)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install -r requirements.txt
```

## Minimal Inference

```python
import torch
from huggingface_hub import hf_hub_download
from models.encoder_decoder import EncDec_VitB8
from vggt.utils.load_fn import load_and_preprocess_images
from vis import create_target_camera_path, render_chunked
from eval.export import save_video

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

image_names = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"]
num_cond_views = len(image_names)
video_length = 100

# Load images (width=512, height scaled proportionally, dims divisible by 8)
images = (
    load_and_preprocess_images(image_names, mode="crop", target_size=512, patch_size=8)
    .to(device)
    .unsqueeze(0)
)
image_size_hw = (images.shape[-2], images.shape[-1])

# Create target camera path (uses VGGT internally for pose estimation)
rays, cam_tokens = create_target_camera_path(
    image_names, video_length, num_cond_views, image_size_hw, device, dtype
)

# Load model and render
# Use attention_to_features_type="full_attention" for Re10k model
# Use attention_to_features_type="bidirectional_cross_attention" for General/DL3DV models
model = EncDec_VitB8(pretrained_vggt=False, attention_to_features_type="bidirectional_cross_attention")
ckpt_path = hf_hub_download("facebook/lagernvs_general_512", filename="model.pt")
model.load_state_dict(torch.load(ckpt_path)["model"])
model.to(device)

with torch.no_grad():
    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        video_out = render_chunked(
            model, (images, rays, cam_tokens), num_cond_views=num_cond_views
        )

save_video(video_out[0], "output_video.mp4")
```

There is also an interactive notebook at `inference.ipynb` and a ready-to-run script at `minimal_inference.py` for step-by-step exploration.

## Available Checkpoints

All models use the `EncDecVitB/8` architecture (VGGT encoder + 12-layer renderer, patch size 8). Three checkpoints are available on HuggingFace:

| Checkpoint | HuggingFace Repo | Training Data | Resolution | Cond. Views | Camera Poses |
|-----------|-----------------|---------------|------------|-------------|--------------|
| General | [`facebook/lagernvs_general_512`](https://huggingface.co/facebook/lagernvs_general_512) | 15 datasets | 512 (longer side) | 1-10 | Posed and unposed |
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

**Note:** You need to request access to the DL3DV datasets on HuggingFace and authenticate via `huggingface-cli login` before running the download script.

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

1. **Prepare data** in Re10k or DL3DV format:
   - Images: `images/<sequence_id>/NNNNN.png`
   - Camera parameters: `metadata/<sequence_id>.json` with:
     - `intrinsics`: [fx, fy, cx, cy] per frame (pixels)
     - `extrinsics`: 4x4 camera-to-world matrices per frame

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
      normalization_mode: cameras
```

### Training Notes

- Checkpoints are saved every 5000 iterations to `output/<exp_name>/checkpoints/`
- Training logs are written to TensorBoard in `output/<exp_name>/`
- The training configs use `normalization_mode: cameras` which normalizes scene scale based on camera positions (does not require depth maps)
- For multi-dataset training, use the base config template and add multiple subdatasets

The training configs for Re10k and DL3DV are provided in `config/train_re10k.yaml` and `config/train_dl3dv.yaml`. For the general model, the config is available at [`facebook/lagernvs_general_512`](https://huggingface.co/facebook/lagernvs_general_512). Full details necessary for reproducing the general model, including hyperparameters and datasets used, are available in our [paper](TODO_PAPER_LINK).

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

---

*This release is intended to support the open source research community.*
