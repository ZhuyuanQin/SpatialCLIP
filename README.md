# SpatialCLIP
<b>SpatialCLIP: Enhancing Quality Assessment of Early Pregnancy Fetal Ultrasound via Structured Multi-Modal Learning  </b> <br><em>Project Contributors</em></br>

This repository is the official implementation of **SpatialCLIP** (Image × Text × Spatial fusion with alignment diagnostics) for early-pregnancy ultrasound **standard vs non-standard** view-quality assessment.

## Overview

<img src="Document/SpatialCLIP Architecture.png" width="230px" align="right" />

**Abstract:** **Background.** In obstetric practice, **ultrasound image quality** critically affects the accuracy of **fetal biometry** and directly influences **missed** and **misdiagnosis** rates. High-quality imaging improves anomaly detection and lowers the risk of misdiagnosing conditions such as **Down syndrome** and **central nervous system (CNS) malformations**. Existing studies are mainly **image-only** or **image–text fusion** and lack **explicit modeling** of the **topological relations** and **spatial distributions** of key anatomical structures, limiting their utility for **structure-sensitive** quality control.

**Methods.** We propose **SpatialCLIP**, a **spatially enhanced tri-modal** model built on **CLIP** with three branches: an **image** branch that lightly fine-tunes **ViT-L/14** for global representations; a **text** branch that leverages **BioGPT** to generate descriptions and semantically model paired clinical narratives; and a **spatial** branch that encodes coordinates and **anatomical relations**. The three modalities are combined via **gated** late fusion followed by a classifier for **standard / non-standard** view discrimination.

**Results.** On a **retrospective** test set, **SpatialCLIP** outperforms a shared-encoder **multi-task baseline** without explicit fusion in both overall **discrimination** and **calibration**. **Ablation** shows independent contributions from all three modalities—removing any branch degrades performance. Stratified by view, all four views remain stable and **robust to class imbalance**. In a **prospective cohort**, the model maintains strong discrimination and usable calibration under **distribution shift**. Furthermore, on a **multicenter external validation** cohort from five independent medical centers, SpatialCLIP sustains these advantages, demonstrating solid **cross-institutional generalization** and **clinical transferability**.

**Conclusion.** The proposed **SpatialCLIP**—with **spatially enhanced tri-modal** modeling—significantly improves early-pregnancy ultrasound **quality control** in terms of **discrimination** and **calibration**, and exhibits stable **generalizability** in multicenter external validation, indicating strong potential for **clinical deployment**. As a front-end QC step, SpatialCLIP may enhance the reliability of **biometry** and reduce **missed** and **misdiagnosis** risks for major anomalies.


<p align="center">
  <img src="Document/SpatialCLIP%20Architecture.png" width="1400px" align="center" />
</p>

## Data preparation

Each sample is a JSON object with **absolute pixel** boxes from the original image. The loader will letterbox/resize to the target size, convert boxes to `(x, y, w, h)` internally, and normalize as needed. Anatomy labels are mapped to indices (default 14 classes).

**Schema (per sample)**

* `image_path` *(str)* — path to the image file
* `description` *(str)* — free/structured caption for the text branch
* `class_label` *(str)* — `"standard"` or `"non-standard"`
* `coordinates` *(list)* — zero or more detections, each with

  * `label` *(str)* — anatomy code (e.g., `CB, CP1, CP2, CF, …`)
  * `bbox` *(float[4])* — `[x_min, y_min, x_max, y_max]` **in pixels** (original image)
  * `confidence` *(float)* — detection confidence in `[0,1]`
  * other fields like `center/width/height` are optional and ignored

**Example**

```json
{
  "image_path": "/data/.../IMG_20220304160958_0001.bmp",
  "description": "Acquired ultrasound representation of a developing fetus captures 6 tissue components: ...",
  "class_label": "non-standard",
  "coordinates": [
    {"label": "CP2", "bbox": [522.2861, 351.3498, 712.2336, 443.1224], "confidence": 0.8106},
    {"label": "CP1", "bbox": [492.4823, 446.4173, 671.1511, 565.0831], "confidence": 0.5410},
    {"label": "CB",  "bbox": [418.8234, 290.6241, 765.0051, 609.8238], "confidence": 0.5165},
    {"label": "CF",  "bbox": [464.8315, 399.1847, 529.4395, 438.4522], "confidence": 0.5142}
  ]
}
```

## Requirements

Install all dependencies from the project’s pinned file:

```bash
# (Optional) create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

**Get CLIP via Git (Hugging Face Hub with Git LFS):**

```bash
# Install Git LFS if not already
git lfs install

# Clone the CLIP ViT-L/14 checkpoint
git clone https://huggingface.co/openai/clip-vit-large-patch14 third_party/clip-vit-large-patch14
```

Then pass the path in your commands, e.g.:

```bash
--clip_model_path third_party/clip-vit-large-patch14
```


## Training & Testing

### 1. Training (three-stage)

1. **Stage 0** – freeze CLIP; train **Spatial Encoder + fusion + classifier**
2. **Stage 1** – unfreeze projection layers; enable **pairwise InfoNCE** and optional modality dropout
3. **Stage 2** – unfreeze top layers (image/text) for **joint fine-tuning**; optional **feature-preservation** loss

**Run**

```bash
python train.py \
  --train_json your path \
  --val_json   your path \
  --test_json  your path \
  --batch_size 16 \
  --fusion_type gated \
  --output_dir ./outputs \
  --experiment_name tri_modal_run1
```

> Checkpoints (`checkpoint_best.pt`, `checkpoint_final.pt`) and `config.json` are saved under `outputs/<experiment>/`.

### 2. Evaluation & visualization

```bash
python evaluate.py \
  --model_path   ./outputs/tri_modal_run1/checkpoint_best.pt \
  --config_path  ./outputs/tri_modal_run1/config.json \
  --test_json    your path \
  --batch_size   16 \
  --output_dir   ./eval_results
```

Reports **Accuracy / F1 / ROC-AUC / PR-AUC**, and saves **confusion matrix**, **PR curve**, **gate-weights** (mean±std).
(Optional) **Alignment health**: cosine-similarity distributions among modality pairs.

## Acknowledgement

This code builds upon **CLIP** via Hugging Face `transformers`. The Spatial Encoder, fusion heads, training utilities, and evaluation plots are tailored for fetal ultrasound view-quality assessment.
