
# Ontology-Guided Multi-Modal Perception for Trusted and Explainable Robotic Action

This repository contains the implementation and ontology resources for the paper
**“Ontology-Guided Multi-Modal Perception for Trusted and Explainable Robotic Action” (2025).**
A single entry-point script, `main.py`, runs the full pipeline: segmentation, ontology export (RDF/OWL-Lite), SHACL validation, explanation generation, and optional planner simulation.

---

## 1. Dataset

**Dataset:** GOOSE multi-modal navigation dataset
**Download:** <PUT YOUR DATASET LINK HERE>
*(Replace this line with your final URL; if you use Zenodo/GOOSE, keep the DOI or direct link.)*

After download, extract to:

```
data/goose/
    RGB/
    Depth/
    Annotations/        # zone masks or labels (Safe, Caution, Restricted)
    Telemetry/          # pose, orientation, velocity (optional)
```

If your folder names differ, pass them via CLI flags (see Section 6).

---

## 2. Requirements

* Python 3.10+
* PyTorch ≥ 2.3
* torchvision ≥ 0.18
* transformers ≥ 4.44
* timm
* rdflib ≥ 7.0
* pySHACL ≥ 0.25
* opencv-python-headless ≥ 4.9
* numpy, pandas, tqdm, matplotlib

Install:

```bash
pip install torch torchvision timm transformers rdflib pyshacl opencv-python-headless numpy pandas tqdm matplotlib
```

---

## 3. Repository Layout

```
main.py                   # single entry point: runs the full pipeline
ontology/
  ontology.owl            # OWL-Lite schema (entities & relations)
  safety_shapes.shacl     # SHACL safety constraints
  perception_schema.ttl   # perception classes/relations (optional modularization)
out/
  masks/                  # segmentation outputs
  rdf/                    # RDF triples (TTL)
  shacl/                  # validation reports / conformance
  explanations/           # factual/contrastive texts and overlays
  planner/                # baseline vs clearance-aware trajectories (optional)
data/
  goose/                  # dataset root (see Section 1)
```

Create empty `out/` subfolders if your OS does not auto-create them.

---

## 4. Quick Start

Default run (SegFormer + full pipeline):

```bash
python main.py \
  --data_root data/goose \
  --model segformer \
  --out_dir out \
  --run_perception \
  --run_mapping \
  --run_validation \
  --run_explanations
```

DeepLabV3 (ResNet-50) instead of SegFormer:

```bash
python main.py \
  --data_root data/goose \
  --model deeplabv3 \
  --out_dir out \
  --run_perception --run_mapping --run_validation --run_explanations
```

Add planner simulation:

```bash
python main.py \
  --data_root data/goose \
  --model segformer \
  --out_dir out \
  --run_perception --run_mapping --run_validation --run_explanations \
  --run_planner baseline clearance
```

---

## 5. Outputs

* `out/masks/` — segmentation masks and optional overlays
* `out/rdf/` — per-frame RDF triples (`.ttl`) and batched graphs
* `out/shacl/` — SHACL conformance reports (valid/violations, JSON/Turtle)
* `out/explanations/` — factual and contrastive text, plus visual highlights
* `out/planner/` — JSON/PNG for baseline and clearance-aware paths (if enabled)

---

## 6. Command-Line Arguments

```
--data_root PATH             Root folder of the dataset (default: data/goose)
--rgb_dir NAME               Subfolder for RGB (default: RGB)
--depth_dir NAME             Subfolder for depth (default: Depth)
--anno_dir NAME              Subfolder for annotations (default: Annotations)
--telemetry_dir NAME         Subfolder for telemetry (default: Telemetry)

--model {segformer,deeplabv3}  Perception model to use (default: segformer)
--input_size N                Square resize (e.g., 512) for inference (default: 512)
--batch_size N                Batch size for perception (default: 4)

--run_perception              Run segmentation
--run_mapping                 Export RDF/OWL-Lite triples
--run_validation              Run SHACL validation
--run_explanations            Generate textual/visual explanations
--run_planner [LIST]          Planner modes to run: baseline, clearance

--ontology_file PATH          Ontology OWL file (default: ontology/ontology.owl)
--shacl_file PATH             SHACL shapes file (default: ontology/safety_shapes.shacl)

--out_dir PATH                Output root (default: out)
--num_workers N               DataLoader workers (default: 4)
--device {cuda,cpu}           Force device selection (default: auto)
--seed N                      Random seed for reproducibility (default: 42)

--max_frames N                Limit processed frames for quick tests (default: None)
--save_overlays               Save mask overlays for inspection
--save_merged_graphs          Save per-sequence merged RDF graphs
```

Examples:

* Custom dataset subfolders:

  ```bash
  python main.py --data_root data/goose \
    --rgb_dir images --anno_dir labels --depth_dir depth
  ```
* Quick smoke test on 200 frames:

  ```bash
  python main.py --data_root data/goose --max_frames 200 \
    --run_perception --run_mapping --run_validation
  ```

---

## 7. Models

### 7.1 SegFormer (default)

* HF model: `nvidia/segformer-b2-finetuned-cityscapes-1024-1024`
* Framework: `transformers` + `timm`
* Good balance of speed/accuracy for navigation scenes

### 7.2 DeepLabV3 (ResNet-50)

* Torchvision checkpoint: `deeplabv3_resnet50_coco`
* Framework: PyTorch/torchvision
* Stable baseline; easily portable

Model selection via `--model` flag. If you must run offline, place checkpoints under `~/.cache` or a custom path and modify loader logic in `main.py`.

---

## 8. Ontology & SHACL

* `ontology/ontology.owl`: OWL-Lite classes for `Robot`, `Zone`, `Obstacle`, etc.
* `ontology/safety_shapes.shacl`: constraints for zone entry, clearance, and speed.
* Generated triples follow `(subject, predicate, object)`, e.g.:

  ```
  :Robot1 :inside :RestrictedZone03 .
  ```
* SHACL validation reports include offending triples and shape IDs to support explanations.

---

## 9. Reproducible Environment (optional)

Create a minimal, pinned environment file:

```
python==3.10
torch==2.3.1
torchvision==0.18.1
transformers==4.44.0
timm==1.0.7
rdflib==7.0.0
pyshacl==0.25.0
opencv-python-headless==4.10.0.84
numpy==1.26.4
pandas==2.2.2
tqdm==4.66.4
matplotlib==3.8.4
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 10. Notes

* GPU is recommended (T4 or better). Use `--device cpu` for CPU-only runs.
* If your annotations encode zone classes differently, adjust the mapping table inside `main.py`.
* For large runs, enable `--save_merged_graphs` to reduce file count by sequence.
* Explanations are template-based; switch off with `--run_explanations` omitted.

---

## 11. Citation and Availability

Zenodo (code/resources): `https://doi.org/10.5281/zenodo.17466601`
If you use this repository, please cite the accompanying paper (add your bibliographic entry here).

---

If you paste your **exact dataset URL**, I’ll drop it into Section 1 and tweak any folder names to match your archive.
