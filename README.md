# 🧠 Cascading Multi-Agent Anomaly Detection in Surveillance Systems  
### via Vision-Language Models and Embedding-Based Classification

---

## 🔍 Overview

This repository hosts the implementation and documentation for  
**“Cascading Multi-Agent Anomaly Detection in Surveillance Systems via Vision-Language Models and Embedding-Based Classification.”**

The system introduces a **multi-stage (cascading) pipeline** that merges classical vision techniques with transformer-based semantic reasoning to achieve **real-time, interpretable anomaly detection** in multi-camera security environments.

---

## 🚀 Key Features

- **Hybrid Cascade Architecture**
  - `YOLOv8` for object-level cues  
  - Autoencoder for reconstruction-gated anomaly scoring  
  - Vision-Language Model (VLM) for semantic interpretation  
  - Sentence-Transformer Embedding Classifier for structured labels  

- **Actionable Intelligence**  
  Converts VLM free-text (e.g., *“the camera is dark and covered”*) → `camera_blocked`.

- **Few-Shot Expandability**  
  Add new anomaly types by defining text examples—no retraining needed.

- **Explainable Outputs**  
  Combines visual bounding boxes, textual context, and structured JSON alerts.


---

## 🧾 Dataset

Primary dataset: **UCF-Crime (Full Dataset)**  
> 1,900 untrimmed surveillance videos (~128 h) covering 13 anomaly classes + normal scenes.

**Download Link:** [https://www.kaggle.com/datasets/minmints/ufc-crime-full-dataset](https://www.kaggle.com/datasets/minmints/ufc-crime-full-dataset)

```bash
# Download via Kaggle API
pip install kaggle
kaggle datasets download -d minmints/ufc-crime-full-dataset -p data/ --unzip

