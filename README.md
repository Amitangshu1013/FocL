# From Clutter to Clarity: Visual Recognition through Foveated Object-Centric Learning (FocL)

Official codebase for **FocL**. (TMLR 2026)

**Paper:** ([OpenReview)](https://openreview.net/forum?id=kVS7sMlv7P)  
**Authors:** Amitangshu Mukherjee, Deepak Ravikumar and Kaushik Roy (Purdue University) 
**License:** MIT

<p align="center">
  <img src="assets/focl_multiglimpse_pipeline.png" width="900" alt="FocL multi-glimpse pipeline">
</p>
<p align="center">
  <em>Multi-glimpse foveation: sample fixation points around a center, crop constrained glimpses, and train on one or multiple foveated views.</em>
</p>

---

## Overview

FocL trains recognition models using **object-centric foveated crops** rather than relying on full-scene context.  
The core idea is to (i) localize an object (e.g., via bounding boxes), (ii) sample one or more fixation points around the object center, and (iii) train with constrained crops (“glimpses”) that emphasize object appearance while reducing background dependence.

This repository contains:
- Training and evaluation code for FocL-style object-centric recognition.
- Inference scripts for **ImageNet-V1**, **ImageNet-V2**, and **COCO**.
- Reproducibility assets: configs, curated metadata, and dataset split utilities.
- Baseline implementations for comparison:
  - **Standard** (full-image classifier)
  - **RoI-Align classifier**
  - **HardMask classifier**
- **Segment Anything Model (SAM)** integration to generate object proposals and support **cross-domain generalization** experiments on **ImageNet-V2** and **COCO**.

**Not included:** datasets (ImageNet/COCO), large checkpoints, and machine-specific annotation folders.

---

## Installation

### Option A: Conda (recommended)
```bash
conda env create -f focl.yml
conda activate focl
