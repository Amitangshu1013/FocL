"""
PGDL2_foclsc_trainset.py

Purpose
-------
Estimate adversarial robustness of a Foc-L (bbox-crop / single-crop) ImageNet ResNet-50 model
using a PGD-L2 attack, and report:
  (1) Robust accuracy vs. epsilon (ε) curve, and
  (2) Mean/median minimal adversarial L2 distance required to flip predictions.

Evaluation Protocol (Appendix A.5-aligned)
------------------------------------------
- Attack: PGD-L2 (untargeted), 10 steps, random initialization within the L2 ball.
- Step size: alpha = ε / steps  (here: steps = 10).
- Epsilon sweep: eps_list = [0.0, 0.25, 0.5, 0.75, 1.0].
- Domain: Attack is performed in pixel space with inputs clamped to [0, 1].
- Model input: Images are normalized using ImageNet mean/std ONLY when fed into the model.
  (i.e., logits = model(normalize(adv))).

Dataset / Subset
----------------
- Base dataset: torchvision.datasets.ImageFolder rooted at `dataset_root` (ImageNet train folder).
- Subset: indices loaded from `fixed10k_json` (must correspond to correctly-classified samples).
- Sampling: if the index list is larger than 15,000, a reproducible random sample of 15,000 is used.
- Crops: BoundingBoxDataset uses bounding-box annotations in `annotation_folder` to produce object crops.

Outputs
-------
- Prints summary statistics:
    * mean ± std of minimal adversarial L2 distances (among found flips, capped at max ε if not flipped)
    * median minimal adversarial L2 distance
    * robust accuracy vs ε (fraction of samples with distance > ε)
- Saves results to: `focl_sc_train_results.json` with keys:
    distances, mean, std, median, robust_curve

Notes / Assumptions
-------------------
- This script approximates "minimal" adversarial distance by (a) sweeping ε over a fixed grid and
  (b) early-stopping once a flip is found for a given ε. It records the achieved ||delta||_2 at flip time.
- If no flip is found up to max ε, the distance is recorded as max ε (censoring at the grid maximum).
- For strict "random restarts", you would add multiple random initializations per ε and keep the best.
"""

import os
import json
import random
import torch
import numpy as np
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
from bounding_box_dataloader import BoundingBoxDataset

# — Reproducibility Setup —————————————————————————————————————
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark   = False

# ——— Settings —————————————————————————————————————————
dataset_root      = "/imagenet/imagenet2012/train/"
annotation_folder = "/Data/ImageNet_BBOX_Folder/Annotation"
fixed10k_json     = "train_all_correct_intersection.json"  # pre-sampled 77k indices
model_ckpt          = "best_model.pth" # Change per variant
batch_size        = 1   # one sample at a time for minimal-ε search
device            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# — Image transforms (pixel domain, no normalization) ———————————————————
pixel_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# — Normalization helpers —————————————————————————————————————
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
def normalize(x):
    return (x - mean) / std

# — Load fixed indices & prepare loader —————————————————————————
with open(fixed10k_json, 'r') as f:
    fixed_indices = json.load(f)
if len(fixed_indices) > 15000:
    fixed_indices = random.sample(fixed_indices, 15000)
print(f"[Info] Loaded {len(fixed_indices)} train samples for Foc-L Base PGD-L2 search")

full_ds   = datasets.ImageFolder(root=dataset_root, transform=pixel_transform)
bb_subset = Subset(full_ds, fixed_indices)
bb_ds     = BoundingBoxDataset(
    subset=bb_subset,
    annotation_folder=annotation_folder,
    crop_transform=pixel_transform
)
loader    = DataLoader(bb_ds, batch_size=batch_size,
                       shuffle=False, num_workers=4, pin_memory=True)

# — Load Foc-L Base model —————————————————————————————————————
def load_model(path):
    model = models.resnet50(pretrained=False, num_classes=1000).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
model = load_model(model_ckpt)

# — PGD-L2 grid & settings ———————————————————————————————————
eps_list = [0.0, 0.25, 0.5, 0.75, 1.0]
steps    = 10

# — Search minimal true ℓ₂ per sample via custom PGD-L2 —————————————————
distances = []
print("[Info] Searching minimal L2 distances via custom PGD-10 on Foc-L SC…")
for _, crop_img, label, _ in tqdm(loader, desc="Foc-L SC PGD-L2 train sweep"):
    x = crop_img.to(device)
    y = label.to(device)
    found = False

    for eps in eps_list:
        # random start within L2-ball
        delta = torch.randn_like(x)
        d_flat = delta.view(1, -1)
        n = d_flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
        r = torch.rand(1, device=device) * eps
        delta = delta / n.view(1,1,1,1) * r.view(1,1,1,1)
        adv = (x + delta).clamp(0,1)

        # PGD-L2 with early-stop
        for _ in range(steps):
            adv.requires_grad_(True)
            logits = model(normalize(adv))
            loss   = torch.nn.CrossEntropyLoss()(logits, y)
            grad   = torch.autograd.grad(loss, adv)[0]

            # gradient step
            g_flat = grad.view(1, -1)
            g_norm = g_flat.norm(p=2, dim=1).clamp(min=1e-12)
            grad   = grad / g_norm.view(1,1,1,1)
            adv    = (adv + (eps/steps) * grad).detach()

            # project to L2-ball
            delta  = adv - x
            d_flat = delta.view(1, -1)
            dist   = d_flat.norm(p=2, dim=1).clamp(min=1e-12)
            factor = torch.min(torch.ones_like(dist), eps/dist)
            delta  = delta * factor.view(1,1,1,1)
            adv    = (x + delta).clamp(0,1)

            # check flip
            with torch.no_grad():
                pred = model(normalize(adv)).argmax(dim=1)
            if pred.item() != y.item():
                true_dist = delta.view(1, -1).norm(p=2, dim=1)
                distances.append(true_dist.item())
                found = True
                break
        if found:
            break

    if not found:
        distances.append(eps_list[-1])

# — Metrics & save —————————————————————————————————————————
d = torch.tensor(distances)
out = {
    "distances":    distances,
    "mean":         float(d.mean()),
    "std":          float(d.std(unbiased=False)),
    "median":       float(d.median()),
    "robust_curve": {f"{eps:.3f}": float((d > eps).float().mean().item()) for eps in eps_list}
}

print(f"Custom PGD-L2 on Foc-L SC (mean±std): {out['mean']:.4f}±{out['std']:.4f}")
print(f"Median: {out['median']:.4f}")
print("Robust Accuracy vs ε:")
for e,a in out["robust_curve"].items():
    print(f"  ε={e}: {a*100:.2f}%")

with open("focl_sc_train_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("[Saved] focl_sc_train_results.json")
