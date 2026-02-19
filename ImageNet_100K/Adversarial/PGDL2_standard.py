"""
PGDL2_standard_trainset.py

Purpose
-------
Compute adversarial robustness of a *standard full-image* ImageNet ResNet-50 model
(using a custom PGD-L2 implementation) and report:
  (1) Robust accuracy vs epsilon (ε) curve, and
  (2) Mean/median minimal adversarial L2 distance required to flip predictions.

This is the "standard/full-frame" counterpart to the Foc-L (bbox-crop) robustness scripts.

Evaluation Protocol (Appendix A.5-style)
----------------------------------------
- Attack: PGD-L2 (untargeted), 10 steps, random initialization within the L2 ball.
- Step size: alpha = ε / steps  (here: steps = 10).
- Epsilon sweep (grid search): eps_list = [0.0, 0.25, 0.5, 0.75, 1.0].
- Domain: Attack is performed in pixel space; adversarial images are clamped to [0, 1].
- Model input: Images are normalized using ImageNet mean/std ONLY when fed into the model
  (i.e., logits = model(normalize(adv))).

Dataset / Subset
----------------
- Base dataset: torchvision.datasets.ImageFolder rooted at `dataset_root` (ImageNet train folder).
- Subset: indices loaded from `fixed10k_json` (expected to be indices of correctly-classified samples).
- Sampling: if the index list is larger than 15,000, a reproducible random sample of 15,000 is used.
- Inputs: full images (NOT bbox crops). Uses Resize(256) + CenterCrop(224).

Outputs
-------
- Prints summary statistics:
    * mean ± std of minimal adversarial L2 distances (achieved ||delta||_2 at first flip)
    * median minimal adversarial L2 distance
    * robust accuracy vs ε (fraction of samples with distance > ε)
- Saves results to: `custom_pgd_l2_train_results.json` with keys:
    distances, mean, std, median, robust_curve

Implementation Notes
--------------------
- "Minimal" distance is approximated by:
    (a) sweeping ε over a fixed grid, and
    (b) early-stopping within PGD once a flip occurs.
  When a flip occurs, we store the achieved true L2 norm ||delta||_2.
- If no flip is found up to max ε, the distance is recorded as max ε (censoring at the grid maximum).
- For strict "random restarts", add K random initializations per ε and keep the best (smallest ||delta||_2).
"""


import os
import json
import random
import torch
import numpy as np
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(0)

# ——— Settings —————————————————————————————————————————
dataset_root  = "/local/a/imagenet/imagenet2012/train/"
fixed10k_json = "train_all_correct_intersection.json"  # 77k indices
model_ckpt      = "/best_model.pth"
batch_size    = 1   # grid search per-sample
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# — Pixel transforms (no normalization) ———————————————————————————
pixel_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# — Normalization helpers —————————————————————————————————————
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
def normalize(x):
    return (x - mean) / std

# — Load indices & build DataLoader ———————————————————————————
with open(fixed10k_json, 'r') as f:
    fixed_indices = json.load(f)

# sample 15 000 without replacement, reproducibly
if len(fixed_indices) > 15_000:
    fixed_indices = random.sample(fixed_indices, 15_000)
    
print(f"[Info] Loaded {len(fixed_indices)} samples for custom PGD-L2 search")

full_ds   = datasets.ImageFolder(root=dataset_root, transform=pixel_transform)
subset_ds = Subset(full_ds, fixed_indices)
loader    = DataLoader(subset_ds, batch_size=batch_size,
                       shuffle=False, num_workers=4, pin_memory=True)

# — Load model ————————————————————————————————————————————
model = models.resnet50(pretrained=False, num_classes=1000).to(device)
ckpt  = torch.load(model_ckpt, map_location=device)
model.load_state_dict(ckpt)
model.eval()

# — PGD-L2 grid settings ————————————————————————————————————
eps_list = [0.0, 0.25, 0.5, 0.75, 1.0]
max_eps  = eps_list[-1]
steps    = 10

# — storage for minimal true norms —————————————————————————
distances = []
print("[Info] Running custom PGD-L2 with early stop...")

for imgs, labels in tqdm(loader, desc="PGD-L2 sweep", leave=False):
    x = imgs.to(device)
    y = labels.to(device)
    found = False

    # grid-search budgets
    for eps in eps_list:
        # random start inside L2-ball
        delta = torch.randn_like(x)
        d_flat = delta.view(1, -1)
        norm_d = d_flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
        # uniform radius [0, eps]
        r = torch.rand(1, device=device) * eps
        delta = delta / norm_d.view(1,1,1,1) * r.view(1,1,1,1)
        adv = (x + delta).clamp(0,1)

        # PGD-L2 iterations with early stop
        for _ in range(steps):
            adv.requires_grad_(True)
            logits = model(normalize(adv))
            loss   = torch.nn.CrossEntropyLoss()(logits, y)
            grad   = torch.autograd.grad(loss, adv)[0]
            # normalize gradient
            g_flat = grad.view(1, -1)
            g_norm = g_flat.norm(p=2, dim=1).clamp(min=1e-12)
            grad   = grad / g_norm.view(1,1,1,1)
            # step
            alpha = eps / steps
            adv = (adv + alpha * grad).detach()
            # project back into L2-ball
            delta = (adv - x)
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
        distances.append(max_eps)

# — compute summary stats & robust curve ———————————————————————
d = torch.tensor(distances)
out = {
    "distances":    distances,
    "mean":         float(d.mean()),
    "std":          float(d.std(unbiased=False)),
    "median":       float(d.median()),
    "robust_curve": {f"{eps:.3f}": float((d > eps).float().mean().item()) for eps in eps_list}
}

# — report & save ———————————————————————————————————————
print(f"Custom PGD-L2 minimal ε (mean±std): {out['mean']:.4f}±{out['std']:.4f}")
print(f"Median: {out['median']:.4f}")
print("Robust Accuracy vs ε:" )
for e,a in out["robust_curve"].items():
    print(f"  ε={e}: {a*100:.2f}%")

with open("custom_pgd_l2_train_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("[Saved] custom_pgd_l2_train_results.json")
