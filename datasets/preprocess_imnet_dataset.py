#!/usr/bin/env python3
import argparse, os, json, csv, math, pathlib
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL
from tqdm import tqdm
from contextlib import nullcontext

# ---------------- Precision manager ----------------
class PrecisionManager:
    def __init__(self, precision: str, device: str = "cuda"):
        assert precision in ("fp32", "fp16")
        self.precision = precision
        self.device = device
        self.compute_dtype = torch.float32 if precision == "fp32" else torch.float16
        self.param_dtype = self.compute_dtype
        self.storage_dtype = self.compute_dtype

    def autocast(self):
        # No AMP for fp32; enable AMP only for fp16
        return nullcontext() if self.precision == "fp32" else torch.cuda.amp.autocast(dtype=torch.float16)

    def cast_inputs(self, x: torch.Tensor):
        return x.to(self.compute_dtype)

    def to_model(self, module: torch.nn.Module):
        return module.to(self.device, dtype=self.param_dtype)

    def to_storage(self, x: torch.Tensor):
        return x.to(self.storage_dtype)

# ---------- transforms (ADM/LFM-compatible) ----------
def center_crop_arr(pil_image, image_size=256):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    h, w = arr.shape[0], arr.shape[1]
    cy = (h - image_size) // 2
    cx = (w - image_size) // 2
    return Image.fromarray(arr[cy:cy+image_size, cx:cx+image_size])

def pil_to_tensor_norm_minus1_1(pil_image):
    arr = np.array(pil_image, dtype=np.float32) / 255.0
    if arr.ndim == 2:  # grayscale -> RGB
        arr = np.stack([arr]*3, axis=-1)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [C,H,W], [0,1]
    tensor = tensor * 2.0 - 1.0                      # [-1,1]
    return tensor

# ---------- HF VAE wrappers ----------
class EncoderWrapper(nn.Module):
    def __init__(self, vae): super().__init__(); self.vae = vae
    def forward(self, x):
        posterior = self.vae.encode(x).latent_dist
        latents = posterior.mean
        log_var = posterior.logvar
        return latents * self.vae.config.scaling_factor, log_var

def load_vae(model_name_or_path, device):
    vae = AutoencoderKL.from_pretrained(model_name_or_path)
    vae.eval().to(device)
    for p in vae.parameters(): p.requires_grad = False
    print(f"[VAE] scaling_factor={getattr(vae.config,'scaling_factor',0.18215)}, "
          f"in={vae.config.in_channels}, out={vae.config.out_channels}, latent={vae.config.latent_channels}")
    return EncoderWrapper(vae)

# ---------- Dataset over folder tree ----------
class ImageNetFolderFlat(Dataset):
    """
    Iterates all images under root_dir/<class_wnid>/*.JPEG
    Returns (img_tensor[-1,1], class_idx, relpath) with center crop 256.
    """
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG"}
    def __init__(self, root_dir, image_size=256, class_to_idx=None):
        self.root = pathlib.Path(root_dir)
        classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        if class_to_idx is None:
            class_to_idx = {c: i for i, c in enumerate(classes)}
        self.class_to_idx = class_to_idx
        self.samples = []
        for cls in classes:
            cls_dir = self.root / cls
            for p in cls_dir.rglob("*"):
                if p.is_file() and p.suffix in self.IMG_EXTS:
                    rel = p.relative_to(self.root)
                    self.samples.append((str(rel), cls))
        self.samples.sort()
        self.image_size = image_size

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        relpath, cls = self.samples[idx]
        full = self.root / relpath
        with Image.open(full) as im:
            im = im.convert("RGB")
            im = center_crop_arr(im, self.image_size)
        x = pil_to_tensor_norm_minus1_1(im)  # [3,256,256], in [-1,1]
        y = self.class_to_idx[cls]
        return x, y, relpath

# ---------- Precompute and save latents ----------
def save_tensor(fp: pathlib.Path, tensor: torch.Tensor, label: int, storage_dtype: torch.dtype, exist_ok=False):
    fp.parent.mkdir(parents=True, exist_ok=True)
    if fp.exists() and not exist_ok:
        return False
    t = tensor.to(storage_dtype)
    torch.save({"latent": t.cpu(), "label": int(label)}, fp)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", type=str, required=True,
        help="Root of ImageNet train split (folders like n01440764, n02119789, ...)")
    ap.add_argument("--dst_root", type=str, required=True,
        help="Parent directory where 'encoded_imagenet_train_split' will be created")
    ap.add_argument("--out_name", type=str, default="encoded_imagenet_train_split")
    ap.add_argument("--vae", type=str, default="stabilityai/sd-vae-ft-mse")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--precision", type=str, choices=["fp16","fp32"], default="fp32")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # Optional: improve TF32 matmul for fp32 on Ampere+/Ada
    if args.precision == "fp32":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    pm = PrecisionManager(args.precision, args.device)

    src_root = args.src_root
    dst_dir = pathlib.Path(args.dst_root) / args.out_name
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset and class mapping
    print("[Scan] indexing ImageNet folders...")
    ds_tmp = ImageNetFolderFlat(src_root, image_size=args.image_size)
    class_to_idx = ds_tmp.class_to_idx

    # Persist mapping for training-time loader
    with open(dst_dir / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    ds = ds_tmp
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # Prepare encoder and move to desired dtype/device
    enc = load_vae(args.vae, args.device)
    enc = pm.to_model(enc)

    # CSV index
    csv_path = dst_dir / "index.csv"
    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["relpath", "class_idx", "latent_path"])

        n_total = len(ds)
        n_saved, n_skipped = 0, 0

        with torch.no_grad():
            pbar = tqdm(dl, total=math.ceil(n_total / args.batch_size))
            for imgs, labels, relpaths in pbar:
                imgs = imgs.to(args.device, non_blocking=True)
                labels = labels.to(args.device, non_blocking=True)
                imgs = pm.cast_inputs(imgs)

                # No AMP used when precision=fp32
                with pm.autocast():
                    latents, _ = enc(imgs)  # [B, 4, 32, 32] for 256px SD VAE

                for i in range(imgs.size(0)):
                    rel = relpaths[i]
                    latent_rel = pathlib.Path(rel).with_suffix(".pt")
                    latent_fp = dst_dir / latent_rel

                    saved = save_tensor(
                        latent_fp,
                        latents[i],
                        int(labels[i].item()),
                        storage_dtype=pm.storage_dtype,
                        exist_ok=args.overwrite,
                    )
                    if saved:
                        n_saved += 1
                    else:
                        n_skipped += 1
                    csv_writer.writerow([rel, int(labels[i].item()), str(latent_rel).replace("\\", "/")])

                pbar.set_description(f"saved={n_saved} skipped={n_skipped}")

    print(f"[Done] Saved={n_saved}, Skipped(existing)={n_skipped}")
    print(f"[Index] {csv_path}")
    print(f"[Out]   {dst_dir}")

if __name__ == "__main__":
    main()


'''
python preprocess_imnet_dataset.py \
  --src_root /data/shahriar/datasets/kagglehub_cache/datasets/ImageNet_ILSVRC_TrainSet/thbdh5765/ilsvrc2012/versions/1 \
  --dst_root /data/shahriar/datasets/kagglehub_cache/datasets/ImageNet_ILSVRC_TrainSet/thbdh5765/ilsvrc2012/versions \
  --out_name encoded_imagenet_train_split \
  --vae stabilityai/sd-vae-ft-mse \
  --batch_size 150 --num_workers 8 --device cuda --precision fp32

'''