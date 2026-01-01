# datasets_imagenet.py
import os, numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

def center_crop_arr(pil_image, image_size):
    # ADM/LFM center-crop (exactly as used in their code)
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    h, w = arr.shape[0], arr.shape[1]
    cy, cx = (h - image_size) // 2, (w - image_size) // 2
    return Image.fromarray(arr[cy:cy+image_size, cx:cx+image_size])

class ImageNetTrainFolder(Dataset):
    """
    Minimal ImageNet train loader matching LFM preprocessing.
    Root must contain 1000 subfolders: n01440764/, n01443537/, ...
    Returns (img: FloatTensor in [-1,1], label: int in [0,999]).
    """
    def __init__(self, root, image_size=256, horizontal_flip=True):
        assert os.path.isdir(root), f"Root not found: {root}"
        base_tfms = [
            T.Lambda(lambda im: center_crop_arr(im, image_size)),
            T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(lambda x: x),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1,1]
        ]
        self.transform = T.Compose(base_tfms)
        # ImageFolder builds class_to_idx from sorted subfolder names (stable)
        self.ifolder = ImageFolder(root=root, transform=self.transform)

    def __len__(self):
        return len(self.ifolder)

    def __getitem__(self, idx):
        img, y = self.ifolder[idx]  # img already transformed
        return img, y  # img: [3,H,W] float in [-1,1]; y: int

def make_imagenet_train_loader(
    root="/data/shahriar/datasets/kagglehub_cache/datasets/ImageNet_ILSVRC_TrainSet/"
         "thbdh5765/ilsvrc2012/versions/1",
    image_size=256,
    batch_size=160,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True,
):
    ds = ImageNetTrainFolder(root=root, image_size=image_size, horizontal_flip=True)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=True,
    )
    return ds, dl


#### PreEncoded 

import csv, json, pathlib, torch
from torch.utils.data import Dataset

class ImageNetLatents(Dataset):
    def __init__(self, root, device=None, dtype=None, strict=True):
        self.root = pathlib.Path(root)
        self.class_to_idx = (
            json.load(open(self.root / "class_to_idx.json"))
            if (self.root / "class_to_idx.json").exists() else None
        )

        # CSV -> list of (latent_relpath, class_idx)
        self.items = []
        with open(self.root / "index.csv", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.items.append((row["latent_path"], int(row["class_idx"])))

        # Keep the original args (API compatibility), but DO NOT move to CUDA here.
        self.device = device
        # allow 'fp16'/'fp32' strings or actual torch.dtype
        self.dtype = dict(fp16=torch.float16, fp32=torch.float32).get(dtype, dtype)
        self.strict = strict

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        # Robust fetch with a few fallbacks if strict=False
        j = i
        tries = 0
        while True:
            latent_rel, cls_csv = self.items[j]
            fp = self.root / latent_rel
            if fp.exists():
                break
            if self.strict or tries >= 8:
                # Strict mode or too many misses -> raise
                raise FileNotFoundError(str(fp))
            # Try a nearby index (helps if a few files are missing/corrupt)
            j = (j + 1) % len(self.items)
            tries += 1

        # Always load on CPU inside workers
        d = torch.load(fp, map_location="cpu")  # {"latent": Tensor, "label": int}
        z = d["latent"]
        y = int(d.get("label", cls_csv))  # prefer saved label; fallback to CSV

        # Optional dtype cast on CPU (safe in workers)
        if self.dtype is not None:
            z = z.to(self.dtype)

        # IMPORTANT: do NOT .to(self.device) here — keep CPU for DataLoader workers.
        # Move to GPU later in the training loop:
        #   x = x.to(device, non_blocking=True); y = y.to(device)
        #
        # Make tensors contiguous & pinned-memory friendly
        z = z.contiguous()

        return z, y


