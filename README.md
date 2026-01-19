# GraphFlowMatching (AAAI 2026)

Official code release for the AAAI 2026 paper **Graph Flow Matching: Enhancing Image Generation with Neighbor-Aware Flow**.

Paper (arXiv): https://arxiv.org/abs/2505.24434

This repository implements **Graph Flow Matching (GFM)**: flow matching in **VAE latent space**, augmented with a **graph-based correction** term that couples each sample to its neighbors via an adjacency matrix (e.g., attention / cosine / kNN).

---

## Setup

Create the environment from the provided Conda file:

```bash
conda env create -f environment.yml
conda activate flowmatch
```

---

## Repository Structure

- `train.py` : main training script
- `datasets/StableVAE_EncodedDatasetCreator.py` : pre-encode datasets into Stable VAE latents (for faster training)
- `datasets/CleanFIDCustomStatsCreator.py` : create CleanFID custom statistics for datasets
- `datasets/` : dataset loaders and preprocessing utilities
- `networks/` : model definitions (velocity networks, hybrid models, backbones)

---

## Pre-encode Dataset (Stable VAE Latents)

To avoid encoding images during every training iteration, pre-encode a dataset into Stable VAE latents. Here's how we did it (similar for the other datasets):

```bash
python datasets/StableVAE_EncodedDatasetCreator.py
```

After encoding, train using:

- `--use_pre_encoded`
- `--encoded_dataset_path /path/to/encoded_dataset`

---

## CleanFID Custom Dataset Stats

To compute FID using CleanFID with custom stats, create the dataset statistics first:

```bash
python datasets/CleanFIDCustomStatsCreator.py
```

Then use the dataset name during training via:

- `--cleanfid_dataset_name <name>`

---

## Training

Run training with:

```bash
python train.py
```

Example (pre-encoded latents):

```bash
python train.py   --dataset lsun_bedrooms   --use_pre_encoded   --encoded_dataset_path /path/to/encoded_dataset   --base_model dit   --adj_mode attention   --flow_model_type nonLinearHeatDiffusion2   --train_batch_size 50   --device cuda:0
```

---

## Common Flags

- `--dataset {ffhq,lsun_bedrooms,lsun_church,celeba-hq,AFHQ-Cat-Full-256}`
- `--use_pre_encoded` and `--encoded_dataset_path`
- `--base_model {dit,adm,resnet,pnpUNet}`
- `--adj_mode {attention,cosine,gaussian,knn}` (use `--knn_k` for kNN)
- `--diffusion` / `--no-diffusion`
- `--use_wandb`
- `--use_pretrained`

---

## Outputs

The script saves:
- model checkpoints to `--model_savepath`
- generated samples (and optional reconstructions) to `--image_savepath`

---

## Citation

If you use this repository, please cite the AAAI 2026 paper:

```bibtex
@misc{siddiqui2025graphflowmatchingenhancing,
      title={Graph Flow Matching: Enhancing Image Generation with Neighbor-Aware Flow Fields}, 
      author={Md Shahriar Rahim Siddiqui and Moshe Eliasof and Eldad Haber},
      year={2025},
      eprint={2505.24434},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.24434}, 
}
```

---

## License

Research use only. See `LICENSE` if included.
