#!/usr/bin/env python3
"""
Parameter-Efficient Fine-tuning of UNI2-h on HEST-1k using HEST APIs

This script leverages HEST's official loaders and preprocessing utilities to:
- Select samples by technology/species from HEST metadata
- Optionally filter to housekeeping genes and (optionally) stromal regions
- Optionally correct batch effects across samples
- Extract H&E patches around spots via HEST and load them efficiently
- Fine-tune UNI2-h with LoRA for ST regression

Author: Refactored for clean, reliable training with HEST utilities
"""

import os
import json
import math
import argparse
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model

import scanpy as sc

import wandb
from tqdm import tqdm

# HEST APIs
import hest
from hest.HESTData import unify_gene_names
from hest.batch_effect import (
    filter_housekeeping,
    filter_stromal_housekeeping,
    correct_batch_effect,
)
from hest.bench.st_dataset import load_adata
from hestcore.datasets import H5HESTDataset


def setup_wandb(project: str, args) -> bool:
    api_key = os.getenv("WANDB_API_KEY", None)
    if not api_key:
        print("wandb: no API key found in env, continuing without wandb.")
        return False
    try:
        wandb.login(key=api_key)
        wandb.init(project=project, config=vars(args))
        return True
    except Exception as e:
        print(f"wandb login failed: {e}. Continuing without wandb.")
        return False


def read_hest_metadata(assets_dir: str, fallback_hest_dir: Optional[str]) -> pd.DataFrame:
    candidates = [
        os.path.join(assets_dir, "HEST_v1_1_0.csv"),
    ]
    if fallback_hest_dir:
        candidates.append(os.path.join(fallback_hest_dir, "HEST_v1_1_0.csv"))
    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(
        f"HEST metadata CSV not found in {candidates}. Set --assets_dir or ensure the CSV exists."
    )


def select_samples(meta_df: pd.DataFrame, technology: str, species: Optional[str], max_samples: int) -> List[str]:
    df = meta_df.copy()
    df = df[df["st_technology"] == technology]
    if species:
        df = df[df["species"] == species]

    # Ensure referenced files exist under the hest_dir structure (st/wsis/metadata)
    # Sample IDs are in column 'id'
    ids = df["id"].tolist()
    if max_samples != -1:
        ids = ids[: max(0, min(max_samples, len(ids)))]
    if len(ids) == 0:
        raise ValueError("No samples selected. Check technology/species filters.")
    return ids


def ensure_patches(sample: hest.HESTData, patch_dir: str, patch_size: int, pixel_size_um: float) -> str:
    os.makedirs(patch_dir, exist_ok=True)
    patch_h5_path = os.path.join(patch_dir, f"{sample.meta['id']}.h5")
    if os.path.exists(patch_h5_path):
        return patch_h5_path
    # Create patches aligned to current adata.obsm['spatial']
    sample.dump_patches(
        patch_dir,
        name=sample.meta["id"],
        target_patch_size=patch_size,
        target_pixel_size=pixel_size_um,
        dump_visualization=False,
        use_mask=True,
        threshold=0.15,
    )
    return patch_h5_path


def build_common_gene_space(adatas: List[sc.AnnData]) -> List[str]:
    common = None
    for ad in adatas:
        genes = set(ad.var_names.tolist())
        common = genes if common is None else (common & genes)
    if not common:
        raise ValueError("No common genes across selected samples after filtering.")
    # deterministic ordering
    return sorted(list(common))


def reorder_genes(adatas: List[sc.AnnData], gene_order: List[str]) -> List[sc.AnnData]:
    out = []
    for ad in adatas:
        out.append(ad[:, gene_order].copy())
    return out


def _housekeeping_assets_path(assets_dir: str, species_key: str) -> str:
    filename = "MostStable_Human.csv" if species_key.lower() == "human" else "MostStable_Mouse.csv"
    cand1 = os.path.join(assets_dir, filename)
    cand2 = os.path.join(os.path.dirname(__file__), "..", "assets", filename)
    for p in [cand1, cand2]:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Housekeeping gene list not found. Tried: {cand1}, {cand2}")


def get_housekeeping_genes(assets_dir: str, species_key: str) -> List[str]:
    path = _housekeeping_assets_path(assets_dir, species_key)
    # Some assets use ';' as separator
    try:
        df = pd.read_csv(path, sep=';')
    except Exception:
        df = pd.read_csv(path)
    col = 'Gene name' if 'Gene name' in df.columns else df.columns[0]
    genes = df[col].astype(str).tolist()
    return genes


def safe_filter_housekeeping(adata: sc.AnnData, species_key: str, unify: bool, assets_dir: str) -> sc.AnnData:
    """Apply housekeeping filter robustly.
    1) Optionally unify gene names
    2) Try hest.filter_housekeeping; if it fails due to missing genes, fallback to intersection-based filtering
    """
    if unify:
        # unify_gene_names expects 'human' or 'mouse'
        adata = unify_gene_names(adata, "human" if species_key.lower() == "human" else "mouse")
    try:
        return filter_housekeeping(adata, species='human' if species_key.lower() == 'human' else 'mouse')
    except Exception as e:
        # Fallback: case-insensitive intersection and Ensembl->symbol remap if needed
        hk_list = get_housekeeping_genes(assets_dir, species_key)
        hk_upper = {g.upper() for g in hk_list}
        var_list = list(map(str, adata.var_names.astype(str)))
        var_upper_to_orig = {g.upper(): g for g in var_list}
        inter_upper = hk_upper.intersection(set(var_upper_to_orig.keys()))
        if len(inter_upper) == 0:
            # Try to convert Ensembl IDs (with/without version suffix) to gene symbols via biomart
            try:
                adata = map_ensembl_to_gene_symbols(adata, species_key)
                var_list = list(map(str, adata.var_names.astype(str)))
                var_upper_to_orig = {g.upper(): g for g in var_list}
                inter_upper = hk_upper.intersection(set(var_upper_to_orig.keys()))
            except Exception:
                inter_upper = set()
        if len(inter_upper) == 0 and unify:
            # As a last resort, unify aliases then intersect again
            adata = unify_gene_names(adata, 'human' if species_key.lower() == 'human' else 'mouse')
            var_list = list(map(str, adata.var_names.astype(str)))
            var_upper_to_orig = {g.upper(): g for g in var_list}
            inter_upper = hk_upper.intersection(set(var_upper_to_orig.keys()))

        if len(inter_upper) == 0:
            # Return empty gene space to allow caller to skip this sample gracefully
            return adata[:, []].copy()

        present_orig = sorted([var_upper_to_orig[u] for u in inter_upper])
        return adata[:, present_orig].copy()


def map_ensembl_to_gene_symbols(adata: sc.AnnData, species_key: str) -> sc.AnnData:
    """Replace var_names from Ensembl IDs to gene symbols using scanpy biomart annotations.
    Keeps original names when mapping is missing. Deduplicates by keeping first occurrence.
    """
    import scanpy as sc
    org = 'hsapiens' if species_key.lower() == 'human' else 'mmusculus'
    ann = sc.queries.biomart_annotations(org=org, attrs=['ensembl_gene_id', 'external_gene_name'], use_cache=True)
    # include entries without version suffix
    ens_to_sym = dict(zip(ann['ensembl_gene_id'].astype(str), ann['external_gene_name'].astype(str)))
    var = adata.var_names.astype(str)
    new_names = []
    for g in var:
        base = g.split('.')[0]
        sym = ens_to_sym.get(g)
        if sym is None:
            sym = ens_to_sym.get(base)
        new_names.append(sym if sym is not None else g)
    adata = adata.copy()
    adata.var_names = new_names
    # Deduplicate
    mask = ~pd.Index(adata.var_names).duplicated(keep='first')
    adata = adata[:, mask]
    return adata


class HESTPatchSTDataset(Dataset):
    """
    Dataset that yields (patch, expression) pairs for a set of HEST samples.
    - Uses HEST-generated patch .h5 files (via dump_patches)
    - Expressions are taken from preprocessed sc.AnnData aligned on common genes
    - Builds a global index of selected barcodes per sample, up to max_patches_per_sample
    """

    def __init__(
        self,
        sample_ids: List[str],
        patch_paths: Dict[str, str],
        adatas_by_id: Dict[str, sc.AnnData],
        transform,
        max_patches_per_sample: int = -1,
    ):
        self.sample_ids = sample_ids
        self.patch_paths = patch_paths
        self.adatas_by_id = adatas_by_id
        self.transform = transform
        self.max_patches_per_sample = max_patches_per_sample

        # Precompute expression dictionaries and indices
        self._expr_by_id: Dict[str, Dict[str, np.ndarray]] = {}
        self._gene_dim: Optional[int] = None
        for sid in self.sample_ids:
            ad = self.adatas_by_id[sid]
            expr_df = ad.to_df()
            # Ensure dense, consistent dtype
            expr_df = expr_df.astype(np.float32)
            expr_dict = {barcode: expr_df.loc[barcode].values for barcode in expr_df.index}
            self._expr_by_id[sid] = expr_dict
            if self._gene_dim is None:
                self._gene_dim = expr_df.shape[1]

        # Build global indices across samples pointing to patch row indices
        self._global_indices: List[Tuple[str, int]] = []
        rng = np.random.default_rng(42)
        for sid in self.sample_ids:
            h5_path = self.patch_paths[sid]
            with h5py.File(h5_path, "r") as f:
                barcodes_ds = f["barcode"]
                num = barcodes_ds.shape[0]
                # Identify valid indices whose barcode exists in filtered adata
                valid_local = []
                for i in range(num):
                    bc_raw = barcodes_ds[i]
                    # barcodes saved as fixed-length bytes or array of bytes
                    if isinstance(bc_raw, (np.ndarray, list)):
                        bc_raw = bc_raw[0]
                    barcode = bc_raw.decode() if isinstance(bc_raw, (bytes, bytearray)) else str(bc_raw)
                    if barcode in self._expr_by_id[sid]:
                        valid_local.append(i)

                if self.max_patches_per_sample != -1 and len(valid_local) > self.max_patches_per_sample:
                    valid_local = rng.choice(valid_local, size=self.max_patches_per_sample, replace=False).tolist()

                self._global_indices.extend([(sid, li) for li in valid_local])

        rng.shuffle(self._global_indices)

        # Lazy-open H5 files per process
        self._h5_files: Dict[str, h5py.File] = {}
        self._pid: Optional[int] = None

    def __len__(self) -> int:
        return len(self._global_indices)

    def _ensure_local_files(self):
        pid = os.getpid()
        if self._pid != pid:
            # New worker or forked process; reopen files
            for f in self._h5_files.values():
                try:
                    f.close()
                except Exception:
                    pass
            self._h5_files = {sid: h5py.File(self.patch_paths[sid], "r") for sid in self.sample_ids}
            self._pid = pid

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._ensure_local_files()
        sid, local_idx = self._global_indices[idx]
        f = self._h5_files[sid]

        # Load image as HxWxC uint8
        img = f["img"][local_idx]
        # Decode barcode
        bc_raw = f["barcode"][local_idx]
        if isinstance(bc_raw, (np.ndarray, list)):
            bc_raw = bc_raw[0]
        barcode = bc_raw.decode() if isinstance(bc_raw, (bytes, bytearray)) else str(bc_raw)

        # Expression vector
        expr_np = self._expr_by_id[sid][barcode]

        # To PIL -> transform -> tensor CHW float
        # timm transforms typically accept PIL or torch tensors; use PIL for safety
        from PIL import Image

        img_pil = Image.fromarray(img)
        img_tensor = self.transform(img_pil)
        expr_tensor = torch.tensor(expr_np, dtype=torch.float32)

        return {
            "patch": img_tensor,
            "st_expression": expr_tensor,
            "sample_id": sid,
            "barcode": barcode,
        }


class UNI2hSTPredictor(nn.Module):
    def __init__(self, backbone_name: str, st_dim: int, dropout: float = 0.1):
        super().__init__()
        # UNI2-h from HF Hub via timm
        # num_classes=0 => returns features
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            img_size=224,
        )
        backbone_dim = getattr(self.backbone, "num_features", 1536)

        self.regression_head = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Dropout(dropout),
            nn.Linear(backbone_dim, backbone_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(backbone_dim // 2, st_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast():
            feats = self.backbone(x)
        out = self.regression_head(feats)
        return out


def apply_lora_to_backbone(backbone: nn.Module, r: int, alpha: int, dropout: float) -> nn.Module:
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["qkv", "proj"],
        bias="none",
    )
    peft_model = get_peft_model(backbone, lora_cfg)
    peft_model.print_trainable_parameters()
    return peft_model


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, epoch, use_wandb: bool):
    model.train()
    running = 0.0
    num = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    for step, batch in enumerate(pbar):
        imgs = batch["patch"].to(device, non_blocking=True)
        targets = batch["st_expression"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            preds = model(imgs)
            # Align dims if slight mismatch
            if preds.shape[1] != targets.shape[1]:
                m = min(preds.shape[1], targets.shape[1])
                preds = preds[:, :m]
                targets = targets[:, :m]
            loss = F.mse_loss(preds, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running += loss.item()
        num += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running/num:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        if use_wandb and (step % 20 == 0):
            wandb.log({
                "train/loss": loss.item(),
                "train/lr": scheduler.get_last_lr()[0],
                "train/step": epoch + step / max(1, len(loader)),
            })
    return running / max(1, num)


def evaluate(model, loader, device, use_wandb: bool, epoch: int):
    model.eval()
    losses = []
    preds_all = []
    tgts_all = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            imgs = batch["patch"].to(device, non_blocking=True)
            targets = batch["st_expression"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                preds = model(imgs)
                if preds.shape[1] != targets.shape[1]:
                    m = min(preds.shape[1], targets.shape[1])
                    preds = preds[:, :m]
                    targets = targets[:, :m]
                loss = F.mse_loss(preds, targets)
            losses.append(loss.item())
            preds_all.append(preds.detach().cpu())
            tgts_all.append(targets.detach().cpu())

    if len(losses) == 0:
        return {"loss": float("inf"), "r2": -float("inf"), "mse": float("inf"), "mean_gene_correlation": 0.0}

    preds_all = torch.cat(preds_all, dim=0).numpy()
    tgts_all = torch.cat(tgts_all, dim=0).numpy()

    from sklearn.metrics import mean_squared_error, r2_score

    mse = mean_squared_error(tgts_all, preds_all)
    r2 = r2_score(tgts_all, preds_all)
    # gene-wise Pearson
    corrs = []
    for g in range(tgts_all.shape[1]):
        y = tgts_all[:, g]
        x = preds_all[:, g]
        if np.std(y) > 0 and np.std(x) > 0:
            c = np.corrcoef(y, x)[0, 1]
            if not np.isnan(c):
                corrs.append(c)
    mgc = float(np.mean(corrs)) if len(corrs) > 0 else 0.0

    metrics = {
        "loss": float(np.mean(losses)),
        "mse": float(mse),
        "r2": float(r2),
        "mean_gene_correlation": mgc,
    }
    if use_wandb:
        wandb.log({f"val/{k}": v for k, v in metrics.items()})
        wandb.log({"epoch": epoch})
    return metrics


def setup_distributed(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def main_worker(rank: int, world_size: int, args):
    if world_size > 1:
        setup_distributed(rank, world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    use_wb = False
    if args.use_wandb and rank == 0:
        use_wb = setup_wandb("uni2h-hest-peft", args)

    # If using benchmark-style loader, go through a different path modeled on HEST's benchmark
    if args.use_bench_loader:
        # Prepare sample ids (rank 0) and shard across ranks
        if rank == 0:
            meta_df = read_hest_metadata(args.assets_dir, args.data_root)
            sample_ids = select_samples(meta_df, args.technology, args.species, args.max_samples)
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, "bench_sample_ids.json"), "w") as f:
                json.dump(sample_ids, f)
        if world_size > 1:
            dist.barrier()
        if rank != 0:
            with open(os.path.join(args.output_dir, "bench_sample_ids.json"), "r") as f:
                sample_ids = json.load(f)

        # Shard samples by rank
        sample_ids = sample_ids[rank::world_size]

        # Determine global gene list for consistent output dimension
        def read_var_names(h5ad_path: str) -> List[str]:
            ad = sc.read_h5ad(h5ad_path, backed='r')
            names = list(map(str, ad.var_names))
            try:
                ad.file.close()
            except Exception:
                pass
            return names

        global_genes_path = os.path.join(args.output_dir, "bench_genes.json")
        if rank == 0:
            if args.gene_list is not None and os.path.exists(args.gene_list):
                with open(args.gene_list, 'r') as f:
                    gg = json.load(f)
                if not isinstance(gg, list):
                    raise ValueError("--gene_list must be a JSON list of gene symbols")
                global_genes = list(map(str, gg))
            else:
                # Compute intersection across all selected samples (pre-sharding so all ranks share the same list)
                meta_df_all = read_hest_metadata(args.assets_dir, args.data_root)
                all_ids = select_samples(meta_df_all, args.technology, args.species, args.max_samples)
                common_set = None
                for sid in all_ids:
                    vn = set(read_var_names(os.path.join(args.data_root, 'st', f'{sid}.h5ad')))
                    common_set = vn if common_set is None else (common_set & vn)
                if not common_set:
                    raise ValueError("No common genes across selected samples.")
                global_genes = sorted(list(common_set))
                if args.target_gene_count is not None and len(global_genes) > args.target_gene_count:
                    global_genes = global_genes[:args.target_gene_count]
            with open(global_genes_path, 'w') as f:
                json.dump(global_genes, f)
        if world_size > 1:
            dist.barrier()
        with open(global_genes_path, 'r') as f:
            global_genes = json.load(f)
        st_dim = len(global_genes)

        # Transforms
        tmp_model = timm.create_model(args.model_name, pretrained=True, num_classes=0, img_size=224)
        data_cfg = resolve_data_config(getattr(tmp_model, "pretrained_cfg", None), model=tmp_model)
        transform = create_transform(**data_cfg)
        del tmp_model

        # Model
        model = UNI2hSTPredictor(backbone_name=args.model_name, st_dim=st_dim, dropout=args.dropout)
        model.backbone = apply_lora_to_backbone(model.backbone, args.lora_r, args.lora_alpha, args.lora_dropout)
        model = model.to(device)
        if world_size > 1:
            model = DDP(model, device_ids=[rank])

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
        # Approximate total steps: sum of ceil(num_patches/bs) over samples. Use a proxy if unknown.
        total_steps = max(1, 1000) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, int(0.1 * total_steps)), num_training_steps=total_steps)
        scaler = torch.cuda.amp.GradScaler()

        def pad_targets(x: torch.Tensor, dim: int) -> torch.Tensor:
            if x.shape[1] == dim:
                return x
            if x.shape[1] > dim:
                return x[:, :dim]
            pad = torch.zeros((x.shape[0], dim - x.shape[1]), dtype=x.dtype, device=x.device)
            return torch.cat([x, pad], dim=1)

        # Ensure patches for all samples (rank 0 on full list)
        if rank == 0 and args.prepare_patches:
            meta_df_all = read_hest_metadata(args.assets_dir, args.data_root)
            all_ids = select_samples(meta_df_all, args.technology, args.species, args.max_samples)
            patch_root = os.path.join(args.output_dir, 'patches')
            os.makedirs(patch_root, exist_ok=True)
            for sid in all_ids:
                try:
                    st = hest.load_hest(args.data_root, id_list=[sid])[0]
                    ensure_patches(st, patch_root, args.patch_size, args.patch_pixel_size_um)
                except Exception:
                    continue
        if world_size > 1:
            dist.barrier()

        # Train epochs by iterating samples and using H5HESTDataset chunking
        for epoch in range(args.epochs):
            model.train()
            running = 0.0
            steps = 0
            for sid in sample_ids:
                patch_h5 = os.path.join(args.output_dir, 'patches', f'{sid}.h5')
                if not os.path.exists(patch_h5):
                    # If not generated, fall back to generating now on this rank
                    try:
                        st = hest.load_hest(args.data_root, id_list=[sid])[0]
                        os.makedirs(os.path.join(args.output_dir, 'patches'), exist_ok=True)
                        ensure_patches(st, os.path.join(args.output_dir, 'patches'), args.patch_size, args.patch_pixel_size_um)
                    except Exception:
                        continue
                expr_h5ad = os.path.join(args.data_root, 'st', f'{sid}.h5ad')

                tile_dataset = H5HESTDataset(patch_h5, chunk_size=args.batch_size)
                tile_loader = DataLoader(tile_dataset, batch_size=1, shuffle=False, num_workers=0)

                # Load full expressions once per sample
                expr_df = load_adata(expr_h5ad, normalize=args.log1p)
                expr_df.index = expr_df.index.astype(str)
                # Align to global gene order, fill missing genes with 0
                expr_df = expr_df.reindex(columns=global_genes, fill_value=0.0)

                for chunk in tile_loader:
                    imgs_np = chunk['imgs'].squeeze(0)
                    if hasattr(imgs_np, 'numpy'):
                        imgs_np = imgs_np.numpy()
                    barcodes_arr = chunk['barcodes']
                    # decode barcodes
                    barcodes = []
                    for b in barcodes_arr:
                        b0 = b[0] if isinstance(b, (list, np.ndarray)) else b
                        barcodes.append(b0.decode() if isinstance(b0, (bytes, bytearray)) else str(b0))

                    # Select expressions, drop missing barcodes
                    sel = expr_df.reindex(barcodes)
                    sel = sel.dropna(axis=0, how='any')
                    if len(sel) == 0:
                        continue
                    # Filter images/barcodes to those present
                    present_mask = [bc in sel.index for bc in barcodes]
                    imgs_np = imgs_np[present_mask]
                    # Transform images
                    from PIL import Image
                    imgs_t = torch.stack([transform(Image.fromarray(img)) for img in imgs_np]).to(device, non_blocking=True)

                    targets = torch.tensor(sel.values, dtype=torch.float32, device=device)

                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast():
                        preds = model(imgs_t)
                        # Ensure pred dim matches st_dim and reorder as global_genes
                        if preds.shape[1] != st_dim:
                            preds = preds[:, :st_dim] if preds.shape[1] > st_dim else torch.cat([preds, torch.zeros((preds.shape[0], st_dim - preds.shape[1]), device=preds.device, dtype=preds.dtype)], dim=1)
                        loss = F.mse_loss(preds, targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    running += loss.item()
                    steps += 1
                    if use_wb and steps % 20 == 0 and rank == 0:
                        wandb.log({"train/loss": running / max(1, steps), "epoch": epoch})

            if rank == 0:
                print(f"Epoch {epoch+1}: avg loss {running / max(1, steps):.4f}")

        if args.use_wandb and rank == 0 and wandb.run is not None:
            wandb.finish()
        if world_size > 1:
            cleanup_distributed()
        return

    # Work dir for shared preprocessed artifacts
    shared_dir = os.path.join(args.output_dir, "preprocessed")
    os.makedirs(shared_dir, exist_ok=True)
    expr_dir = os.path.join(shared_dir, "expr")
    os.makedirs(expr_dir, exist_ok=True)

    try:
        # 1) Select samples from metadata (rank 0)
        if rank == 0:
            meta_df = read_hest_metadata(args.assets_dir, args.data_root)
            sample_ids = select_samples(meta_df, args.technology, args.species, args.max_samples)
            with open(os.path.join(shared_dir, "sample_ids.json"), "w") as f:
                json.dump(sample_ids, f)
        if world_size > 1:
            dist.barrier()
        if rank != 0:
            with open(os.path.join(shared_dir, "sample_ids.json"), "r") as f:
                sample_ids = json.load(f)

        # 2) Load samples and preprocess ONLY on rank 0; save per-sample expressions to parquet
        if rank == 0:
            print(f"Loading {len(sample_ids)} samples via HEST...")
            st_list: List[hest.HESTData] = hest.load_hest(args.data_root, id_list=sample_ids)
            id_to_hest: Dict[str, hest.HESTData] = {st.meta["id"]: st for st in st_list}

            common_gene_set: Optional[set] = None
            kept_ids: List[str] = []
            for sid in sample_ids:
                st = id_to_hest[sid]
                species_key = "human" if st.meta.get("species", "Homo sapiens") == "Homo sapiens" else "mouse"

                if args.use_stromal:
                    ad = filter_stromal_housekeeping(
                        st,
                        species=species_key,
                        whole_tissue=not args.only_stroma,
                        unify_genes=args.unify_genes,
                        verbose=False,
                    )
                else:
                    ad = safe_filter_housekeeping(st.adata, species_key, unify=args.unify_genes, assets_dir=args.assets_dir)

                if args.log1p:
                    sc.pp.log1p(ad)

                if ad.shape[1] == 0:
                    print(f"[rank0] Skipping sample {sid}: no housekeeping genes found after normalization/mapping.")
                    continue
                genes = list(map(str, ad.var_names.tolist()))
                common_gene_set = set(genes) if common_gene_set is None else (common_gene_set & set(genes))
                kept_ids.append(sid)

            if not kept_ids:
                raise ValueError("All samples were filtered out: no housekeeping overlap. Check species/genes.")
            sample_ids = kept_ids
            with open(os.path.join(shared_dir, "sample_ids.json"), "w") as f:
                json.dump(sample_ids, f)
            if not common_gene_set or len(common_gene_set) == 0:
                raise ValueError("No common genes across kept samples after preprocessing.")
            common_genes = sorted(list(common_gene_set))
            with open(os.path.join(shared_dir, "gene_order.json"), "w") as f:
                json.dump(common_genes, f)

            # Save aligned expressions to parquet per kept sample
            for sid in sample_ids:
                st = id_to_hest[sid]
                species_key = "human" if st.meta.get("species", "Homo sapiens") == "Homo sapiens" else "mouse"
                if args.use_stromal:
                    ad = filter_stromal_housekeeping(
                        st,
                        species=species_key,
                        whole_tissue=not args.only_stroma,
                        unify_genes=args.unify_genes,
                        verbose=False,
                    )
                else:
                    ad = safe_filter_housekeeping(st.adata, species_key, unify=args.unify_genes, assets_dir=args.assets_dir)
                if args.log1p:
                    sc.pp.log1p(ad)
                ad = ad[:, common_genes]
                df = ad.to_df()
                df.to_parquet(os.path.join(expr_dir, f"{sid}.parquet"))

            # Optional batch correction across samples
            if args.batch_correction != "none":
                print(f"Applying batch correction: {args.batch_correction}")
                adatas = []
                for sid in sample_ids:
                    df = pd.read_parquet(os.path.join(expr_dir, f"{sid}.parquet"))
                    ad = sc.AnnData(df.values)
                    ad.var_names = list(df.columns)
                    ad.obs_names = list(df.index)
                    adatas.append(ad)
                adatas = correct_batch_effect(adatas, method=args.batch_correction)
                for sid, ad in zip(sample_ids, adatas):
                    pd.DataFrame(ad.X, index=ad.obs_names, columns=ad.var_names).to_parquet(os.path.join(expr_dir, f"{sid}.parquet"))

            # Ensure patches exist (rank 0)
            patch_dir = os.path.join(args.output_dir, "patches")
            os.makedirs(patch_dir, exist_ok=True)
            if args.prepare_patches:
                print("Ensuring patch files exist (this may take time on first run)...")
            for sid in sample_ids:
                if args.prepare_patches:
                    ensure_patches(id_to_hest[sid], patch_dir, args.patch_size, args.patch_pixel_size_um)
            with open(os.path.join(shared_dir, "patch_dir.json"), "w") as f:
                json.dump({"patch_dir": patch_dir}, f)

        # Sync before loading artifacts on other ranks
        if world_size > 1:
            dist.barrier()

        # Load shared artifacts
        with open(os.path.join(shared_dir, "sample_ids.json"), "r") as f:
            sample_ids = json.load(f)
        with open(os.path.join(shared_dir, "gene_order.json"), "r") as f:
            common_genes = json.load(f)
        with open(os.path.join(shared_dir, "patch_dir.json"), "r") as f:
            patch_dir = json.load(f)["patch_dir"]
        st_dim = len(common_genes)
        id_to_patch_path = {sid: os.path.join(patch_dir, f"{sid}.h5") for sid in sample_ids}

    except Exception as e:
        if rank == 0:
            with open(os.path.join(shared_dir, "error.json"), "w") as f:
                json.dump({"error": str(e)}, f)
        if world_size > 1:
            try:
                dist.barrier()
            except Exception:
                pass
        # Ensure clean shutdown
        if world_size > 1:
            cleanup_distributed()
        raise

    # 5) Transforms
    backbone_name = args.model_name
    # Create a temporary model to resolve transforms
    tmp_model = timm.create_model(backbone_name, pretrained=True, num_classes=0, img_size=224)
    data_cfg = resolve_data_config(getattr(tmp_model, "pretrained_cfg", None), model=tmp_model)
    transform = create_transform(**data_cfg)
    del tmp_model

    # 6) Split into train/val by samples
    # Simple 80/20 split on sample list, then dataset will sample patches per sample
    n = len(ordered_ids)
    n_val = max(1, int(math.ceil(0.2 * n)))
    val_ids = ordered_ids[:n_val]
    train_ids = ordered_ids[n_val:]
    if len(train_ids) == 0:
        train_ids = val_ids
        val_ids = []
    print(f"Train samples: {len(train_ids)}, Val samples: {len(val_ids)}")

    # Replace in-memory AnnData with per-sample parquet loaders to avoid reprocessing and reduce memory
    def load_expr_df(sid: str) -> pd.DataFrame:
        return pd.read_parquet(os.path.join(expr_dir, f"{sid}.parquet"))

    class HESTPatchSTDatasetFromParquet(HESTPatchSTDataset):
        def __init__(self, sample_ids, patch_paths, transform, max_patches_per_sample):
            # Build minimal adatas_by_id surrogate: dict of barcode -> np.ndarray from parquet on init
            adatas_by_id = {}
            for s in sample_ids:
                df = load_expr_df(s).astype(np.float32)
                ad = sc.AnnData(df.values)
                ad.var_names = list(df.columns)
                ad.obs_names = list(df.index)
                adatas_by_id[s] = ad
            super().__init__(sample_ids, patch_paths, adatas_by_id, transform, max_patches_per_sample)

    train_ds = HESTPatchSTDatasetFromParquet(
        sample_ids=train_ids,
        patch_paths={sid: id_to_patch_path[sid] for sid in train_ids},
        transform=transform,
        max_patches_per_sample=args.max_patches_per_sample,
    )
    val_base_ids = val_ids if len(val_ids) > 0 else train_ids
    val_ds = HESTPatchSTDatasetFromParquet(
        sample_ids=val_base_ids,
        patch_paths={sid: id_to_patch_path[sid] for sid in val_base_ids},
        transform=transform,
        max_patches_per_sample=min(args.max_patches_per_sample, 512) if args.max_patches_per_sample != -1 else 512,
    )

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
        if world_size > 1
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, min(2, args.num_workers)),
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # 7) Model + LoRA
    model = UNI2hSTPredictor(backbone_name=backbone_name, st_dim=st_dim, dropout=args.dropout)
    # Apply LoRA on backbone only
    model.backbone = apply_lora_to_backbone(model.backbone, args.lora_r, args.lora_alpha, args.lora_dropout)
    model = model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    # 8) Optimizer & scheduler
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, int(0.1 * total_steps)), num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler()

    # 9) Train
    best_r2 = -float("inf")
    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch, use_wb)
        metrics = evaluate(model, val_loader, device, use_wb, epoch)
        if rank == 0:
            print(f"Epoch {epoch+1}: loss={metrics['loss']:.4f} r2={metrics['r2']:.4f} mgc={metrics['mean_gene_correlation']:.4f}")
            if metrics["r2"] > best_r2:
                best_r2 = metrics["r2"]
                model_to_save = model.module if hasattr(model, "module") else model
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                    "gene_order": common_genes,
                    "technology": args.technology,
                }, os.path.join(args.output_dir, f"best_model_{args.technology}.pth"))

    if args.use_wandb and rank == 0 and wandb.run is not None:
        wandb.finish()

    if world_size > 1:
        cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="UNI2-h LoRA training on HEST with official APIs")

    # Data
    parser.add_argument("--data_root", type=str, required=True, help="Path to HEST data root (contains st/wsis/metadata)")
    parser.add_argument("--assets_dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "assets"), help="Path to HEST metadata assets (CSV)")
    parser.add_argument("--technology", type=str, default="Visium", choices=["Visium", "Spatial Transcriptomics", "Xenium", "Visium HD"], help="ST technology to use")
    parser.add_argument("--species", type=str, default=None, choices=["Homo sapiens", "Mus musculus", None], help="Filter by species (optional)")
    parser.add_argument("--max_samples", type=int, default=50, help="Max number of samples (-1 for all)")
    parser.add_argument("--max_patches_per_sample", type=int, default=1000, help="Max patches per sample (-1 for all)")
    parser.add_argument("--prepare_patches", action="store_true", help="Generate patch h5 files if missing")
    parser.add_argument("--patch_size", type=int, default=224, help="Patch size (pixels)")
    parser.add_argument("--patch_pixel_size_um", type=float, default=0.5, help="Target patch pixel size in um/px")

    # Preprocessing
    parser.add_argument("--use_stromal", action="store_true", help="Filter to stromal housekeeping (requires cell segmentation)")
    parser.add_argument("--only_stroma", action="store_true", help="If set, keep only stromal regions (ignored if --use_stromal is False)")
    parser.add_argument("--unify_genes", action="store_true", help="Unify gene names before housekeeping filter")
    parser.add_argument("--log1p", action="store_true", help="Apply log1p to expressions")
    parser.add_argument("--batch_correction", type=str, default="none", choices=["none", "combat", "mnn", "harmony"], help="Batch correction method across samples")

    # Model
    parser.add_argument("--model_name", type=str, default="hf-hub:MahmoodLab/UNI2-h", help="timm model name for UNI2-h")
    parser.add_argument("--dropout", type=float, default=0.1)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)

    # Logging / output
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--use_wandb", action="store_true")

    # Multi-GPU
    parser.add_argument("--world_size", type=int, default=1)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"args_{args.technology}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.world_size > 1:
        torch.multiprocessing.spawn(main_worker, nprocs=args.world_size, args=(args.world_size, args))
    else:
        main_worker(0, 1, args)


if __name__ == "__main__":
    main()

