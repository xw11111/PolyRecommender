from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import List, Optional
import math
import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from chemprop import data, featurizers, models, nn
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

assert torch.cuda.is_available(), "CUDA GPU not detected"

try:
    torch.set_float32_matmul_precision("medium")
except Exception:
    pass

DEVICE = torch.device("cuda")


def parse_args():
    p = argparse.ArgumentParser(
        description="train Chemprop D-MPNN and export embeddings"
    )
    p.add_argument("--train", type=Path, default=Path("../data/train.csv"))
    p.add_argument("--val", type=Path, default=Path("../data/valid.csv"))
    p.add_argument("--test", type=Path, default=Path("../data/test.csv"))
    p.add_argument("--all", type=Path, default=Path("../data/all_combined.csv"))
    p.add_argument("--save_dir", type=Path, default=Path("../outputs/gnn/"))
    p.add_argument("--npz_out", type=Path, default=Path("../outputs/graph_embeddings.npz"))
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--mp_depth", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.1)
    return p.parse_args()


num_workers = 4


def _read_supervised_csv(
    csv_path: Path,
    smiles_col: str = "Smiles",
    target_cols: List[str] | None = None,
    id_col: str = "id",
):
    df = pd.read_csv(csv_path)
    if target_cols is None:
        target_cols = []
    
    required = [smiles_col] + ([id_col] if id_col in df.columns else [])
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {csv_path}")
    
    smiles = df[smiles_col].astype(str).tolist()
    ys = df[target_cols].astype(float).values if target_cols else None
    
    dps = []
    keep_mask = []
    for i, smi in enumerate(smiles):
        y = ys[i] if ys is not None else None
        dp = data.MoleculeDatapoint.from_smi(smi, y)
        dps.append(dp)
        keep_mask.append(True)
    
    df = df.loc[np.nonzero(keep_mask)[0]].reset_index(drop=True)
    return dps, df


def _build_loaders(
    train_dps: List[data.MoleculeDatapoint],
    val_dps: Optional[List[data.MoleculeDatapoint]] = None,
    test_dps: Optional[List[data.MoleculeDatapoint]] = None,
    num_workers: int = 4,
    batch_size: int = 64,
):
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dset = data.MoleculeDataset(train_dps, featurizer)
    scaler = train_dset.normalize_targets()
    
    val_loader = None
    if val_dps is not None:
        val_dset = data.MoleculeDataset(val_dps, featurizer)
        val_dset.normalize_targets(scaler)
        val_loader = data.build_dataloader(
            val_dset, shuffle=False, num_workers=num_workers, batch_size=batch_size
        )
    
    test_loader = None
    if test_dps is not None:
        test_dset = data.MoleculeDataset(test_dps, featurizer)
        test_dset.normalize_targets(scaler)
        test_loader = data.build_dataloader(
            test_dset, shuffle=False, num_workers=num_workers, batch_size=batch_size
        )
    
    train_loader = data.build_dataloader(
        train_dset, shuffle=True, num_workers=num_workers, batch_size=batch_size
    )
    
    return train_loader, val_loader, test_loader, scaler


def _build_model(
    scaler,
    hidden_dim: int = 300,
    n_layers: int = 1,
    dropout: float = 0.0,
    n_tasks: int = 3,
):
    mp = nn.BondMessagePassing(d_h=hidden_dim, depth=n_layers, dropout=dropout)
    agg = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(
        n_tasks=n_tasks,
        input_dim=mp.output_dim,
        hidden_dim=hidden_dim,
        n_layers=1,
        dropout=dropout,
        output_transform=output_transform,
        criterion=nn.metrics.MSE(),  # masked MSE for multi-task regression
    )
    metrics = [nn.metrics.MSE(), nn.metrics.MAE()]
    mpnn = models.MPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        batch_norm=True,
        metrics=metrics,
    )
    return mpnn


@torch.no_grad()
def evaluate_rmse_r2(dataloader, model, scaler, device, target_names):
    if dataloader is None:
        return None
    
    def to_numpy(x):
        import numpy as _np
        import torch as _torch
        if isinstance(x, _torch.Tensor):
            return x.detach().cpu().numpy()
        return _np.asarray(x)
    
    y_true_all, y_pred_all = [], []
    model.eval()
    for batch in dataloader:
        # move graph + optional features to device
        bmg = batch.bmg
        bmg.to(device)  # in-place
        V_d = batch.V_d.to(device) if batch.V_d is not None else None
        X_d = batch.X_d.to(device) if batch.X_d is not None else None
        
        y_hat = model(bmg, V_d, X_d)
        y_hat = to_numpy(y_hat)
        
        y_scaled = getattr(batch, "Y", None)
        if y_scaled is None:
            y_scaled = getattr(batch, "y")  # fallback if API changes
        y = scaler.inverse_transform(y_scaled)  # could be torch or numpy
        y = to_numpy(y)
        
        y_true_all.append(y)
        y_pred_all.append(y_hat)
    
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)
    
    rmse_per, r2_per = [], []
    for j in range(y_true_all.shape[1]):
        mask = ~np.isnan(y_true_all[:, j])
        yt = y_true_all[mask, j]
        yp = y_pred_all[mask, j]
        if yt.size == 0:
            rmse_per.append(np.nan)
            r2_per.append(np.nan)
            continue
        
        diff = yp - yt
        mse = float(np.mean(diff ** 2))
        rmse = mse ** 0.5
        sst = float(np.sum((yt - yt.mean()) ** 2))
        r2 = 1.0 - (float(np.sum(diff ** 2)) / sst) if sst > 0 else np.nan
        
        rmse_per.append(rmse)
        r2_per.append(r2)
    
    avg_rmse = float(np.nanmean(rmse_per))
    avg_r2 = float(np.nanmean(r2_per))
    
    return {
        "rmse_per": rmse_per,
        "r2_per": r2_per,
        "avg_rmse": avg_rmse,
        "avg_r2": avg_r2,
    }


def train_and_save(
    train_csv: Path,
    val_csv: Path,
    test_csv: Optional[Path],
    all_csv: Path,
    save_dir: Path,
    npz_out: Path,
    batch_size: int = 64,
    max_epochs: int = 100,
    patience: int = 20,
    hidden_dim: int = 300,
    mp_depth: int = 3,
    dropout: float = 0.0,
    num_workers: int = 4,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    npz_out.parent.mkdir(parents=True, exist_ok=True)
    
    targets = ["Tg", "Tm", "Eg"]
    
    train_dps, _ = _read_supervised_csv(train_csv, target_cols=targets)
    val_dps, _ = _read_supervised_csv(val_csv, target_cols=targets)
    
    test_dps = None
    if test_csv is not None and test_csv.exists():
        test_dps, _ = _read_supervised_csv(test_csv, target_cols=targets)
    
    train_loader, val_loader, test_loader, scaler = _build_loaders(
        train_dps, val_dps, test_dps, num_workers=num_workers, batch_size=batch_size
    )
    
    mpnn = _build_model(
        scaler=scaler,
        hidden_dim=hidden_dim,
        n_layers=mp_depth,
        dropout=dropout,
        n_tasks=len(targets),
    )
    
    ckpt_cb = ModelCheckpoint(
        dirpath=str(save_dir),
        filename="best-{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    
    es_cb = EarlyStopping(monitor="val_loss", mode="min", patience=patience)
    
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True,
        callbacks=[ckpt_cb, es_cb],
        enable_progress_bar=True,
        accelerator="gpu",
        devices=1,
        max_epochs=max_epochs,
        gradient_clip_val=None,
    )
    
    trainer.fit(mpnn, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    best_ckpt_path = ckpt_cb.best_model_path or (Path(save_dir) / "last.ckpt")
    best_model = models.MPNN.load_from_checkpoint(best_ckpt_path)
    best_model.to(DEVICE).eval()
    
    if test_loader is not None:
        trainer.test(dataloaders=test_loader, ckpt_path="best")
    
    targets = ["Tg", "Tm", "Eg"]
    print("\n" + "="*50)
    print("FINAL TEST RESULTS:")
    print("\n" + "="*50)
    
    test_metrics = evaluate_rmse_r2(test_loader, best_model, scaler, DEVICE, targets)
    for name, r2, rm in zip(targets, test_metrics["r2_per"], test_metrics["rmse_per"]):
        print(f"{name}: R2: {r2:.4f} RMSE: {rm:.6g}")
    print(f"Average: R2: {test_metrics['avg_r2']:.4f} RMSE: {test_metrics['avg_rmse']:.6g}")
    
    all_dps, df_all = _read_supervised_csv(all_csv, target_cols=None)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    all_dset = data.MoleculeDataset(all_dps, featurizer)
    all_loader = data.build_dataloader(
        all_dset,
        shuffle=False,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=True,
    )
    
    preds, zs = [], []
    with torch.no_grad():
        for batch in all_loader:
            bmg = batch.bmg; bmg.to(DEVICE)  # in-place
            V_d = batch.V_d.to(DEVICE) if batch.V_d is not None else None
            X_d = batch.X_d.to(DEVICE) if batch.X_d is not None else None
            
            z = best_model.fingerprint(bmg, V_d)  # [B, hidden]
            y_hat = best_model(bmg, V_d, X_d)  # [B, n_tasks]
            
            zs.append(z.cpu()); preds.append(y_hat.cpu())
    
    z_graph = torch.cat(zs, dim=0).numpy()
    y_pred = torch.cat(preds, dim=0).numpy()
    
    ids = df_all["id"].to_numpy() if "id" in df_all.columns else np.arange(len(df_all))
    smiles = df_all["Smiles"].astype(str).to_numpy()
    
    np.savez_compressed(
        npz_out,
        id=ids,
        Smiles=smiles,
        z_graph=z_graph,
        y_pred=y_pred,
    )
    
    print(f"Graph embeddings are saved")


if __name__ == "__main__":
    args = parse_args()
    train_and_save(
        train_csv=args.train,
        val_csv=args.val,
        test_csv=args.test,
        all_csv=args.all,
        save_dir=args.save_dir,
        npz_out=args.npz_out,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        hidden_dim=args.hidden_dim,
        mp_depth=args.mp_depth,
        dropout=args.dropout,
        num_workers=num_workers,
    )