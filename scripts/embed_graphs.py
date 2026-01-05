import argparse, numpy as np, torch
from pathlib import Path
from chemprop import data, featurizers, models
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    p = argparse.ArgumentParser(description="Export graph embeddings from a trained Chemprop model")
    p.add_argument("--ckpt", type=Path, default=Path("../outputs/gnn/best-epoch=73-val_loss=0.1287.ckpt"),
                   help="Path to the trained checkpoint file")
    p.add_argument("--data", type=Path, default=Path("../data/all_combined.csv"))
    p.add_argument("--npz_out", type=Path, default=Path("../outputs/graph_embeddings.npz"))
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    args.npz_out.parent.mkdir(parents=True, exist_ok=True)

    # --- load the trained model directly ---
    print(f"Loading checkpoint: {args.ckpt}")
    model = models.MPNN.load_from_checkpoint(str(args.ckpt))
    model.to(DEVICE).eval()

    # --- load all data (for embeddings only, no targets) ---
    df_all = pd.read_csv(args.all)
    smiles = df_all["Smiles"].astype(str).tolist()
    dps = [data.MoleculeDatapoint.from_smi(s, None) for s in smiles]

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    all_dset = data.MoleculeDataset(dps, featurizer)
    all_loader = data.build_dataloader(
        all_dset, shuffle=False, num_workers=args.num_workers,
        batch_size=args.batch_size, pin_memory=True
    )

    # --- collect embeddings & predictions ---
    zs, preds = [], []
    for batch in all_loader:
        bmg = batch.bmg; bmg.to(DEVICE)
        V_d = batch.V_d.to(DEVICE) if batch.V_d is not None else None
        X_d = batch.X_d.to(DEVICE) if batch.X_d is not None else None
        z = model.fingerprint(bmg, V_d)   # embeddings
        y = model(bmg, V_d, X_d)          # predictions
        zs.append(z.cpu()); preds.append(y.cpu())

    z_graph = torch.cat(zs, dim=0).numpy().astype("float32")
    y_pred  = torch.cat(preds, dim=0).numpy().astype("float32")

    ids = df_all["id"].to_numpy() if "id" in df_all.columns else np.arange(len(df_all))
    smiles = df_all["Smiles"].astype(str).to_numpy()

    # sort by id for consistency
    order = np.argsort(ids)
    ids, smiles, z_graph, y_pred = ids[order], smiles[order], z_graph[order], y_pred[order]

    np.savez_compressed(args.npz_out, id=ids, Smiles=smiles, z_graph=z_graph, y_pred=y_pred)
    print(f"Embeddings saved to {args.npz_out}")

if __name__ == "__main__":
    main()
