# generate embeddings for polymers by using the finetuned polyBERT model
import os, json, argparse, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

def mean_pool(last_hidden_state, attention_mask):
    m = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    return (last_hidden_state*m).sum(1)/m.sum(1).clamp(min=1e-9)

def load_reg_head(path, hidden, tasks):
    head = torch.nn.Linear(hidden, tasks)
    head.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return head

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="../data/all_combined.csv")
    ap.add_argument("--smiles_col", default="Smiles")
    ap.add_argument("--ckpt_dir", default="../outputs/polybert_lora")
    ap.add_argument("--out_npz", default="../outputs/language_embeddings.npz")
    ap.add_argument("--batch", type=int, default=256)
    return ap.parse_args()

@torch.no_grad()
def main():
    a = parse()
    df = pd.read_csv(a.data_path)
    tok = AutoTokenizer.from_pretrained(a.ckpt_dir) #best model loaded
    base = AutoModel.from_pretrained("kuelumbus/polyBERT", use_safetensors=True)
    enc = PeftModel.from_pretrained(base, a.ckpt_dir).eval().to("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(a.ckpt_dir,"target_stats.json")) as f:
        ts = json.load(f)
    targets, mu, sd = ts["targets"], ts["mean"], ts["std"]
    head = load_reg_head(os.path.join(a.ckpt_dir,"regression_head.pt"),
                         enc.base_model.config.hidden_size, len(targets)).eval().to(enc.device)

    smiles = df[a.smiles_col].astype(str).tolist()

    embs = []
    preds_scaled = []
    bs = a.batch
    for i in range(0, len(smiles), bs):
        bx = smiles[i:i+bs]
        encd = tok(bx, truncation=True, max_length=160, padding=True, return_tensors="pt").to(enc.device)
        out = enc(**encd)
        rep = mean_pool(out.last_hidden_state, encd["attention_mask"])
        embs.append(rep.cpu().numpy())
        y_hat = head(rep).cpu().numpy()
        preds_scaled.append(y_hat)

    Z = np.concatenate(embs, 0)                       # [N, H]
    Yz = np.concatenate(preds_scaled, 0)              # [N, 3] scaled
    # unscale predictions to original units (imputation values)
    Y = Yz * np.array(sd)[None,:] + np.array(mu)[None,:]

    np.savez(a.out_npz, id=df["id"].values, Smiles=np.array(smiles, dtype=object), z_text=Z, y_pred=Y)
    print("Language embeddings are saved")

if __name__ == "__main__":
    main()
