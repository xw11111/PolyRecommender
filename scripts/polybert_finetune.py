import os, json, math, argparse
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import r2_score
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, DataCollatorWithPadding, set_seed
from peft import LoraConfig, get_peft_model, TaskType

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
USE_BF16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../data/", help="Folder with train.csv, valid.csv, test.csv")
    ap.add_argument("--smiles_col", default="Smiles")
    ap.add_argument("--targets", default="Tg,Tm,Eg")
    ap.add_argument("--model_name", default="kuelumbus/polyBERT")
    ap.add_argument("--out_dir", default="../outputs/polybert_lora")
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bsz", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=998)
    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--fp16", action="store_true")
    return ap.parse_args()

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(1)
    denom = mask.sum(1).clamp(min=1e-9)
    return summed / denom

class PolyBERTRegressor(nn.Module):
    def __init__(self, enc, n_tasks, hidden_size):
        super().__init__()
        self.enc = enc
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(hidden_size, n_tasks)

    def forward(self, input_ids=None, attention_mask=None, labels=None, return_dict=True):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        x = mean_pool(out.last_hidden_state, attention_mask)
        x = self.dropout(x)
        preds = self.head(x)
        result = {"logits": preds}
        if labels is not None:
            mask = ~torch.isnan(labels)
            se = torch.zeros_like(preds)
            se[mask] = (preds[mask] - labels[mask])**2
            per_task_n = mask.sum(0).clamp(min=1)
            loss = (se.sum(0)/per_task_n).mean()
            result["loss"] = loss
        return result

def main():
    a = parse_args()
    set_seed(a.seed)

    targets = [t.strip() for t in a.targets.split(",") if t.strip()]

    # -------- Load train/valid/test CSVs
    df_tr = pd.read_csv(os.path.join(a.data_dir, "train.csv"))
    df_va = pd.read_csv(os.path.join(a.data_dir, "valid.csv"))
    df_te = pd.read_csv(os.path.join(a.data_dir, "test.csv"))
    y_test_unscaled = df_te[targets].astype("float32").values

    for name, df in [("train", df_tr), ("valid", df_va), ("test", df_te)]:
        print(f"{name} shape: {df.shape}")

    # -------- Standardization stats from TRAIN only
    stats = {"mean": [], "std": []}
    def scale_targets(df, fit=False):
        y = df[targets].astype("float32").copy()
        if fit:
            for t in targets:
                obs = y[t].dropna().values
                mu = float(obs.mean()) if len(obs) else 0.0
                sd = float(obs.std()) if len(obs) and obs.std() > 1e-8 else 1.0
                stats["mean"].append(mu); stats["std"].append(sd)
                m = y[t].notna()
                y.loc[m,t] = (y.loc[m,t] - mu)/sd
        else:
            for i,t in enumerate(targets):
                mu, sd = stats["mean"][i], stats["std"][i]
                m = y[t].notna()
                y.loc[m,t] = (y.loc[m,t] - mu)/sd
        return y

    y_tr = scale_targets(df_tr, fit=True).values
    y_va = scale_targets(df_va).values
    y_te = scale_targets(df_te).values

    # -------- Tokenizer & model
    tok = AutoTokenizer.from_pretrained(a.model_name)
    base = AutoModel.from_pretrained(a.model_name, use_safetensors=True)
    lora = LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["query_proj","key_proj","value_proj","o_proj"],
        bias="none"
    )
    enc = get_peft_model(base, lora)
    model = PolyBERTRegressor(enc, n_tasks=len(targets), hidden_size=base.config.hidden_size)

    def preprocess(smiles_series):
        return tok(smiles_series.tolist(), truncation=True, max_length=a.max_len, padding=False, return_token_type_ids=False)

    tr_enc = preprocess(df_tr[a.smiles_col])
    va_enc = preprocess(df_va[a.smiles_col])
    te_enc = preprocess(df_te[a.smiles_col])
    
    for enc in (tr_enc, va_enc, te_enc):
        enc.pop("token_type_ids", None)

    class DS(torch.utils.data.Dataset):
        def __init__(self, enc, labels):
            self.enc, self.labels = enc, labels
        def __len__(self): return len(self.enc["input_ids"])
        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k,v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[i], dtype=torch.float32)
            return item

    train_ds, val_ds, test_ds = DS(tr_enc, y_tr), DS(va_enc, y_va), DS(te_enc, y_te)
    collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds, labels = torch.tensor(preds), torch.tensor(labels)
        mask = ~torch.isnan(labels)
        out = {}
        for i,t in enumerate(targets):
            m = mask[:,i]
            if m.sum()==0: continue
            mu, sd = stats["mean"][i], stats["std"][i]
            y_true = labels[m,i]*sd + mu
            y_pred = preds[m,i]*sd + mu
            rmse = torch.sqrt(((y_pred - y_true)**2).mean()).item()
            r2 = r2_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
            out[f"rmse_{t}"] = rmse; out[f"r2_{t}"] = r2
        if any(k.startswith("rmse_") for k in out):
            out["rmse_avg"] = float(np.mean([v for k,v in out.items() if k.startswith("rmse_")]))
            out["r2_avg"]   = float(np.mean([v for k,v in out.items() if k.startswith("r2_")]))
        return out

    steps = max(1, math.ceil(len(train_ds)/a.bsz))
    args = TrainingArguments(
        output_dir=a.out_dir, per_device_train_batch_size=a.bsz, per_device_eval_batch_size=a.bsz,
        learning_rate=a.lr, num_train_epochs=a.epochs, warmup_ratio=a.warmup_ratio,
        weight_decay=a.weight_decay, eval_strategy="steps", logging_steps=100, eval_steps=steps,
        save_steps=steps, save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, tf32=True,
        dataloader_num_workers=4, dataloader_pin_memory=True, report_to="none", seed=a.seed
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=collator, processing_class=tok, compute_metrics=compute_metrics)

    trainer.train()
    # Log and save train metrics (on full training set)
    train_metrics = trainer.evaluate(eval_dataset=train_ds, metric_key_prefix="train")
    trainer.save_metrics("train", train_metrics)
    print("\nTrain metrics:")
    print({k: v for k, v in train_metrics.items() if k.startswith(("train_rmse","train_r2"))})

    # Best validation snapshot (lowest eval_loss)
    eval_logs = [log for log in trainer.state.log_history if "eval_loss" in log]
    if eval_logs:
        best_metrics = min(eval_logs, key=lambda x: x["eval_loss"])
        print("\nBEST VAL METRICS:")
        print(f"Epoch: {best_metrics.get('epoch', 'N/A')}")
        print(f"Step: {best_metrics.get('step', 'N/A')}")
        print(f"Val Loss: {best_metrics['eval_loss']:.4f}")
        print("\nVal RMSE:")
        for t in targets:
            if f"eval_rmse_{t}" in best_metrics:
                print(f"  {t}: {best_metrics[f'eval_rmse_{t}']:.2f}")
        if "eval_rmse_avg" in best_metrics:
            print(f"  Average: {best_metrics['eval_rmse_avg']:.2f}")
        print("\nVal R²:")
        for t in targets:
            if f"eval_r2_{t}" in best_metrics:
                print(f"  {t}: {best_metrics[f'eval_r2_{t}']:.3f}")
        if "eval_r2_avg" in best_metrics:
            print(f"  Average: {best_metrics['eval_r2_avg']:.3f}")

    # ---- Final test evaluation on held-out 10% ----
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    trainer.save_metrics("test", test_metrics)

    print("\n" + "="*50)
    print("FINAL TEST RESULTS:")
    print("\n" + "="*50)
    for t in targets:
        rmse_k, r2_k = f"test_rmse_{t}", f"test_r2_{t}"
        if rmse_k in test_metrics or r2_k in test_metrics:
            rmse = test_metrics.get(rmse_k, float("nan"))
            r2   = test_metrics.get(r2_k, float("nan"))
            print(f"{t}:  RMSE = {rmse:.2f}   R² = {r2:.3f}")
    if "test_rmse_avg" in test_metrics:
        print(f"Average RMSE: {test_metrics['test_rmse_avg']:.2f}")
    if "test_r2_avg" in test_metrics:
        print(f"Average R²:   {test_metrics['test_r2_avg']:.3f}")

    
    # Save per-sample TEST predictions (denormalized) with SMILES/ID
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Build a small dataloader to get predictions in order
    class PlainDS(torch.utils.data.Dataset):
        def __init__(self, enc):
            self.enc = enc
        def __len__(self): return len(self.enc["input_ids"])
        def __getitem__(self, i):
            return {k: torch.tensor(v[i]) for k,v in self.enc.items()}

    test_plain = PlainDS(te_enc)
    test_loader = torch.utils.data.DataLoader(test_plain, batch_size=a.bsz, shuffle=False, collate_fn=collator)

    preds = []
    with torch.no_grad():
        for batch in test_loader:
            for k in ["input_ids","attention_mask"]:
                batch[k] = batch[k].to(device)
            out = model(**batch)
            preds.append(out["logits"].cpu().numpy())
    preds = np.vstack(preds)  # scaled space

    # denormalize predictions target-wise
    for i,(mu,sdv) in enumerate(zip(stats["mean"], stats["std"])):
        preds[:,i] = preds[:,i]*sdv + mu

    out_df = pd.DataFrame({
        "id": df_te["id"].values,
        "Smiles": df_te[a.smiles_col].values
    })
    for i,t in enumerate(targets):
        out_df[f"y_true_{t}"] = y_test_unscaled[:,i]
        out_df[f"y_pred_{t}"] = preds[:,i]

    out_df.to_csv(os.path.join(a.out_dir, "test_predictions.csv"), index=False)
    print(f"\nSaved test predictions to: {os.path.join(a.out_dir, 'test_predictions.csv')}")

    # Save artifacts
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "target_stats.json"), "w") as f:
        json.dump({"targets": targets, **stats}, f, indent=2)
    trainer.save_model(a.out_dir)
    model.enc.save_pretrained(a.out_dir)
    tok.save_pretrained(a.out_dir)
    torch.save(model.head.state_dict(), os.path.join(a.out_dir,"regression_head.pt"))

    print("Finetuning completed successfully!")

if __name__ == "__main__":
    main()
