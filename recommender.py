# Polymer similarity search using FAISS index with TAPD relevancy scoring.
# Finds similar polymers based on language embeddings and ranks by property similarity.

import argparse
import numpy as np
import pandas as pd
import faiss
import torch
import torch.nn as nn
from pathlib import Path
import sys
import os
import sys
import importlib.util
import time
import json

stime = time.time()
def import_models():
    """Import model classes without executing the module's main code"""
    spec = importlib.util.spec_from_file_location("mmoe_train", "mmoe_train.py")
    mmoe_module = importlib.util.module_from_spec(spec)

    original_argv = sys.argv
    sys.argv = ['mmoe_train.py']

    try:
        spec.loader.exec_module(mmoe_module)
    finally:
        sys.argv = original_argv

    return (mmoe_module.ConcatRegressor, 
            mmoe_module.MoERegressor, 
            mmoe_module.TrueMMoERegressor, 
            mmoe_module.MLP)
def load_target_stats(train_csv, scaler_json=None, targets=("Tg","Tm","Eg")):
    # Prefer saved stats from training if you wrote them out
    if scaler_json and os.path.exists(scaler_json):
        with open(scaler_json) as f:
            ts = json.load(f)
        mu  = np.array([ts[t]["mean"] for t in targets], dtype=np.float32)
        std = np.array([ts[t]["std"]  for t in targets], dtype=np.float32)
        return mu, std

    # Fallback: recompute exactly like StandardScaler (ddof=0) on TRAIN ONLY
    train = pd.read_csv(train_csv)
    mu  = np.array([np.nanmean(train[t].values) for t in targets], dtype=np.float32)
    std = np.array([max(np.nanstd(train[t].values, ddof=0), 1e-8) for t in targets], dtype=np.float32)
    return mu, std


ConcatRegressor, MoERegressor, TrueMMoERegressor, MLP = import_models()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_fusion_model(model_path, lang_dim=600, graph_dim=512, hidden_dim1=256, hidden_dim2=128, output_dim=3):
    model_path = Path(model_path)
    model_name = model_path.stem.replace("_best", "")
    
    if model_name == "Concat":
        model = ConcatRegressor(lang_dim, graph_dim, n_tasks=3, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, dropout=0.4)
    elif model_name == "Binary_MoE":
        model = MoERegressor(lang_dim, graph_dim, n_tasks=3, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, dropout=0.4)
    elif model_name == "MMoE":
        model = TrueMMoERegressor(lang_dim, graph_dim, n_tasks=3, n_experts=4, expert_hidden_dim=256, hidden_dim=256, dropout=0.4)
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    
    # Load weights with weights_only=True for security
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    return model

def calculate_tapd(y_pred, y_target):
    with np.errstate(divide='ignore', invalid='ignore'):
        tapd = np.sum(np.abs(y_pred - y_target) / np.abs(y_target), axis=1)
        # Handle division by zero or invalid values
        tapd = np.where(np.isfinite(tapd), tapd, np.inf)
    return tapd

def parse_args():
    parser = argparse.ArgumentParser(description="Polymer similarity search with relevancy")
    parser.add_argument("--query_smiles", required=True)
    parser.add_argument("--data_path", default="../data/all_combined.csv")
    parser.add_argument("--lang_emb_path", default="../outputs/language_embeddings.npz")
    parser.add_argument("--graph_emb_path", default="../outputs/graph_embeddings.npz") 
    parser.add_argument("--faiss_index", default="../outputs/faiss_polybert_cosine.index")
    parser.add_argument("--fusion_model", default="MMoE", choices=["MMoE", "Binary_MoE", "Concat"])
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--num_return", type=int, default=10)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    if args.fusion_model == "MMoE":
        model_pth = "../outputs/fusion_results/MMoE_best.pth"
    elif args.fusion_model == "Binary_MoE":
        model_pth = "../outputs/fusion_results/Binary_MoE_best.pth"
    else:
        model_pth = "../outputs/fusion_results/Concat_best.pth"
    

    df = pd.read_csv(args.data_path)
    property_names = ["Tg","Tm","Eg"]
    # means/stds from training (JSON if saved; else recompute from train.csv)
    mu, std = load_target_stats("../data/train.csv",
    scaler_json="../outputs/fusion_results/target_scalers.json",
    targets=property_names)

    lang_data = np.load(args.lang_emb_path, allow_pickle=True)
    lang_ids = lang_data["id"]
    lang_embeddings = lang_data["z_text"]

    graph_data = np.load(args.graph_emb_path, allow_pickle=True)
    graph_ids = graph_data["id"]
    graph_embeddings = graph_data["z_graph"]
    

    faiss_index = faiss.read_index(args.faiss_index)
    
    query_mask = df["Smiles"] == args.query_smiles
    query_row = df[query_mask].iloc[0]
    query_id = query_row["id"]
    
    query_lang_idx = np.where(lang_ids == query_id)[0]
    query_graph_idx = np.where(graph_ids == query_id)[0]
    
    if len(query_lang_idx) == 0 or len(query_graph_idx) == 0:
        raise ValueError(f"Embeddings not found for query ID: {query_id}")
    
    query_lang_emb = lang_embeddings[query_lang_idx[0]:query_lang_idx[0]+1]
    query_graph_emb = graph_embeddings[query_graph_idx[0]:query_graph_idx[0]+1]
    
    # Search similar polymers using FAISS
    similarities, candidate_indices = faiss_index.search(
        query_lang_emb.astype(np.float32), args.topk
    )

    candidate_ids = lang_ids[candidate_indices[0]]
    candidate_lang_embs = lang_embeddings[candidate_indices[0]]

    candidate_graph_embs = []
    valid_candidates = []
    
    for i, cand_id in enumerate(candidate_ids):
        graph_idx = np.where(graph_ids == cand_id)[0]
        if len(graph_idx) > 0:
            candidate_graph_embs.append(graph_embeddings[graph_idx[0]])
            valid_candidates.append(i)
    
    if len(valid_candidates) == 0:
        raise ValueError("No valid candidates found with both language and graph embeddings")

    candidate_ids = candidate_ids[valid_candidates]
    candidate_lang_embs = candidate_lang_embs[valid_candidates]
    candidate_graph_embs = np.array(candidate_graph_embs)
    
    # Load fusion model
    model = load_fusion_model(
        model_pth, 
        lang_dim=candidate_lang_embs.shape[1],
        graph_dim=candidate_graph_embs.shape[1]
    )
    
    query_lang_tensor = torch.tensor(query_lang_emb, dtype=torch.float32)
    query_graph_tensor = torch.tensor(query_graph_emb, dtype=torch.float32)
    candidate_lang_tensor = torch.tensor(candidate_lang_embs, dtype=torch.float32)
    candidate_graph_tensor = torch.tensor(candidate_graph_embs, dtype=torch.float32)
    
    with torch.no_grad():
        query_output = model(query_lang_tensor, query_graph_tensor)
        candidate_outputs = model(candidate_lang_tensor, candidate_graph_tensor)
        
        query_pred = query_output["logits"].numpy()
        candidate_preds = candidate_outputs["logits"].numpy()
        
    query_pred_phys      = query_pred * std + mu
    candidate_preds_phys = candidate_preds * std + mu
    tapd_scores = calculate_tapd(candidate_preds_phys, query_pred_phys)
    # tapd_scores = calculate_tapd(candidate_preds, query_pred)
    relevancy_scores = 1.0 / (tapd_scores + 1) * 100
    sorted_indices = np.argsort(relevancy_scores)[::-1]
    property_names = ["Tg", "Tm", "Eg"]
    
    print(f"\nQuery Polymer Properties:")
    print(f"SMILES: {args.query_smiles}")
    print(f"ID: {query_id}")
    for i, prop_name in enumerate(property_names):
        print(f"{prop_name}: {query_pred_phys[0, i]:.4f}")
    
    print(f"\nTop {args.num_return} Most Relevant Candidates:")
    print("-" * 100)
    # print(f"{'Rank':<4} {'ID':<8} {'Relevancy':<12} {'Tg':<10} {'Tm':<10} {'Eg':<10} {'SMILES':<30}")
    print(f"{'Rank':<4} {'ID':<8} {'SMILES':<30} {'Tg':<10} {'Tm':<10} {'Eg':<10} {'Relevancy (%)':<12} ")
    print("-" * 100)
    
    for rank, idx in enumerate(sorted_indices[:args.num_return]):
        cand_id = candidate_ids[idx]
        tapd = tapd_scores[idx]
        relevancy = relevancy_scores[idx]
        pred = candidate_preds_phys[idx]

        cand_smiles = df[df["id"] == cand_id]["Smiles"].iloc[0] if len(df[df["id"] == cand_id]) > 0 else "N/A"
        
        print(f"{rank+1:<4} {cand_id:<8} {cand_smiles[:30]:<30}"
              f"{pred[0]:<10.4f} {pred[1]:<10.4f} {pred[2]:<10.4f} {relevancy:<12.4f} ")
    etime = time.time()
    print(f"\nSearch completed within {(etime-stime):.2f} s")

if __name__ == "__main__":
    main()