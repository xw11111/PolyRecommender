import os, numpy as np, pandas as pd
import argparse
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
import time
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Modal Fusion for Molecular Property Prediction')
    parser.add_argument('--seed', type=int, default=42)
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    
    # Model architecture hyperparameters
    parser.add_argument('--hidden_dim1', type=int, default=256)
    parser.add_argument('--hidden_dim2', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.4)
    # MMoE specific
    parser.add_argument('--n_experts', type=int, default=4)
    parser.add_argument('--expert_hidden_dim', type=int, default=256)
    
    # Scheduler hyperparameters
    parser.add_argument('--scheduler_factor', type=float, default=0.5)
    parser.add_argument('--scheduler_patience', type=int, default=10)
    
    parser.add_argument('--out_dir', type=Path, default=Path("../outputs/fusion_results/"))
    return parser.parse_args()



def load_and_align_splits(train_csv, val_csv, test_csv, text_npz, graph_npz):
    """Load pre-split data and align with embeddings using ID matching"""
    # Load CSV files
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # Load embeddings
    lang = np.load(text_npz, allow_pickle=True)
    graph = np.load(graph_npz, allow_pickle=True)

    # Get embedding IDs and data
    lang_ids = lang.get("id", np.arange(len(lang["z_text"])))
    graph_ids = graph.get("id", np.arange(len(graph["z_graph"])))
    lang_embeddings = lang["z_text"]
    graph_embeddings = graph["z_graph"]


    def align_split(df, split_name):
        split_ids = df["id"].values
        
        lang_indices = np.searchsorted(lang_ids, split_ids)
        graph_indices = np.searchsorted(graph_ids, split_ids)
        
        # Verify all IDs are found
        valid_lang = (lang_indices < len(lang_ids)) & (lang_ids[lang_indices] == split_ids)
        valid_graph = (graph_indices < len(graph_ids)) & (graph_ids[graph_indices] == split_ids)
        valid_both = valid_lang & valid_graph
        
        if not np.all(valid_both):
            missing_count = (~valid_both).sum()
            print(f"Warning: {split_name} - {missing_count} IDs not found in embeddings")
            
            # Keep only valid samples
            df = df[valid_both].reset_index(drop=True)
            split_ids = split_ids[valid_both]
            lang_indices = lang_indices[valid_both]
            graph_indices = graph_indices[valid_both]
        
        # Extract embeddings for this split
        Z_text = lang_embeddings[lang_indices].astype(np.float32)
        Z_graph = graph_embeddings[graph_indices].astype(np.float32)
        
        return df, Z_text, Z_graph
    
    # Align each split
    train_df_aligned, Z_text_train, Z_graph_train = align_split(train_df, "Train")
    val_df_aligned, Z_text_val, Z_graph_val = align_split(val_df, "Val")
    test_df_aligned, Z_text_test, Z_graph_test = align_split(test_df, "Test")

    return (train_df_aligned, val_df_aligned, test_df_aligned,
            Z_text_train, Z_text_val, Z_text_test,
            Z_graph_train, Z_graph_val, Z_graph_test)

def standardize_targets(y_train, y_val, y_test):
    """Standardize targets using only training data statistics, handling NaN values"""
    scalers = []
    y_train_scaled = y_train.copy()
    y_val_scaled = y_val.copy()
    y_test_scaled = y_test.copy()
    
    for i in range(y_train.shape[1]):
        # Get non-NaN values for this target from training data only
        train_mask = ~np.isnan(y_train[:, i])
        if train_mask.sum() == 0:
            scalers.append(None)
            continue
            
        # Fit scaler on training data only
        scaler = StandardScaler()
        scaler.fit(y_train[train_mask, i].reshape(-1, 1))
        scalers.append(scaler)
        
        # Transform train data
        y_train_scaled[train_mask, i] = scaler.transform(y_train[train_mask, i].reshape(-1, 1)).flatten()
        
        # Transform validation data using training statistics
        val_mask = ~np.isnan(y_val[:, i])
        if val_mask.sum() > 0:
            y_val_scaled[val_mask, i] = scaler.transform(y_val[val_mask, i].reshape(-1, 1)).flatten()
        
        # Transform test data using training statistics
        test_mask = ~np.isnan(y_test[:, i])
        if test_mask.sum() > 0:
            y_test_scaled[test_mask, i] = scaler.transform(y_test[test_mask, i].reshape(-1, 1)).flatten()
    
    return y_train_scaled, y_val_scaled, y_test_scaled, scalers

def compute_metrics(predictions, targets, scalers, target_names):
    """Compute metrics with denormalization and NaN handling"""
    metrics = {}
    
    # Convert to numpy if needed
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    
    for i, target in enumerate(target_names):
        # Get mask for non-NaN values
        mask = ~np.isnan(targets[:, i])
        if mask.sum() == 0:
            continue
            
        pred_i = predictions[mask, i]
        target_i = targets[mask, i]
        
        # Denormalize using scaler
        pred_i = scalers[i].inverse_transform(pred_i.reshape(-1, 1)).flatten()
        target_i = scalers[i].inverse_transform(target_i.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(target_i, pred_i)
        rmse = np.sqrt(mse)
        r2 = r2_score(target_i, pred_i)
        
        metrics[target] = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'count': mask.sum()
        }
    
    return metrics

def masked_mse(pred, y):
    """MSE loss that handles NaN values"""
    mask = ~torch.isnan(y)
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)
    
    squared_errors = torch.zeros_like(pred)
    squared_errors[mask] = (pred[mask] - y[mask]) ** 2
    per_task_count = mask.sum(dim=0).clamp(min=1)
    per_task_loss = squared_errors.sum(dim=0) / per_task_count
    return per_task_loss.mean()

class MLP(nn.Module):
    def __init__(self, d_in, d_out, hid1=256, hid2=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hid1), nn.ReLU(), nn.LayerNorm(hid1), nn.Dropout(dropout),
            nn.Linear(hid1, hid2), nn.ReLU(), nn.LayerNorm(hid2), nn.Dropout(dropout),
            nn.Linear(hid2, d_out),
        )
    def forward(self, x):
        return self.net(x)

# early fusion -> one shared MLP
class ConcatRegressor(nn.Module):
    def __init__(self, d_text, d_graph, n_tasks=3, hidden_dim1=256, hidden_dim2=128,dropout=0.2):
        super().__init__()
        self.mlp = MLP(d_text + d_graph, n_tasks, hid1=hidden_dim1, hid2=hidden_dim2, dropout=dropout)

    def forward(self, z_text, z_graph, labels=None):
        x = torch.cat([z_text, z_graph], dim=-1)
        logits = self.mlp(x)
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = masked_mse(logits, labels)
        return out

# late fusion of two modality experts with a single gate
# two experts (text expert, graph expert) each expert only sees one modality, produces predictions y_text, y_graph
# one shared gate sees concatenated features [z_text, z_graph] as input, outputs a single scaler weight g between 0 and 1 (via sigmoid)
# logits = g * y_t + (1 - g) * y_g final prediction is g text expert + 1-g graph expert
# explicit modality seperation: The gate essentially learns "should I trust the text-based prediction or the graph-based prediction more for this polymer?"
class MoERegressor(nn.Module):
    """
    Two experts (text, graph) + gating network (on concatenated features).
    Output is convex combo: y = g * E_text + (1-g) * E_graph
    """
    def __init__(self, d_text, d_graph, n_tasks=3, hidden_dim1=256, hidden_dim2=128, dropout=0.2):
        super().__init__()
        self.expert_text = MLP(d_text, n_tasks, hid1=hidden_dim1, hid2=hidden_dim2, dropout=dropout)
        self.expert_graph = MLP(d_graph, n_tasks, hid1=hidden_dim1, hid2=hidden_dim2, dropout=dropout)
        self.gate = nn.Sequential(nn.Linear(d_text+d_graph, 128), nn.ReLU(), nn.Linear(128, n_tasks))

    
    def forward(self, z_text, z_graph, labels=None):
        y_t = self.expert_text(z_text)
        y_g = self.expert_graph(z_graph)
        g = torch.sigmoid(self.gate(torch.cat([z_text,z_graph], -1)))  # [B, T]
        logits = g * y_t + (1 - g) * y_g
        out = {"logits": logits, "gate": g}
        if labels is not None: 
            out["loss"] = masked_mse(logits, labels)
        return out

# true multigate MoE: concatenate the embeddings as input, 4 experts, 3 task-specific gates
# both feature level (experts see concatenated features) and task-specific mixing.
# implicit task-driven specialization.
class TrueMMoERegressor(nn.Module):
    """
    True Multi-gate Mixture of Experts with task-specific gates
    """
    def __init__(self, d_text, d_graph, n_tasks=3, n_experts=4, expert_hidden_dim=128, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.n_tasks = n_tasks
        self.n_experts = n_experts
        input_dim = d_text + d_graph
        
        # Create experts that process concatenated features
        self.experts = nn.ModuleList([
            MLP(input_dim, expert_hidden_dim, hidden_dim, hidden_dim, dropout) 
            for _ in range(n_experts)
        ])
        
        # Task-specific gates
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden_dim),
                nn.ReLU(),
                nn.Linear(expert_hidden_dim, n_experts)
            ) for _ in range(n_tasks)
        ])
        
        # Task-specific towers
        self.towers = nn.ModuleList([
            nn.Sequential(
            nn.Linear(expert_hidden_dim, expert_hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_dim//2, 1),
            ) for _ in range(n_tasks)
        ])
    
    def forward(self, z_text, z_graph, labels=None):
        x = torch.cat([z_text, z_graph], dim=-1)
        
        # Get expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [B, n_experts, hidden]
        
        # Get task-specific gate weights
        gate_weights = []
        task_outputs = []
        
        for task_idx in range(self.n_tasks):
            gate_weight = F.softmax(self.gates[task_idx](x), dim=-1)  # [B, n_experts]
            gate_weights.append(gate_weight)
            
            # Weighted combination of expert outputs
            weighted_output = torch.sum(gate_weight.unsqueeze(-1) * expert_outputs, dim=1)  # [B, hidden]
            
            # Task-specific prediction
            task_pred = self.towers[task_idx](weighted_output)  # [B, 1]
            task_outputs.append(task_pred)
        
        logits = torch.cat(task_outputs, dim=-1)  # [B, n_tasks]
        
        out = {
            "logits": logits,
            "gate_weights": torch.stack(gate_weights, dim=1)  # [B, n_tasks, n_experts]
        }
        
        if labels is not None:
            out["loss"] = masked_mse(logits, labels)
        
        return out

class DS(Dataset):
    def __init__(self, zt, zg, y): 
        self.zt, self.zg, self.y = zt, zg, y
    
    def __len__(self): 
        return len(self.y)
    
    def __getitem__(self, i): 
        return (torch.from_numpy(self.zt[i]), 
                torch.from_numpy(self.zg[i]), 
                torch.from_numpy(self.y[i]))


    
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for z_text, z_graph, targets in dataloader:
        z_text = z_text.to(device)
        z_graph = z_graph.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        output = model(z_text, z_graph, targets)
        loss = output['loss']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #grad_clipping
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def evaluate(model, dataloader, device, scalers, target_names):
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for z_text, z_graph, targets in dataloader:
            z_text = z_text.to(device)
            z_graph = z_graph.to(device)
            targets = targets.to(device)
            
            output = model(z_text, z_graph, targets)
            loss = output['loss']
            
            all_predictions.append(output['logits'])
            all_targets.append(targets)
            total_loss += loss.item()
            num_batches += 1
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    metrics = compute_metrics(predictions, targets, scalers, target_names)
    avg_loss = total_loss / num_batches
    
    return metrics, avg_loss

def plot_training_curves(history, model_name):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend()
    ax1.set_title(f'{model_name} - Loss')

    plt.tight_layout()
    plt.savefig(f'../outputs/fusion_results/{model_name}_training_curves.png', dpi=300)
    plt.close()


def plot_prediction_scatter(model, train_dl, test_dl, device, scalers, targets, model_name):
    """Create scatter plots of predicted vs actual values for each target, showing train and test data"""
    if model_name != 'MMoE':
        return None
        
    model.eval()
    
    # Function to get predictions for a dataloader
    def get_predictions(dataloader):
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for z_text, z_graph, targets_batch in dataloader:
                z_text = z_text.to(device)
                z_graph = z_graph.to(device)
                targets_batch = targets_batch.to(device)
                
                output = model(z_text, z_graph)
                all_predictions.append(output['logits'])
                all_targets.append(targets_batch)
        
        predictions = torch.cat(all_predictions, dim=0).cpu().numpy()
        targets_tensor = torch.cat(all_targets, dim=0).cpu().numpy()
        return predictions, targets_tensor
    
    # Get predictions for both train and test
    train_predictions, train_targets = get_predictions(train_dl)
    test_predictions, test_targets = get_predictions(test_dl)
    
    # Create 3 subplots for the 3 targets
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, target in enumerate(targets):
        # Process train data
        train_mask = ~np.isnan(train_targets[:, i])
        test_mask = ~np.isnan(test_targets[:, i])
        
        if train_mask.sum() == 0 and test_mask.sum() == 0:
            continue
            
        # Get overall min/max for consistent axis scaling
        all_actual = []
        all_pred = []
        
        if train_mask.sum() > 0:
            train_pred_i = train_predictions[train_mask, i]
            train_target_i = train_targets[train_mask, i]
            train_pred_denorm = scalers[i].inverse_transform(train_pred_i.reshape(-1, 1)).flatten()
            train_target_denorm = scalers[i].inverse_transform(train_target_i.reshape(-1, 1)).flatten()
            all_actual.extend(train_target_denorm)
            all_pred.extend(train_pred_denorm)
        
        if test_mask.sum() > 0:
            test_pred_i = test_predictions[test_mask, i]
            test_target_i = test_targets[test_mask, i]
            test_pred_denorm = scalers[i].inverse_transform(test_pred_i.reshape(-1, 1)).flatten()
            test_target_denorm = scalers[i].inverse_transform(test_target_i.reshape(-1, 1)).flatten()
            all_actual.extend(test_target_denorm)
            all_pred.extend(test_pred_denorm)
        
        # Plot train data
        if train_mask.sum() > 0:
            train_r2 = r2_score(train_target_denorm, train_pred_denorm)
            axes[i].scatter(
                train_target_denorm, train_pred_denorm, 
                color='#84ADDC', alpha=0.6, s=10, linewidth=0,
                label=f'Train (R²={train_r2:.3f})'
            )

        if test_mask.sum() > 0:
            test_r2 = r2_score(test_target_denorm, test_pred_denorm)
            axes[i].scatter(
                test_target_denorm, test_pred_denorm, 
                color='#FFA288', alpha=0.6, s=10, linewidth=0,
                label=f'Test (R²={test_r2:.3f})'
            )

        # Add perfect prediction line
        min_val = min(all_actual + all_pred)
        max_val = max(all_actual + all_pred)
        axes[i].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)

        axes[i].set_xlabel(f'{target}')
        axes[i].set_ylabel(f'Pred_{target}')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_aspect('equal', adjustable='box')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(f'../outputs/fusion_results/{model_name}_prediction_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return None


def analyze_gate_behavior(model, dataloader, device, model_name, targets):
    """Analyze gate behavior for MoE models"""
    if 'MoE' not in model_name:
        return None
    model.eval()
    all_gates = []
    with torch.no_grad():
        for z_text, z_graph, _targets_batch in dataloader:
            z_text, z_graph = z_text.to(device), z_graph.to(device)
            output = model(z_text, z_graph)
            if 'gate_weights' in output:
                all_gates.append(output['gate_weights'].cpu().numpy())
            elif 'gate' in output:
                all_gates.append(output['gate'].cpu().numpy())
    if not all_gates:
        return None
    gates = np.concatenate(all_gates, axis=0)

    if model_name == 'MMoE':
        # gates: [N_samples, N_tasks, N_experts] 
        avg_gates_per_task = gates.mean(axis=0)  # [N_tasks, N_experts]
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(avg_gates_per_task,
                    xticklabels=[f'Expert {i}' for i in range(gates.shape[2])],
                    yticklabels=targets, annot=True, fmt='.3f', cmap='viridis', ax=ax)
        plt.tight_layout()
        plt.savefig(f'../outputs/fusion_results/{model_name}_specialization_heatmap.png', dpi=300)
        plt.close()

    elif model_name == 'Binary_MoE':
        # gates shape: [N, T]
        avg_per_task = gates.mean(axis=0)  # [T]
        fig, ax = plt.subplots(figsize=(4,3))
        ax.bar(range(len(avg_per_task)), avg_per_task, width=0.5)
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels(targets)
        ax.set_ylabel('Avg gate to Text (1-g is Graph)')
        plt.tight_layout()
        plt.savefig(f'../outputs/fusion_results/{model_name}_gate_per_task.png', dpi=300)
        plt.close()


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
        
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load & align pre-split data
    (train_df, val_df, test_df,
     Z_text_train, Z_text_val, Z_text_test,
     Z_graph_train, Z_graph_val, Z_graph_test) = load_and_align_splits(
        "../data/train.csv",
        "../data/valid.csv", 
        "../data/test.csv",
        "../outputs/language_embeddings.npz", # polybert
        "../outputs/graph_embeddings.npz"  # gnn
    )
    
    targets = ["Tg", "Tm", "Eg"]
    
    # ===============================
    # Standardize input embeddings
    # ===============================
    text_scaler = StandardScaler()
    graph_scaler = StandardScaler()
    
    Z_text_train = text_scaler.fit_transform(Z_text_train).astype(np.float32)
    Z_text_val   = text_scaler.transform(Z_text_val).astype(np.float32)
    Z_text_test  = text_scaler.transform(Z_text_test).astype(np.float32)
    
    Z_graph_train = graph_scaler.fit_transform(Z_graph_train).astype(np.float32)
    Z_graph_val   = graph_scaler.transform(Z_graph_val).astype(np.float32)
    Z_graph_test  = graph_scaler.transform(Z_graph_test).astype(np.float32)
    # ===============================
    
    # Extract target values
    Y_train = train_df[targets].values.astype(np.float32)
    Y_val = val_df[targets].values.astype(np.float32)
    Y_test = test_df[targets].values.astype(np.float32)
    
    print(f"\nData shapes:")
    print(f"Train - Text: {Z_text_train.shape}, Graph: {Z_graph_train.shape}, Targets: {Y_train.shape}")
    print(f"Val   - Text: {Z_text_val.shape}, Graph: {Z_graph_val.shape}, Targets: {Y_val.shape}")
    print(f"Test  - Text: {Z_text_test.shape}, Graph: {Z_graph_test.shape}, Targets: {Y_test.shape}")
    
    # Standardize targets using only training data
    y_tr_s, y_va_s, y_te_s, scalers = standardize_targets(Y_train, Y_val, Y_test)
    
    train_ds = DS(Z_text_train, Z_graph_train, y_tr_s)
    val_ds   = DS(Z_text_val,   Z_graph_val,   y_va_s)
    test_ds  = DS(Z_text_test,  Z_graph_test,  y_te_s)
    
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    d_text, d_graph = Z_text_train.shape[1], Z_graph_train.shape[1]
    
    models = {
        'Concat': ConcatRegressor(d_text, d_graph, n_tasks=3, hidden_dim1=args.hidden_dim1, hidden_dim2=args.hidden_dim2, dropout=args.dropout),
        'Binary_MoE': MoERegressor(d_text, d_graph, n_tasks=3, hidden_dim1=args.hidden_dim1, hidden_dim2=args.hidden_dim2, dropout=args.dropout),
        'MMoE': TrueMMoERegressor(d_text, d_graph, n_tasks=3, n_experts=args.n_experts, expert_hidden_dim=args.expert_hidden_dim, hidden_dim=args.hidden_dim1, dropout=args.dropout),
    }
    
    results = {}
    training_times = {}      # Store durations per model
    all_histories = {}       # For plotting curves
    
    for model_name, model in models.items():

        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.scheduler_factor, patience=args.scheduler_patience
        )

        best_val_loss = float('inf')
        best_model_state = None
        early_stopping = EarlyStopping(patience=args.early_stopping_patience)

        # History for plots
        history = {'train_loss': [], 'val_loss': [], 'val_r2': []}

        # Timing
        start_time = time.time()

        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_dl, optimizer, device)
            val_metrics, val_loss = evaluate(model, val_dl, device, scalers, targets)

            # avg R^2 across available targets
            valid_r2 = [val_metrics[t]['r2'] for t in targets if t in val_metrics]
            avg_r2 = float(np.mean(valid_r2)) if len(valid_r2) > 0 else float('nan')

            # log history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_r2'].append(avg_r2)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_path = args.out_dir / f"{model_name}_best.pth"   # e.g., Concat_best.pth
                torch.save(best_model_state, best_path)

            # Early stopping check
            if early_stopping(val_loss):
                break

        # End timing
        end_time = time.time()
        duration_sec = end_time - start_time
        training_times[model_name] = duration_sec
        print(f"{model_name} Duration: {duration_sec:.2f} s ({duration_sec/60:.2f} min)")

        # Save history & curves
        all_histories[model_name] = history
        plot_training_curves(history, model_name)

        model.load_state_dict(best_model_state)
        test_metrics, test_loss = evaluate(model, test_dl, device, scalers, targets)

        # Optional analyses
        analyze_gate_behavior(model, test_dl, device, model_name, targets)
        plot_prediction_scatter(model, train_dl, test_dl, device, scalers, targets, model_name)

        results[model_name] = test_metrics
        

    # Compare results
    print(f"\n{'='*50}")
    print("COMPARISON OF ALL MODELS")
    print(f"{'='*50}")

    comparison_df = []
    for model_name, metrics in results.items():
        for target in targets:
            if target in metrics:
                comparison_df.append({
                    'Model': model_name,
                    'Target': target,
                    'R²': metrics[target]['r2'],
                    'RMSE': metrics[target]['rmse'],
                    'Samples': metrics[target]['count']
                })

    comparison_df = pd.DataFrame(comparison_df)
    print(comparison_df.pivot(index=['Target', 'Samples'], columns='Model', values='R²').round(3))
    print()
    print(comparison_df.pivot(index=['Target', 'Samples'], columns='Model', values='RMSE').round(2))

    output_dir = args.out_dir
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    comparison_df.to_csv(f'{output_dir}/comparison_table.csv', index=False)

    with open(f'{output_dir}/training_times.json', 'w') as f:
        json.dump({k: float(v) for k, v in training_times.items()}, f, indent=2)
    
    print(f"\nfusion model is trained")


if __name__ == "__main__":
    main()