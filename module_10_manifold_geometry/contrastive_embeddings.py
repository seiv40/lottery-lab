"""
Module 10: Contrastive Embeddings (SimCLR-style)
Self-supervised learning via contrastive loss on augmented views.

Uses NT-Xent loss to learn embeddings where augmentations of the same
draw are close, while different draws are pushed apart.

"""

import argparse
import os
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

from config import (
    LotteryName,
    RANDOM_STATE_DEFAULT,
    get_module10_output_dir,
    get_module10_figures_dir,
)
from data import load_features


class LotteryDataset(Dataset):
    """Dataset for contrastive learning with augmentations."""
    
    def __init__(self, X: np.ndarray, random_state: int = RANDOM_STATE_DEFAULT):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.rng = np.random.default_rng(random_state)
        
    def __len__(self):
        return len(self.X)
    
    def augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations: Gaussian noise, permutation, scaling."""
        x_aug = x.clone()
        
        # Gaussian jitter (sigma = 0.1)
        if self.rng.random() > 0.5:
            noise = torch.randn_like(x_aug) * 0.1
            x_aug = x_aug + noise
        
        # Feature permutation (swap 2 random features)
        if self.rng.random() > 0.5 and len(x_aug) > 1:
            idx = self.rng.choice(len(x_aug), size=2, replace=False)
            x_aug[idx[0]], x_aug[idx[1]] = x_aug[idx[1]].clone(), x_aug[idx[0]].clone()
        
        # Random scaling (0.9-1.1x)
        if self.rng.random() > 0.5:
            scale = 0.9 + 0.2 * self.rng.random()
            x_aug = x_aug * scale
            
        return x_aug
    
    def __getitem__(self, idx):
        x = self.X[idx]
        x_i = self.augment(x)
        x_j = self.augment(x)
        return x_i, x_j


class ContrastiveEncoder(nn.Module):
    """Simple MLP encoder for contrastive learning."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, embedding_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, embedding_dim),
        )
        
    def forward(self, x):
        return F.normalize(self.encoder(x), dim=1)


def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """NT-Xent loss (SimCLR)."""
    batch_size = z_i.shape[0]
    
    # Concatenate augmented views
    z = torch.cat([z_i, z_j], dim=0)
    
    # Compute similarity matrix
    sim = torch.mm(z, z.t()) / temperature
    
    # Remove self-similarities
    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)
    
    positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(2 * batch_size, 1)
    
    # Mask to remove positive pairs and self-comparisons
    mask = torch.ones((2 * batch_size, 2 * batch_size), dtype=bool, device=z.device)
    mask = mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    
    negative_samples = sim[mask].reshape(2 * batch_size, -1)
    
    logits = torch.cat([positive_samples, negative_samples], dim=1)
    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)
    
    loss = F.cross_entropy(logits, labels)
    return loss


def train_contrastive_model(
    X: np.ndarray,
    hidden_dim: int = 128,
    embedding_dim: int = 32,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    temperature: float = 0.5,
    random_state: int = RANDOM_STATE_DEFAULT,
    device: str = "cpu",
) -> Tuple[np.ndarray, list]:
    """Train contrastive model and return embeddings + loss history."""
    
    dataset = LotteryDataset(X, random_state=random_state)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = ContrastiveEncoder(
        input_dim=X.shape[1],
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for x_i, x_j in loader:
            x_i, x_j = x_i.to(device), x_j.to(device)
            
            z_i = model(x_i)
            z_j = model(x_j)
            
            loss = nt_xent_loss(z_i, z_j, temperature=temperature)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Generate embeddings for full dataset
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        embeddings = model(X_tensor).cpu().numpy()
    
    return embeddings, loss_history


def compute_random_baseline_clustering(
    X: np.ndarray,
    n_clusters: int = 5,
    n_trials: int = 10,
    random_state: int = RANDOM_STATE_DEFAULT,
) -> dict:
    """Compute clustering metrics on randomly shuffled data as baseline."""
    from sklearn.cluster import KMeans
    
    rng = np.random.default_rng(random_state)
    silhouettes = []
    ch_scores = []
    
    for trial in range(n_trials):
        # Shuffle each column independently
        X_random = X.copy()
        for j in range(X.shape[1]):
            rng.shuffle(X_random[:, j])
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state + trial, n_init=10)
        labels = kmeans.fit_predict(X_random)
        
        silhouettes.append(silhouette_score(X_random, labels))
        ch_scores.append(calinski_harabasz_score(X_random, labels))
    
    return {
        "silhouette_mean": float(np.mean(silhouettes)),
        "silhouette_std": float(np.std(silhouettes)),
        "ch_score_mean": float(np.mean(ch_scores)),
        "ch_score_std": float(np.std(ch_scores)),
    }


def run_contrastive_embeddings(
    lottery: LotteryName,
    hidden_dim: int = 128,
    embedding_dim: int = 32,
    epochs: int = 50,
    batch_size: int = 64,
    max_samples: int = 800,
    random_state: int = RANDOM_STATE_DEFAULT,
) -> None:
    """Run contrastive embedding analysis."""
    
    print(f"\n[Module10][Contrastive] Starting for {lottery}...")
    
    df = load_features(lottery, numeric_only=True)
    X_full = df.to_numpy(dtype=float)
    
    # Downsample if needed
    if X_full.shape[0] > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X_full.shape[0], size=max_samples, replace=False)
        X = X_full[idx]
        print(f"  Downsampled from {X_full.shape[0]} to {X.shape[0]} samples.")
    else:
        X = X_full
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    
    # Train model
    embeddings, loss_history = train_contrastive_model(
        X,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        epochs=epochs,
        batch_size=batch_size,
        random_state=random_state,
        device=device,
    )
    
    # Compute clustering metrics
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=5, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    silhouette = silhouette_score(embeddings, labels)
    ch_score = calinski_harabasz_score(embeddings, labels)
    
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Calinski-Harabasz Score: {ch_score:.4f}")
    
    # Compute random baseline
    print("  Computing random baseline clustering...")
    baseline = compute_random_baseline_clustering(
        embeddings,
        n_clusters=5,
        n_trials=10,
        random_state=random_state,
    )
    
    print(f"  Random Baseline Silhouette: {baseline['silhouette_mean']:.4f} +/- {baseline['silhouette_std']:.4f}")
    print(f"  Random Baseline CH Score: {baseline['ch_score_mean']:.4f} +/- {baseline['ch_score_std']:.4f}")
    
    # Save outputs
    output_dir = get_module10_output_dir(lottery)
    fig_dir = get_module10_figures_dir(lottery)
    
    # Save embeddings
    out_npy = os.path.join(output_dir, f"{lottery}_contrastive_embeddings.npy")
    np.save(out_npy, embeddings)
    
    # Save metrics with baseline
    out_csv = os.path.join(output_dir, f"{lottery}_contrastive_metrics.csv")
    pd.DataFrame([{
        "lottery": lottery,
        "final_loss": loss_history[-1],
        "silhouette_score": silhouette,
        "calinski_harabasz_score": ch_score,
        "baseline_silhouette_mean": baseline["silhouette_mean"],
        "baseline_silhouette_std": baseline["silhouette_std"],
        "baseline_ch_score_mean": baseline["ch_score_mean"],
        "baseline_ch_score_std": baseline["ch_score_std"],
        "embedding_dim": embedding_dim,
        "epochs": epochs,
    }]).to_csv(out_csv, index=False)
    
    # Save loss history
    loss_csv = os.path.join(output_dir, f"{lottery}_contrastive_loss_history.csv")
    pd.DataFrame({
        "epoch": np.arange(len(loss_history)),
        "loss": loss_history,
    }).to_csv(loss_csv, index=False)
    
    # Plotting
    try:
        import matplotlib.pyplot as plt
        
        # Loss curve
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("NT-Xent Loss")
        plt.title(f"{lottery.capitalize()} - Contrastive Loss")
        plt.grid(True, alpha=0.3)
        
        # t-SNE of embeddings
        from sklearn.manifold import TSNE
        
        plt.subplot(1, 2, 2)
        tsne = TSNE(n_components=2, random_state=random_state)
        emb_2d = tsne.fit_transform(embeddings)
        plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="tab10", s=10, alpha=0.6)
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title(f"{lottery.capitalize()} - Contrastive Embeddings (t-SNE)")
        plt.colorbar(label="K-Means Cluster")
        
        fig_path = os.path.join(fig_dir, f"{lottery}_contrastive_analysis.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"  Saved figure to {fig_path}")
        
    except ImportError:
        pass
    
    print(f"[Module10][Contrastive] Completed for {lottery}.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Module 10 - Contrastive Embeddings (SimCLR-style)"
    )
    parser.add_argument(
        "--lottery",
        type=str,
        choices=["powerball", "megamillions"],
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=800)
    parser.add_argument("--random_state", type=int, default=RANDOM_STATE_DEFAULT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_contrastive_embeddings(
        lottery=args.lottery,  # type: ignore
        epochs=args.epochs,
        max_samples=args.max_samples,
        random_state=args.random_state,
    )