import os
import h5py
import torch
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random
import torch.multiprocessing as mp
from sklearn.manifold import TSNE
mp.set_start_method('spawn', force=True)
mp.set_sharing_strategy('file_system')

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file_paths, layer, device):
        self.hdf5_file_paths = hdf5_file_paths  
        self.layer = layer
        self.device = device  
        self.num_files = sum(h5py.File(file, 'r')[self.layer].shape[1] for file in hdf5_file_paths)  
        print(f"Total number of files: {self.num_files}")

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        current_idx = idx
        for hdf5_file_path in self.hdf5_file_paths:
            with h5py.File(hdf5_file_path, 'r') as hf:
                if current_idx < hf[self.layer].shape[1]: 
                    X = hf[self.layer][:, current_idx, :] 
                    y = hf['LID'][current_idx].decode('utf-8')
                    break
                current_idx -= hf[self.layer].shape[1]  
        X = torch.tensor(X, dtype=torch.float32).to(self.device)  
        print(f"Data on device: {X.device}") 
        label_map = {'English': 0, 'Mandarin': 1}  
        y = label_map[y]  
        return X, y


def collate_fn(batch):
    X, y = zip(*batch)
    X_padded = pad_sequence(X, batch_first=True)
    y_tensor = torch.tensor(y, dtype=torch.long)
    print(f"Padded batch shape: {X_padded.shape}")
    return X_padded, y_tensor

def apply_clustering(embeddings, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    score = silhouette_score(embeddings, cluster_labels)
    print(f"Silhouette Score: {score}")
    return cluster_labels

def plot_cluster_results(embeddings, cluster_labels, save_dir):
    tsne = TSNE(n_components=2, random_state=42, n_iter=1000, perplexity=30)
    reduced_embeddings = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.6)
    plt.colorbar(scatter)
    plt.title("t-SNE Cluster Visualization")
    print("save dir", save_dir)
    plt.savefig(os.path.join(save_dir, "tsne_cluster_plot.png"))
    print("Saving plot...")
    plt.show()


def main(hdf5_dir, save_dir_plot, batch_size, layer, n_clusters=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    layer_dir_plot = os.path.join(save_dir_plot, layer)
    os.makedirs(layer_dir_plot, exist_ok=True)
    hdf5_file_paths = [os.path.join(hdf5_dir, f) for f in os.listdir(hdf5_dir) if f.endswith('.h5')]
    dataset1 = HDF5Dataset(hdf5_file_paths, layer, device)  

    dataset_size = len(dataset1)
    if dataset_size == 0:
        raise ValueError("Dataset is empty. Check the HDF5 directory and layer configuration.")
    sampled_size = max(1, int(dataset_size * 0.03))
    print(f"Sampled size: {sampled_size}")
    sampled_indices = random.sample(range(dataset_size), sampled_size)
    dataset = torch.utils.data.Subset(dataset1, sampled_indices)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    embeddings = []
    for i, (X, _) in enumerate(data_loader):
        print(f"Processing batch {i}...")
        if X.device != device: 
            print(f"Warning: Batch is not on {device}, it's on {X.device}")
        batch_embeddings = X.view(-1, X.shape[-1]).to(device).cpu().numpy()
        embeddings.append(batch_embeddings)
    embeddings = np.vstack(embeddings)  
    print(f"Embeddings shape: {embeddings.shape}")
    cluster_labels = apply_clustering(embeddings, n_clusters)
    plot_cluster_results(embeddings, cluster_labels, layer_dir_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a language identification model.')
    parser.add_argument('--hdf5_dir', type=str,  help='Path to the HDF5 dir')
    parser.add_argument('--save_dir_plot', type=str,help='save plots')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 32).')
    parser.add_argument('--layer', type=str, help='Layer to include.')
    parser.add_argument('--n_clusters', type=int, default=2, help='Number of clusters for KMeans.')
    args = parser.parse_args()
    main(args.hdf5_dir, args.save_dir_plot,  args.batch_size, args.layer, args.n_clusters)
