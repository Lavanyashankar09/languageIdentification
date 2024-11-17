import os
import h5py
import torch
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE  
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import random

# Custom Dataset for reading multiple HDF5 files
class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file_paths, layer):
        self.hdf5_file_paths = hdf5_file_paths  # List of HDF5 file paths
        self.layer = layer
        self.num_files = sum(h5py.File(file, 'r')[self.layer].shape[1] for file in hdf5_file_paths)  # Total number of audio files
        print(f"Total number of files: {self.num_files}")


    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        current_idx = idx
        for hdf5_file_path in self.hdf5_file_paths:
            with h5py.File(hdf5_file_path, 'r') as hf:
                if current_idx < hf[self.layer].shape[1]:  # Check if index is within this file's range
                    X = hf[self.layer][:, current_idx, :]  # Load embeddings
                    y = hf['LID'][current_idx].decode('utf-8')
                    break
                current_idx -= hf[self.layer].shape[1]  # Move to the next file
        # Convert embeddings (X) from NumPy to PyTorch tensor
        X = torch.tensor(X, dtype=torch.float32)  # Change dtype if necessary
        label_map = {'English': 0, 'Mandarin': 1}  # Example mapping
        y = label_map[y]  # Directly map y to an integer

        #print("X shape: ", X.shape)  # Debugging

        return X, y


def collate_fn(batch):
    # Unzip the batch into separate lists for X (embeddings) and y (labels)
    X, y = zip(*batch)
    # Stack the embeddings into a single tensor
    # X will be a list of tensors with shape [sequence_length, embedding_dim]
    # If the sequence lengths are variable, use padding
    X_padded = pad_sequence(X, batch_first=True)
    # Convert labels to a tensor
    y_tensor = torch.tensor(y, dtype=torch.long)
    print(f"Padded batch shape: {X_padded.shape}")
    return X_padded, y_tensor

# Function to perform clustering
def apply_clustering(embeddings, n_clusters=2):
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Optionally, print silhouette score for clustering evaluation
    score = silhouette_score(embeddings, cluster_labels)
    print(f"Silhouette Score: {score}")
    
    return cluster_labels

def plot_cluster_results(embeddings, cluster_labels, save_dir):
    # Use t-SNE for dimensionality reduction (2D)
    tsne = TSNE(n_components=2, random_state=42, n_iter=1000, perplexity=30)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Plot the t-SNE results
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.6)
    plt.colorbar(scatter)
    plt.title("t-SNE Cluster Visualization")
    plt.savefig(os.path.join(save_dir, "tsne_cluster_plot.png"))
    plt.show()

def main(hdf5_dir, save_dir_plot, batch_size,layer,  n_clusters=2):
    # Create the directory for saving plots
  
    # print(f"Layer: {layer}")
    # print(f"Number of save_dir_plot: {save_dir_plot}")
    layer_dir_plot = os.path.join(save_dir_plot, layer)
    os.makedirs(layer_dir_plot, exist_ok=True)


    hdf5_file_paths = [os.path.join(hdf5_dir, f) for f in os.listdir(hdf5_dir) if f.endswith('.h5')]
    dataset = HDF5Dataset(hdf5_file_paths, layer)

    # # Get the total number of samples in dataset1
    # dataset_size = len(dataset1)

    # # Use only 0.1% of the dataset
    # sampled_size = int(dataset_size * 0.001)
    # print(f"Sampled size: {sampled_size}")

    # # Randomly sample indices for the 0.1% subset
    # sampled_indices = random.sample(range(dataset_size), sampled_size)

    # # Create a new dataset with only the sampled data
    # dataset = torch.utils.data.Subset(dataset1, sampled_indices)

    # Create a DataLoader for the sampled data
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    embeddings = []
    for i, (X, _) in enumerate(data_loader):
        if X.shape[0] != 32:  # Skip incomplete batch (last batch might have less than 32 samples)
            continue
        # Reshape to maintain individual embeddings
        batch_embeddings = X.numpy().reshape(-1, 512)  # Flatten along sequence length but keep embedding dim
        embeddings.append(batch_embeddings)

        print("flattened shape: ", batch_embeddings)  # Debugging

    embeddings = np.vstack(embeddings)  # Convert list to NumPy array
    print(f"Embeddings shape: {embeddings.shape}")
    # Apply clustering
    cluster_labels = apply_clustering(embeddings, n_clusters)

    # Plot clustering results using t-SNE
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
