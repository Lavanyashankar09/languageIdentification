import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
from torch.nn.utils.rnn import pad_sequence
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file_paths, layer, fraction=1.0):
        self.hdf5_file_paths = hdf5_file_paths  
        self.layer = layer
        self.num_files = sum(h5py.File(file, 'r')[self.layer].shape[1] for file in hdf5_file_paths)  
        
        # Calculate the number of samples to keep based on the fraction
        self.num_samples = int(self.num_files * fraction)
        print(f"Initialized dataset with {self.num_samples} samples (using {fraction*100}% of the data) from {len(hdf5_file_paths)} files.")

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        current_idx = idx
        for hdf5_file_path in self.hdf5_file_paths:
            with h5py.File(hdf5_file_path, 'r') as hf:
                if current_idx < hf[self.layer].shape[1]:  
                    X = hf[self.layer][:, current_idx, :]  
                    y = hf['LID'][current_idx].decode('utf-8')
                    break
                current_idx -= hf[self.layer].shape[1]  
        X = torch.tensor(X, dtype=torch.float32) 
        label_map = {'English': 0, 'Mandarin': 1}  
        y = label_map[y]  
        return X, y

def collate_fn(batch):
    X, y = zip(*batch)
    X_padded = pad_sequence(X, batch_first=True)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return X_padded, y_tensor

def main(hdf5_dir, save_dir_plot, layer, batch_size, fraction):
    layer_dir_plot = os.path.join(save_dir_plot, layer)
    if not os.path.exists(layer_dir_plot):
        os.makedirs(layer_dir_plot)
    hdf5_file_paths = [os.path.join(hdf5_dir, f) for f in os.listdir(hdf5_dir) if f.endswith('.h5')]
    print(f"Found {len(hdf5_file_paths)} HDF5 files in directory: {hdf5_dir}")

    dataset = HDF5Dataset(hdf5_file_paths, layer, fraction)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)

    # Step 1: Collect all features and labels for t-SNE
    all_features = []
    all_labels = []

    for batch_X, batch_y in train_loader:
        all_features.append(batch_X)
        all_labels.append(batch_y)
        #print(f"Loaded batch - Features shape: {batch_X.shape}, Labels shape: {batch_y.shape}")
    
    # Concatenate all batches to create a single feature matrix
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0).numpy()
    print(f"Total features shape: {all_features.shape}, Total labels length: {len(all_labels)}")

    # Step 2: Perform t-SNE
    all_features_np = all_features.numpy()  # Convert to numpy array
    all_features_flattened = all_features_np.reshape(all_features_np.shape[0], -1)  # Flatten the features
    print("Flattened features shape for t-SNE:", all_features_flattened.shape)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)  # t-SNE with 2 components for visualization
    tsne_components = tsne.fit_transform(all_features_flattened)
    print("t-SNE transformation completed.")

    # Step 3: Visualize t-SNE results
    plt.figure(figsize=(8, 6))
    # Define colors for each label
    colors = ['yellow' if label == 0 else 'purple' for label in all_labels]
    scatter = plt.scatter(tsne_components[:, 0], tsne_components[:, 1], c=colors, alpha=0.7)
    plt.title('t-SNE of Language Identification Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    # Add color legend for languages
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='English (0)'),  # Green for English
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Mandarin (1)')  # Red for Mandarin
    ], title='Language')
    # Save and show the plot
    plt.savefig(os.path.join(layer_dir_plot, 'tsne_plot.png'))
    plt.show()
    print(f"t-SNE plot saved to {layer_dir_plot}/tsne_plot.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a language identification model.')
    parser.add_argument('--hdf5_dir', type=str, default='/export/c09/lavanya/languageIdentification/zinglish/large/embeddingLarge/', help='Path to the HDF5 dir')
    parser.add_argument('--save_dir_plot', type=str, default='/export/c09/lavanya/languageIdentification/zinglish/large/compareLarge', help='Save plots')
    parser.add_argument('--layer', type=str, default='Layer_1', help='Layer to include (e.g., Layer_1).')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 32).')
    parser.add_argument('--fraction', type=float, default=1, help='Fraction of the dataset to use for testing (default: 0.1 for 10%).')
    args = parser.parse_args()
    main(args.hdf5_dir, args.save_dir_plot, args.layer, args.batch_size, args.fraction)
