# Standard library imports
import os
import argparse
import numpy as np

# Third-party imports
import h5py
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, accuracy_score, balanced_accuracy_score, f1_score
import random
# Mixed precision training tools
from torch.cuda.amp import autocast, GradScaler


### DATA HANDLING ###

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file_paths, layer):
        self.hdf5_file_paths = hdf5_file_paths
        self.layer = layer
        self.num_files = sum(h5py.File(file, 'r')[self.layer].shape[1] for file in hdf5_file_paths)

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
        X = torch.tensor(X, dtype=torch.float32)
        label_map = {'English': 0, 'Mandarin': 1}
        y = label_map[y]
        return X, y

def collate_fn(batch):
    X, y = zip(*batch)
    X_padded = pad_sequence(X, batch_first=True)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return X_padded, y_tensor


### MODEL DEFINITION ###

class LanguageIdentificationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate=0.5):
        super(LanguageIdentificationModel, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout after the LSTM layers
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        lstm_out = self.dropout(lstm_out)  # Dropout applied after LSTM
        out = lstm_out[:, -1, :]  # Take the last time step's output
        out = self.fc(out)  # Final classification layer
        return out


### TRAINING AND EVALUATION ###

def save_checkpoint(model, optimizer, epoch, val_accuracy, val_loss, learning_rate, best_val_accuracy, save_path, device):
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_accuracy,
        'val_loss': val_loss,
        'learning_rate': learning_rate,
        'best_val_accuracy': best_val_accuracy,
        'device': str(device)
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at {save_path}, Epoch: {epoch+1}, Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Best Val Accuracy: {best_val_accuracy:.4f}")


def load_checkpoint(model, optimizer, checkpoint_path, device='cpu'):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        val_accuracy = checkpoint['val_accuracy']
        val_loss = checkpoint['val_loss']
        learning_rate = checkpoint['learning_rate']
        best_val_accuracy = checkpoint['best_val_accuracy']
        
        print(f"Checkpoint loaded from {checkpoint_path}, starting from epoch {epoch}, "
              f"Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}, "
              f"Best Val Accuracy: {best_val_accuracy:.4f}, Learning Rate: {learning_rate:.4e}")
        
        return epoch, val_accuracy, val_loss, learning_rate, best_val_accuracy
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting training from scratch.")
        return 0, 0.0, None, None, 0.0

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, layer_dir_cp, start_epoch=0, patience=5, save_every_n_epochs=3, device='cuda'):
    scaler = GradScaler()
    train_loss_values, train_accuracy_values = [], []
    val_loss_values, val_accuracy_values = [], []
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for batch in train_loader:
            X_batch, y_batch = batch

            # Move data to the correct device (GPU or CPU)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        train_loss_values.append(running_loss / len(train_loader))
        train_accuracy_values.append(100 * correct / total)

        # Validation loop
        val_loss, val_correct, val_total = 0.0, 0, 0
        model.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                X_val_batch, y_val_batch = val_batch

                # Move validation data to the correct device (GPU or CPU)
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                with autocast():
                    outputs_val = model(X_val_batch)
                    val_loss += criterion(outputs_val, y_val_batch).item()
                    _, val_predicted = torch.max(outputs_val.data, 1)
                    val_total += y_val_batch.size(0)
                    val_correct += (val_predicted == y_val_batch).sum().item()

        val_loss_values.append(val_loss / len(val_loader))
        val_accuracy_values.append(100 * val_correct / val_total)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss_values[-1]:.4f}, Train Accuracy: {train_accuracy_values[-1]:.2f}%, '
              f'Validation Loss: {val_loss_values[-1]:.4f}, Validation Accuracy: {val_accuracy_values[-1]:.2f}%')

        if (epoch + 1) % save_every_n_epochs == 0:
            checkpoint_path = os.path.join(layer_dir_cp, f'model_epoch_{epoch + 1}.pth')
            save_checkpoint(model, optimizer, epoch, val_accuracy_values[-1], val_loss_values[-1], 0.001, 0.0, checkpoint_path, device)

        if val_loss_values[-1] < best_val_loss:
            best_val_loss = val_loss_values[-1]
            epochs_without_improvement = 0
            checkpoint_path = os.path.join(layer_dir_cp, f'model_best.pth')
            save_checkpoint(model, optimizer, epoch, val_accuracy_values[-1], val_loss_values[-1], 0.001, 0.0, checkpoint_path, device)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {patience} epochs of no improvement.")
                break

    return train_loss_values, train_accuracy_values, val_loss_values, val_accuracy_values


def test_model(model, test_loader, device):
    model = model.to(device)  # Ensure model is on the same device as the input data
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Ensure data is on the same device as the model
            outputs = model(X_batch)
            y_true.extend(y_batch.tolist())
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.tolist())
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            y_scores.extend(probabilities.tolist())
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr  
    difference = np.abs(fpr - fnr)  
    eer_index = np.argmin(difference)  
    eer = (fpr[eer_index] + fnr[eer_index]) / 2.0  
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    print(f"EER: {eer:.4f}")
    return y_true, y_pred, y_scores, accuracy, balanced_accuracy, f1, eer


### PLOTTING ###

def plot_loss(train_loss, val_loss, num_epochs, layer_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_loss, marker='o', label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_loss, marker='o', label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(range(1, num_epochs + 1))
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(layer_dir, 'loss_plot.pdf'))
    plt.close()

def plot_accuracy(train_accuracy, val_accuracy, num_epochs, layer_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_accuracy, marker='o', label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracy, marker='o', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.xticks(range(1, num_epochs + 1))
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(layer_dir, 'accuracy_plot.pdf'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, layer_dir):
    conf_matrix = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(layer_dir, 'confusion_matrix.pdf'))
    plt.close()

def plot_roc_curve(y_true, y_scores, layer_dir):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(layer_dir, 'roc_curve.pdf'))
    plt.close()


### MAIN EXECUTION ###

def main(hdf5_dir, save_dir_plot, save_dir_cp, layer, hidden_dim, batch_size, num_epochs, checkpoint_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Training model for layer: {layer}")
    print
    layer_dir_plot = os.path.join(save_dir_plot, layer)
    os.makedirs(layer_dir_plot, exist_ok=True)
    layer_dir_cp = os.path.join(save_dir_cp, layer)
    os.makedirs(layer_dir_cp, exist_ok=True)

    hdf5_file_paths = [os.path.join(hdf5_dir, f) for f in os.listdir(hdf5_dir) if f.endswith('.h5')]
    dataset = HDF5Dataset(hdf5_file_paths, layer)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    input_dim = dataset[0][0].shape[1]
    num_classes = 2
    model = LanguageIdentificationModel(input_dim, hidden_dim, num_classes)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    if checkpoint_path:
        start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)

    # Training
    train_loss_values, train_accuracy_values, val_loss_values, val_accuracy_values = train_model(
    model, train_loader, val_loader, optimizer, criterion, num_epochs, layer_dir_cp, start_epoch, patience=5, save_every_n_epochs=3, device=device
    )


    # Testing
    y_true, y_pred, y_scores, accuracy, balanced_accuracy, f1, eer = test_model(model, test_loader, device)

    # Save Plots
    plot_loss(train_loss_values, val_loss_values, num_epochs, layer_dir_plot)
    plot_accuracy(train_accuracy_values, val_accuracy_values, num_epochs, layer_dir_plot)
    plot_confusion_matrix(y_true, y_pred, layer_dir_plot)
    plot_roc_curve(y_true, y_scores, layer_dir_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a language identification model.')
    parser.add_argument('--hdf5_dir', type=str, required=True, help='Path to the HDF5 dir')
    parser.add_argument('--save_dir_plot', type=str, required=True, help='Directory to save plots')
    parser.add_argument('--save_dir_cp', type=str, required=True, help='Directory to save checkpoints')
    parser.add_argument('--layer', type=str, required=True, help='Layer to include')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to a checkpoint to load')
    
    args = parser.parse_args()
    main(args.hdf5_dir, args.save_dir_plot, args.save_dir_cp, args.layer, args.hidden_dim, args.batch_size, args.num_epochs, args.checkpoint_path)