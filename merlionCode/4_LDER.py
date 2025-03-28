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



### PLOTTING ###
def plot_loss(train_loss, val_loss, layer_dir):
    print("start plot loss")
    num_epochs_completed = len(train_loss)  # Use the actual number of completed epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs_completed + 1), train_loss, marker='o', label='Train Loss')
    plt.plot(range(1, num_epochs_completed + 1), val_loss, marker='o', label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(range(1, num_epochs_completed + 1))
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(layer_dir, 'loss_plot.pdf'))
    plt.close()

def plot_accuracy(train_accuracy, val_accuracy, layer_dir):
    print("start plot accuracy")
    num_epochs_completed = len(train_accuracy)  # Use the actual number of completed epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs_completed + 1), train_accuracy, marker='o', label='Train Accuracy')
    plt.plot(range(1, num_epochs_completed + 1), val_accuracy, marker='o', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.xticks(range(1, num_epochs_completed + 1))
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(layer_dir, 'accuracy_plot.pdf'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, layer_dir):
    print("start plot confusion matrix")
    conf_matrix = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(layer_dir, 'confusion_matrix.pdf'))
    plt.close()

def plot_roc_curve(y_true, y_scores, layer_dir):
    print("start plot roc curve")
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
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        'epoch': epoch + 1,  # Save epoch incremented by 1
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_accuracy,
        'val_loss': val_loss,
        'learning_rate': learning_rate,
        'best_val_accuracy': best_val_accuracy,
    }
    # Save checkpoint
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at {save_path}, Epoch: {epoch + 1}, "
          f"Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}, "
          f"Best Val Accuracy: {best_val_accuracy:.4f}")


def load_checkpoint(model, optimizer, checkpoint_path, device='cpu'):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model and optimizer state dicts
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Ensure the model is on the correct device
        model.to(device)
        
        # Extract other information from the checkpoint
        epoch = checkpoint.get('epoch', 0)  # Default to 0 if 'epoch' is missing
        val_accuracy = checkpoint.get('val_accuracy', 0.0)
        val_loss = checkpoint.get('val_loss', None)  # Default to None if 'val_loss' is missing
        learning_rate = checkpoint.get('learning_rate', None)
        best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
        
        # Print checkpoint loading information
        print(f"Checkpoint loaded from {checkpoint_path}, starting from epoch {epoch}, "
        f"Validation Accuracy: {val_accuracy:.4f}, "
        f"Validation Loss: {val_loss if val_loss is not None else 'N/A'}, "
        f"Best Val Accuracy: {best_val_accuracy:.4f}, "
        f"Learning Rate: {learning_rate if learning_rate is not None else 'N/A'}")

        return epoch, val_accuracy, val_loss, learning_rate, best_val_accuracy
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting training from scratch.")
        return 0, 0.0, None, None, 0.0

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, layer_dir_cp, start_epoch, patience, save_every_n_epochs, device='cuda'):
    scaler = GradScaler()
    train_loss_values, train_accuracy_values = [], []
    val_loss_values, val_accuracy_values = [], []
    best_val_loss = float('inf')
    best_val_accuracy = 0.0  # Track best validation accuracy
    epochs_without_improvement = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for batch in train_loader:
            X_batch, y_batch = batch
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

        # Append training loss and accuracy for visualization
        train_loss_values.append(running_loss / len(train_loader))
        train_accuracy_values.append(100 * correct / total)

        # Validation phase
        val_loss, val_correct, val_total = 0.0, 0, 0
        model.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                X_val_batch, y_val_batch = val_batch
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                with autocast():
                    outputs_val = model(X_val_batch)
                    val_loss += criterion(outputs_val, y_val_batch).item()
                    _, val_predicted = torch.max(outputs_val.data, 1)
                    val_total += y_val_batch.size(0)
                    val_correct += (val_predicted == y_val_batch).sum().item()

        # Append validation loss and accuracy for visualization
        val_loss_values.append(val_loss / len(val_loader))
        val_accuracy_values.append(100 * val_correct / val_total)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss_values[-1]:.4f}, Train Accuracy: {train_accuracy_values[-1]:.2f}%, Validation Loss: {val_loss_values[-1]:.4f}, Validation Accuracy: {val_accuracy_values[-1]:.2f}%')

        # Save checkpoint every `save_every_n_epochs`
        if (epoch + 1) % save_every_n_epochs == 0:
            checkpoint_path = os.path.join(layer_dir_cp, f'model_epoch_{epoch + 1}.pth')
            save_checkpoint(model, optimizer, epoch, val_accuracy_values[-1], val_loss_values[-1], optimizer.param_groups[0]['lr'], best_val_accuracy, checkpoint_path, device)

        # Check if validation loss improved, and update best validation accuracy
        if val_loss_values[-1] < best_val_loss:
            best_val_loss = val_loss_values[-1]
            best_val_accuracy = val_accuracy_values[-1]  # Update best validation accuracy
            epochs_without_improvement = 0
            checkpoint_path = os.path.join(layer_dir_cp, 'model_best.pth')
            save_checkpoint(model, optimizer, epoch, best_val_accuracy, best_val_loss, optimizer.param_groups[0]['lr'], best_val_accuracy, checkpoint_path, device)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {patience} epochs of no improvement.")
                break
        

    return train_loss_values, train_accuracy_values, val_loss_values, val_accuracy_values

def test_model(model, test_loader, device):
    model = model.to(device)
    model.eval()
    
    y_true, y_pred, y_scores = [], [], []

    if len(test_loader) == 0:
        print("Test loader is empty. No data to test.")
        return None

    with torch.no_grad():
        for batch in test_loader:
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            
            y_true.extend(y_batch.tolist())
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.tolist())
            
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            y_scores.extend(probabilities.tolist())

    # Convert lists to numpy arrays for easier computation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate confusion matrix elements
    tn = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
    fn = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives
    
    # Calculate LDER components
    total_samples = len(y_true)
    
    # False Alarm Rate (FAR)
    far = fp / (tn + fp) if (tn + fp) > 0 else 0
    
    # Miss Rate (MR)
    mr = fn / (tp + fn) if (tp + fn) > 0 else 0
    
    # Language Confusion Error (CE)
    ce = (fp + fn) / total_samples if total_samples > 0 else 0
    
    # Language Diarization Error Rate (LDER)
    lder = (far + mr + ce) * 100  # Convert to percentage

    # Compute other metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Compute ROC curve and EER
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    difference = np.abs(fpr - fnr)
    eer_index = np.argmin(difference)
    eer = (fpr[eer_index] + fnr[eer_index]) / 2.0

    # Print all evaluation metrics
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    print(f"EER: {eer:.4f}")
    print("\nLanguage Diarization Error Rate Components:")
    print(f"False Alarm Rate (FAR): {far * 100:.2f}%")
    print(f"Miss Rate (MR): {mr * 100:.2f}%")
    print(f"Language Confusion Error (CE): {ce * 100:.2f}%")
    print(f"LDER: {lder:.2f}%")
    
    # Print confusion matrix details
    print("\nConfusion Matrix Details:")
    print(f"True Negatives: {tn}")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")

    results = {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_scores': y_scores,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'f1_score': f1,
        'eer': eer,
        'lder': lder,
        'lder_components': {
            'far': far,
            'mr': mr,
            'ce': ce
        },
        'confusion_matrix': {
            'tn': tn,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    }

    print("\nTest complete.")
    return results

### MAIN EXECUTION ###

def main(hdf5_dir, save_dir_plot, save_dir_cp, layer, hidden_dim, batch_size, num_epochs, patience, save_every_n_epochs, checkpoint_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training model for layer: {layer}")
    
    layer_dir_plot = os.path.join(save_dir_plot, layer)
    os.makedirs(layer_dir_plot, exist_ok=True)
    layer_dir_cp = os.path.join(save_dir_cp, layer)
    os.makedirs(layer_dir_cp, exist_ok=True)
    
    # Load Dataset
    hdf5_file_paths = [os.path.join(hdf5_dir, f) for f in os.listdir(hdf5_dir) if f.endswith('.h5')]
    dataset = HDF5Dataset(hdf5_file_paths, layer)

    # dataset_size = len(dataset1)
    # if dataset_size == 0:
    #     raise ValueError("Dataset is empty. Check the HDF5 directory and layer configuration.")

    # # Sample Dataset
    # sampled_size = max(1, int(dataset_size * 0.001))
    # print(f"Sampled size: {sampled_size}")
    # sampled_indices = random.sample(range(dataset_size), sampled_size)
    # dataset = torch.utils.data.Subset(dataset1, sampled_indices)

    # # Split Dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    # Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # Model Setup
    input_dim = dataset[0][0].shape[1]
    num_classes = 2
    model = LanguageIdentificationModel(input_dim, hidden_dim, num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Load Checkpoint if Provided
    start_epoch = 0
    best_val_accuracy = 0.0
    if checkpoint_path and os.path.isfile(checkpoint_path):
        start_epoch, val_accuracy, val_loss, learning_rate, best_val_accuracy = load_checkpoint(model, optimizer, checkpoint_path, device)
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting training from scratch.")

    print("Starting training...")
    # Training
    train_loss_values, train_accuracy_values, val_loss_values, val_accuracy_values = train_model(
        model, train_loader, val_loader, optimizer, criterion, num_epochs, layer_dir_cp, start_epoch, patience, save_every_n_epochs, device=device
    )

    print("Training complete.")
    print(f"Train Loss: {train_loss_values}")
    print(f"Train Accuracy: {train_accuracy_values}")
    print(f"Validation Loss: {val_loss_values}")
    print(f"Validation Accuracy: {val_accuracy_values}")
    print(f"Epochs Completed: {len(train_loss_values)}")

    print("Starting testing...")
    # Testing
    results = test_model(model, test_loader, device)
    if results is None:
        print("No test results to process.")
        return

    y_true, y_pred, y_scores = results['y_true'], results['y_pred'], results['y_scores']

    # Plot Results
    plot_loss(train_loss_values, val_loss_values, layer_dir_plot)
    plot_accuracy(train_accuracy_values, val_accuracy_values, layer_dir_plot)
    plot_confusion_matrix(y_true, y_pred, layer_dir_plot)
    plot_roc_curve(y_true, y_scores, layer_dir_plot)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a language identification model.")
    parser.add_argument('--hdf5_dir', type=str, required=True, help="Path to the HDF5 directory.")
    parser.add_argument('--save_dir_plot', type=str, required=True, help="Directory to save plots.")
    parser.add_argument('--save_dir_cp', type=str, required=True, help="Directory to save checkpoints.")
    parser.add_argument('--layer', type=str, required=True, help="Layer to include.")
    parser.add_argument('--hidden_dim', type=int, default=128, help="Hidden dimension size.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size.")
    parser.add_argument('--num_epochs', type=int, required=True, help="Number of epochs.")
    parser.add_argument('--patience', type=int, required=True, help="Number of epochs for early stopping patience.")
    parser.add_argument('--save_every_n_epochs', type=int, required=True, help="Frequency to save checkpoints.")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Path to checkpoint file.")
    args = parser.parse_args()
    main(args.hdf5_dir, args.save_dir_plot, args.save_dir_cp, args.layer, args.hidden_dim, args.batch_size, args.num_epochs, args.patience, args.save_every_n_epochs, args.checkpoint_path)
