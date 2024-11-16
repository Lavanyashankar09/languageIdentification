import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Custom Dataset for reading HDF5 data
class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file_path, layer):
        self.hdf5_file_path = hdf5_file_path
        self.layer = layer  # Single layer name
        with h5py.File(hdf5_file_path, 'r') as hf:
            self.num_files = hf[self.layer].shape[1]  # Number of audio files

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file_path, 'r') as hf:
            X = hf[self.layer][:, idx, :]  # Load embeddings from the specified layer
            y = hf['LID'][idx].decode('utf-8')
        
        label_map = {'English': 0, 'Mandarin': 1}  # Example mapping
        y = label_map[y]  # Directly map y to an integer
        return X, y

# Define your language identification model
class LanguageIdentificationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(LanguageIdentificationModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])  # Use the hidden state of the last LSTM layer
        return out

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
    print("y true", y_true)
    print("len is", len(y_true))
    print("y true", y_pred)
    print("len is", len(y_pred))
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
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(layer_dir, 'roc_curve.pdf'))
    plt.close()

def save_checkpoint(model, optimizer, epoch, val_accuracy, save_path):
    checkpoint = {
        'epoch': epoch + 1,  # Save the current epoch number
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_accuracy
    }

    torch.save(checkpoint, save_path)
    print(f'Checkpoint saved at {save_path}')

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, layer_dir_cp):
    train_loss_values = []
    train_accuracy_values = []
    val_loss_values = []
    val_accuracy_values = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            X_batch, y_batch = batch  # Get a batch of embeddings and labels
            y_batch = torch.tensor(y_batch, dtype=torch.long)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)  # Calculate the loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        train_loss_values.append(running_loss / len(train_loader))
        train_accuracy_values.append(100 * correct / total)

        # Validation step
        val_loss, val_correct, val_total = 0.0, 0, 0
        model.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                X_val_batch, y_val_batch = val_batch
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

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(layer_dir_cp, f'model_epoch_{epoch + 1}.pth')
            save_checkpoint(model, optimizer, epoch, val_accuracy_values[-1], checkpoint_path)
    return train_loss_values, train_accuracy_values, val_loss_values, val_accuracy_values

def test_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for batch in test_loader:
            X_batch, y_batch = batch
            outputs = model(X_batch)

            y_true.extend(y_batch.tolist())
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.tolist())
            y_scores.extend(torch.softmax(outputs, dim=1)[:, 1].tolist())  # Get scores for the positive class

    accuracy = 100 * sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true)
    print(f'Accuracy of the model on the test set: {accuracy:.2f}%')

    return y_true, y_pred, y_scores

def main(hdf5_file_path, save_dir_plot, save_dir_cp, layer, hidden_dim, batch_size, num_epochs):
    # Directory for saving plots    
    layer_dir_plot = os.path.join(save_dir_plot, layer)
    os.makedirs(layer_dir_plot, exist_ok=True)  # Ensure directory for plots exists

    # Directory for saving checkpoints
    layer_dir_cp = os.path.join(save_dir_cp, layer)
    os.makedirs(layer_dir_cp, exist_ok=True)  # Ensure directory for checkpoints exists

    # Initialize dataset and dataloader
    dataset = HDF5Dataset(hdf5_file_path, layer)

    # Split the dataset into training, validation, and testing sets
    train_size = int(0.7 * len(dataset))  # 70% for training
    val_size = int(0.15 * len(dataset))  # 15% for validation
    test_size = len(dataset) - train_size - val_size  # 15% for testing
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    print("loading data")
    # Create dataloaders for training, validation, and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model parameters
    input_dim = dataset[0][0].shape[1]  # Dimension from the specified layer
    print(input_dim)
    num_classes = 2  # Number of languages (English, Mandarin, etc.)

    # Instantiate the model, optimizer, and loss function
    model = LanguageIdentificationModel(input_dim, hidden_dim, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    print("traing model")
    # Train the model
    train_loss_values, train_accuracy_values, val_loss_values, val_accuracy_values = train_model(
    model, train_loader, val_loader, optimizer, criterion, num_epochs, layer_dir_cp)

    print("testing model")
    # Test the model
    y_true, y_pred, y_scores = test_model(model, test_loader)
    print("plotting")
    # Plot results
    plot_loss(train_loss_values, val_loss_values, num_epochs, layer_dir_plot)
    plot_accuracy(train_accuracy_values, val_accuracy_values, num_epochs, layer_dir_plot)
    plot_confusion_matrix(y_true, y_pred, layer_dir_plot)
    plot_roc_curve(y_true, y_scores, layer_dir_plot)
    print("save the plots")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a language identification model.')
    parser.add_argument('--hdf5_file', type=str, default='/export/c09/lavanya/languageIdentification/zinglish/small/embeddingSmall/embed_1727983789.h5', help='Path to the HDF5 file.')
    parser.add_argument('--save_dir_plot', type=str, default='/export/c09/lavanya/languageIdentification/zinglish/small/plotSmall', help='save plots')
    parser.add_argument('--save_dir_cp', type=str, default='/export/c09/lavanya/languageIdentification/zinglish/small/checkpointSmall', help='save cp')
    
    # change here for layer
    parser.add_argument('--layer', type=str, default='Layer_1', help='Layer to include (e.g., Layer_1).')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dim')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 32).')
    parser.add_argument('--num_epochs', type=int, default=13, help='Number of epochs for training (default: 10).')
    args = parser.parse_args()
    main(args.hdf5_file, args.save_dir_plot, args.save_dir_cp, args.layer, args.hidden_dim, args.batch_size, args.num_epochs)
