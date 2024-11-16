import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load your DataFrame
seg_df = pd.read_pickle('/export/c09/lavanya/languageIdentification/test_pickles_40/updated_segments_with_embeddings.pkl')

# Split data into features and target
X = seg_df['embedding'].tolist()  # Convert to list of paths
y = seg_df['language_tag']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to load embeddings
def load_embeddings(file_path):
    data = torch.load(file_path)
    return data['embeddings']  # Ensure this returns a Tensor

# Convert paths to embeddings
X_train_embeddings = [load_embeddings(path) for path in X_train]
X_test_embeddings = [load_embeddings(path) for path in X_test]

# Check if the embeddings are tensors
print(f"Sample type in X_train_embeddings: {type(X_train_embeddings[0])}")
# Print details of embeddings for inspection
print("Sample of X_train_embeddings:")
for i, emb in enumerate(X_train_embeddings[:3]):  # Print first 3 samples
    print(f"Embedding {i}:")
    print(f"Shape: {emb.shape}")
    print(f"Type: {type(emb)}")
    print(f"Values: {emb}")

print("\nSample of X_test_embeddings:")
for i, emb in enumerate(X_test_embeddings[:3]):  # Print first 3 samples
    print(f"Embedding {i}:")
    print(f"Shape: {emb.shape}")
    print(f"Type: {type(emb)}")
    print(f"Values: {emb}")
    
# Convert list of tensors to a single tensor
X_train_embeddings = torch.stack(X_train_embeddings)
X_test_embeddings = torch.stack(X_test_embeddings)

# Flatten embeddings for classification
X_train_flattened = X_train_embeddings.view(X_train_embeddings.size(0), -1).numpy()
X_test_flattened = X_test_embeddings.view(X_test_embeddings.size(0), -1).numpy()

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flattened)
X_test_scaled = scaler.transform(X_test_flattened)

# Initialize and train the classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
