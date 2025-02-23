import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset, DataLoader

# Load your CSV
df = pd.read_csv('HateSpeechData/data1.csv')

# Check the first few rows
print(df.head())

# Convert boolean labels to 0/1
df['label'] = df['label'].map({'hate': True, 'nothate': False})

print(df['label'].value_counts())  # Check distribution

# Vectorize text data
vectorizer = TfidfVectorizer(max_features=1000)  # Limit to top 1000 words
X = vectorizer.fit_transform(df['text']).toarray()

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(df['label'].values, dtype=torch.float32).unsqueeze(1)  # For BCEWithLogitsLoss

print(f"Feature shape: {X_tensor.shape}, Label shape: {y_tensor.shape}")

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create dataset and dataloader
dataset = TextDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, input_dim):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)  # Single output neuron for binary classification

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # BCEWithLogitsLoss will apply sigmoid internally

import torch.optim as optim

model = TextClassifier(input_dim=X_tensor.shape[1])
criterion = nn.BCEWithLogitsLoss()  # For binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
    total_loss = 0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Evaluation (Accuracy)
model.eval()
with torch.no_grad():
    outputs = model(X_tensor)
    predictions = torch.sigmoid(outputs).round()  # Sigmoid + threshold at 0.5
    accuracy = (predictions == y_tensor).float().mean()
    print(f"Accuracy: {accuracy.item() * 100:.2f}%")

# Assuming 'vectorizer' is the TfidfVectorizer used during training
new_text = ["you suck loserrrr"]
new_text_vectorized = vectorizer.transform(new_text).toarray()
new_text_tensor = torch.tensor(new_text_vectorized, dtype=torch.float32)
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    output = model(new_text_tensor)
probability = torch.sigmoid(output).item()
predicted_label = "hate" if probability > 0.5 else "nothate"
print(f"Predicted label: {predicted_label} (probability: {probability:.4f})")
expected_label = "nothate"  # Replace with your expected label
is_correct = (predicted_label == expected_label)
print(f"Is the model correct? {is_correct}")