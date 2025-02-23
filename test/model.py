import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import os
import glob
import pickle
import csv

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

#paths for saving and loading
MODEL_PATH = "/workspaces/haimaidataorbit2025/test/saved_models/model.pth"
VECTORIZER_PATH = "/workspaces/haimaidataorbit2025/test/saved_models/vectorizer.pkl"

# Concatenate all datasets into one and load dataset
csv_files = glob.glob("/workspaces/haimaidataorbit2025/test/HateSpeechData/*.csv")  # Adjust the path as needed
#print("Found CSV files:", csv_files)

#if not csv_files:
#    print("❌ No CSV files found! Check your directory path.")

required_columns = ['text','label']
df_list = []

for file in csv_files:
    temp_df = pd.read_csv(file)
    
    # Add missing columns if necessary
    for col in required_columns:
        if col not in temp_df.columns:
            temp_df[col] = None  # Fill missing columns with NaN

    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)

# Load all CSVs into a list of DataFrames
df_list = [pd.read_csv(file) for file in csv_files]

# Concatenate all DataFrames into one
df = pd.concat(df_list, ignore_index=True)

# Load your CSV
#df = pd.read_csv('HateSpeechData/data1.csv')

# Check the first few rows
#print(df.head())
#print(f"Total records: {len(df)}")

df['text'] = df['text'].fillna("")  # Fill empty text with ""

# Convert boolean labels to 0/1
df['label'] = df['label'].map({'hate': True, 'nothate': False})

df['label'] = df['label'].fillna(0).astype(int)

# Always initialize vectorizer & X_tensor, regardless of training or loading
if os.path.exists(VECTORIZER_PATH):
    print("🔄 Loading existing vectorizer...")
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
else:
    print("🚨 No vectorizer found. Creating new one...")
    vectorizer = TfidfVectorizer(max_features=1000)
    vectorizer.fit(df['text'])  # Fit on text data

# Transform text data using vectorizer (moved critical variables X_tensor, Y_tensor, and model outside of if train_model so they always initialize)
X = vectorizer.transform(df['text']).toarray()
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(df['label'].values, dtype=torch.float32).unsqueeze(1)

# Check if a saved model exists
if os.path.exists(MODEL_PATH):
    print("🔄 Loading existing model...")
    model = TextClassifier(input_dim=X.shape[1])  # Ensure input size is correct
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()  # Set to evaluation mode
    print("✅ Model loaded successfully! Skipping training.")
    train_model = False
else:
    print("🚨 No saved model found. Training from scratch...")
    model = TextClassifier(input_dim=X.shape[1])  # Initialize model
    train_model = True

#TODO: Modify Training Code ot Include Corrections
#IF training_model = false
if train_model:
    print("🚀 Starting model training...")

    #print(f"X dtype: {X.dtype}, y dtype: {df['label'].dtype}")

    # Convert to PyTorch tensors

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

    #define loss function and optimizer
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

    # Save trained model
    torch.save(model.state_dict(), "/workspaces/haimaidataorbit2025/test/saved_models/model.pth")

    # Save vectorizer
    with open("/workspaces/haimaidataorbit2025/test/saved_models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("✅ Model and vectorizer saved successfully!")
else:
    print("🚀 Using pre-trained model for predictions.")

#TODO: Modify Evaluation Code to Allow Manual Corrections (See ChatGPT)
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