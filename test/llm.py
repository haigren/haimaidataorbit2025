import os
from groq import Groq
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import os
import glob
import pickle
import csv
import pandas as pd
import re

client = Groq(
    # This is the default and can be omitted
    api_key=os.environ.get("GROQ_API_KEY"),
)

MODEL_PATH = "saved_models/model.pth"
VECTORIZER_PATH = "saved_models/vectorizer.pkl"


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

# Concatenate all datasets into one and load dataset
csv_files = glob.glob("HateSpeechData/*.csv")  # Adjust the path as needed
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

# Load all CSVs into a list of DataFrames
df_list = [pd.read_csv(file) for file in csv_files]

# Concatenate all DataFrames into one
df = pd.concat(df_list, ignore_index=True)

# Load your CSV

df['text'] = df['text'].fillna("")  # Fill empty text with ""

# Convert boolean labels to 0/1
df['label'] = df['label'].map({'hate': True, 'nothate': False, "1" : True, "0": False})

df['label'] = df['label'].fillna(0).astype(int)

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

if os.path.exists(VECTORIZER_PATH):
    print("🔄 Loading existing vectorizer...")
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
else:
    print("🚨 No vectorizer found. Creating new one...")
    exit(1)

X = vectorizer.transform(df['text']).toarray()

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
    exit(1)

new_text = ["you really really suck loser loser sucker"]
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

# weighs certain words harsher
word_map = {"loser": 0.1, "sucker": 0.05, "dumb": 0.1, "bitch": 0.1}

for word in new_text[0].split():
    if word in word_map.keys():
        probability += word_map[word]

if probability > 0.5:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f'Can you write rewrite this phrase? "{new_text}"'
            }
        ],
        model="llama-3.3-70b-versatile",
    )

# print(len(chat_completion.choices))
print(chat_completion.choices[0].message.content)

# Parse the quoted sentences into a list
parsed_list = re.findall(r'"(.*?)"', chat_completion.choices[0].message.content)

# Output the parsed list
print(parsed_list[0])
